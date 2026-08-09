"""
models.py -- The six architectures and the cost instrumentation.

Five classical models plus one tabular foundation model, each behind a uniform
fit/predict interface, with wall-clock time and peak memory measured on every
fit. CRITERIA.md makes cost part of the deliverable: a table of error alone
does not answer the question this experiment asks.

Preprocessing differs by model family, which is the point rather than an
inconvenience: LightGBM and CatBoost consume the categoricals and NaNs
natively, while Ridge, RF, MLP and TabPFN need them encoded and imputed. Each
model gets the representation it is actually designed for - handicapping one
with the wrong encoding would make the comparison meaningless.

On compute accounting: wall-clock and peak memory are genuinely measured.
FLOPs are reported only where they are honestly derivable from the algorithm
(Ridge's closed form, the MLP's forward/backward passes); for tree ensembles a
FLOP count is not a meaningful unit and is left as None rather than invented.
Energy is not measurable on this machine (no RAPL/NVML exposure on Windows),
and is recorded as such rather than estimated.
"""

from __future__ import annotations

import datetime as _dt
import threading
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
import psutil
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# TabPFN's own documented limits, read from tabpfn.inference_config
TABPFN_MAX_SAMPLES = 10_000
TABPFN_MAX_FEATURES = 500
# On CPU it enforces a second, stricter ceiling (tabpfn.validation): more than
# 5,000 training rows is "not allowed by default due to slow performance". We
# size the regime-limited arm under this rather than overriding it, so TabPFN
# runs in the configuration its authors support on this hardware.
TABPFN_CPU_MAX_SAMPLES = 5_000

TREE_NATIVE = {"lightgbm", "catboost"}   # consume categoricals/NaN directly
CLASSICAL = ["ridge", "random_forest", "lightgbm", "catboost", "mlp"]
ALL_MODELS = CLASSICAL + ["tabpfn"]


# --------------------------------------------------------------------------
# cost instrumentation
# --------------------------------------------------------------------------
class PeakMemory:
    """Sample this process's RSS in a background thread and report the peak
    increase over the pre-fit baseline.

    Process-wide RSS (not tracemalloc) because the libraries that matter here
    allocate in C/C++ where Python's allocator tracking is blind. Reported as a
    delta so the interpreter and already-loaded data are not charged to the
    model. Concurrent allocation elsewhere in the process would inflate it, so
    fits are run one at a time.
    """

    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self.baseline_mb = 0.0
        self.peak_mb = 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak_mb = max(self.peak_mb, self._proc.memory_info().rss / 1e6)
            except psutil.Error:
                break
            self._stop.wait(self.interval)

    def __enter__(self) -> "PeakMemory":
        self.baseline_mb = self._proc.memory_info().rss / 1e6
        self.peak_mb = self.baseline_mb
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    @property
    def delta_mb(self) -> float:
        return max(0.0, self.peak_mb - self.baseline_mb)


@dataclass
class FitResult:
    model: str
    arm: str                 # "full" or "regime_limited"
    seed: int
    fold: int
    n_train: int
    n_test: int
    mae: float               # primary metric, on the log10 target
    r2: float
    mdape_uf_m2: float       # median abs % error, back-transformed to UF/m²
    fit_seconds: float
    predict_seconds: float
    peak_memory_mb: float
    flops_estimate: float | None = None
    energy_joules: float | None = None   # not measurable on this machine
    # Wall-clock start of the fit. Recorded because timing is a headline metric
    # and CPU contention silently inflates it: with a timestamp, a contaminated
    # window is auditable after the fact instead of invisible.
    started_at: str = ""
    notes: str = ""

    def as_row(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------
def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split into (numeric, categorical) columns.

    Tested via pandas' own dtype predicate rather than a list of dtype names:
    pandas 3 returns the new `str` dtype from `astype(str)`, which a
    name-matching check on ("category", "object") silently misses - and a
    silently-empty categorical list makes CatBoost treat comuna names as
    numbers and crash.
    """
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    return num, cat


def dense_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """One-hot + median-impute + scale, for the models that need a dense
    numeric matrix (Ridge, RF, MLP, TabPFN).

    Median imputation is defensible here because the dominant NaN is
    structural, not missing-at-random: apartments have no land area of their
    own, and `es_departamento` carries that information explicitly, so the
    imputed value is never the model's only signal about it.
    """
    num, cat = split_columns(X)
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), num),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                       sparse_output=False))]), cat),
        ],
        remainder="drop",
    )


def prepare_for_model(name: str, X: pd.DataFrame) -> pd.DataFrame:
    """Tree models that handle categoricals natively get the frame as-is;
    CatBoost additionally needs its categorical cells to be strings."""
    if name == "catboost":
        out = X.copy()
        for c in split_columns(X)[1]:
            out[c] = out[c].astype(str).fillna("missing")
        return out
    return X


# --------------------------------------------------------------------------
# model construction (defaults, per CRITERIA.md's "competent engineer" baseline)
# --------------------------------------------------------------------------
def build_estimator(name: str, seed: int, X: pd.DataFrame, n_jobs: int = -1):
    if name == "ridge":
        return Pipeline([("prep", dense_preprocessor(X)), ("model", Ridge(random_state=seed))])

    if name == "random_forest":
        return Pipeline([("prep", dense_preprocessor(X)),
                         ("model", RandomForestRegressor(random_state=seed, n_jobs=n_jobs))])

    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(random_state=seed, n_jobs=n_jobs, verbose=-1)

    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_seed=seed, verbose=0, allow_writing_files=False,
                                 cat_features=split_columns(X)[1])

    if name == "mlp":
        return Pipeline([("prep", dense_preprocessor(X)),
                         ("model", MLPRegressor(random_state=seed))])

    if name == "tabpfn":
        from tabpfn import TabPFNRegressor
        return Pipeline([("prep", dense_preprocessor(X)),
                         ("model", TabPFNRegressor(random_state=seed, device="cpu"))])

    raise ValueError(f"unknown model: {name}")


def estimate_flops(name: str, n_train: int, n_features: int, estimator) -> float | None:
    """Approximate training FLOPs where the algorithm makes that a meaningful
    unit; None where it does not (tree ensembles are dominated by comparisons
    and sorts, not floating-point arithmetic, so a FLOP count would be
    theatre)."""
    if name == "ridge":
        # closed form: X^T X (n p^2) + Cholesky solve (p^3 / 3)
        p = n_features
        return float(n_train * p * p + (p ** 3) / 3)
    if name == "mlp":
        try:
            mlp = estimator.named_steps["model"]
            sizes = [n_features] + list(mlp.hidden_layer_sizes) + [1]
            per_sample_fwd = sum(2 * sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1))
            # backward ≈ 2x forward; n_iter_ epochs over the training set
            return float(3 * per_sample_fwd * n_train * getattr(mlp, "n_iter_", 0))
        except Exception:  # noqa: BLE001
            return None
    return None


# --------------------------------------------------------------------------
# a single instrumented fit/evaluate
# --------------------------------------------------------------------------
def fit_evaluate(
    name: str, X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, y_te: pd.Series,
    seed: int, fold: int, arm: str, n_jobs: int = -1, notes: str = "",
    keep_predictions: bool = False,
) -> FitResult | tuple[FitResult, np.ndarray]:
    """Fit, predict and measure. With keep_predictions the test-set predictions
    are returned alongside the result, so a caller that needs them (plots of
    predicted-vs-actual, residuals) does not have to refit the model."""
    started_at = _dt.datetime.now().isoformat(timespec="seconds")
    Xtr = prepare_for_model(name, X_tr)
    Xte = prepare_for_model(name, X_te)
    est = build_estimator(name, seed, Xtr, n_jobs=n_jobs)

    with PeakMemory() as mem:
        t0 = time.perf_counter()
        est.fit(Xtr, y_tr)
        fit_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        pred = est.predict(Xte)
        pred_s = time.perf_counter() - t1

    resid = np.asarray(y_te) - np.asarray(pred)
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((np.asarray(y_te) - np.mean(y_te)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    # back-transform to UF/m² for an interpretable percentage error
    true_uf, pred_uf = 10 ** np.asarray(y_te), 10 ** np.asarray(pred)
    mdape = float(np.median(np.abs(pred_uf - true_uf) / true_uf) * 100)

    result = FitResult(
        model=name, arm=arm, seed=seed, fold=fold,
        n_train=len(X_tr), n_test=len(X_te),
        mae=mae, r2=r2, mdape_uf_m2=mdape,
        fit_seconds=fit_s, predict_seconds=pred_s, peak_memory_mb=mem.delta_mb,
        flops_estimate=estimate_flops(name, len(X_tr), X_tr.shape[1], est),
        energy_joules=None,
        started_at=started_at,
        notes=notes,
    )
    return (result, np.asarray(pred)) if keep_predictions else result
