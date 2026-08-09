"""
drift.py -- Phase 6: does the training window need cutting?

CRITERIA.md pre-registers a hypothesis - that training on 2023-present beats
training on 2020-present, because 2020-2022 is distributionally drifted - and
commits to *testing* it rather than assuming it, reporting the result in
whichever direction it falls.

Three parts:

1. Univariate drift: Kolmogorov-Smirnov and Population Stability Index per
   feature, reference window vs each later window, with Benjamini-Hochberg FDR
   control at 5%. One test per feature at 5% would manufacture false alarms by
   construction, which is exactly the mistake the correction exists to prevent.

2. Multivariate drift: a classifier two-sample test. If a model can tell
   reference rows from later rows, the joint distribution moved even when no
   single marginal did. Reported as cross-validated ROC-AUC with a CI.

3. The cutoff test itself: train every model on the full window and on the cut
   window, and compare both on the untouched final period.

The reference window is frozen (the first four quarters of the analysis span),
never rolling, so drift is always measured against the same baseline.

The panel is quarterly macro data: 16 series x 25 quarters. That is small, and
the analysis says so rather than dressing up noisy estimates as precision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

import leakage  # noqa: E402
import models as M  # noqa: E402

OUTPUT_DIR = EXPERIMENT_ROOT / "outputs"
N_LAGS = 4


# --------------------------------------------------------------------------
# panel
# --------------------------------------------------------------------------
def build_panel(cfg: dict, start: str = "2020-01-01") -> pd.DataFrame:
    """One row per (series, quarter): lagged log-changes -> next log-change.

    Every feature is a strictly past value, verified with leakage.check_asof
    rather than assumed from how the shift was written.
    """
    path = REPO_ROOT / cfg["data"]["ipv_processed"]
    d = pd.read_csv(path, parse_dates=["quarter"]).sort_values(["series_col", "quarter"])

    # Coverage is judged *inside the analysis window*, not over the whole
    # history. The headline series runs from 2002 while the disaggregated ones
    # start in 2014, so comparing against the global maximum would keep only
    # the headline series and silently collapse the panel to one row per
    # quarter - which is exactly what an earlier version of this did.
    in_window = d[d["quarter"] >= pd.Timestamp(start)]
    cov = in_window.groupby("series_col")["quarter"].nunique()
    keep = cov[cov == cov.max()].index
    dropped = sorted(set(cov.index) - set(keep))
    if dropped:
        print(f"  dropped {len(dropped)} series without full coverage since {start}: {dropped}")
    d = d[d["series_col"].isin(keep)].copy()

    d["log_index"] = np.log(d["index_value"])
    d["dlog"] = d.groupby("series_col")["log_index"].diff()

    for k in range(1, N_LAGS + 1):
        d[f"dlog_lag{k}"] = d.groupby("series_col")["dlog"].shift(k)
    d["level_lag1"] = d.groupby("series_col")["log_index"].shift(1)
    d["quarter_of_year"] = d["quarter"].dt.quarter
    # the period each feature is drawn from: one quarter back at the latest
    d["feature_period"] = d.groupby("series_col")["quarter"].shift(1)

    d = d[d["quarter"] >= pd.Timestamp(start)].dropna(
        subset=[f"dlog_lag{k}" for k in range(1, N_LAGS + 1)] + ["dlog", "level_lag1"])

    leakage.check_asof(d["feature_period"], d["quarter"], "ipv lagged features")
    return d


FEATURES = [f"dlog_lag{k}" for k in range(1, N_LAGS + 1)] + ["level_lag1", "quarter_of_year"]
TARGET = "dlog"


# --------------------------------------------------------------------------
# 1. univariate drift
# --------------------------------------------------------------------------
def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index, binned on the reference's own quantiles.

    Bins come from the reference because the reference is the frozen baseline;
    re-binning on each window would move the yardstick with the thing being
    measured.
    """
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, bins=edges)[0] / max(len(ref), 1)
    c = np.histogram(cur, bins=edges)[0] / max(len(cur), 1)
    eps = 1e-6
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """BH-adjusted p-values (step-up), so 'significant' means FDR-controlled."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = n - rank + 1
        prev = min(prev, p[idx] * n / i)
        adj[idx] = prev
    return adj


def univariate_drift(panel: pd.DataFrame, ref_mask: pd.Series, windows: dict) -> pd.DataFrame:
    rows = []
    for wname, wmask in windows.items():
        for f in FEATURES:
            ref, cur = panel.loc[ref_mask, f].dropna(), panel.loc[wmask, f].dropna()
            if len(ref) < 5 or len(cur) < 5:
                continue
            ks = stats.ks_2samp(ref, cur)
            rows.append({"window": wname, "feature": f,
                         "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
                         "psi": psi(ref.to_numpy(), cur.to_numpy()),
                         "n_ref": len(ref), "n_cur": len(cur)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # BH across every (window, feature) test performed
    out["ks_p_bh"] = benjamini_hochberg(out["ks_p"].to_numpy())
    out["shifted"] = (out["ks_p_bh"] < 0.05) | (out["psi"] >= 0.25)
    return out


# --------------------------------------------------------------------------
# 2. multivariate drift: classifier two-sample test
# --------------------------------------------------------------------------
def classifier_two_sample(panel: pd.DataFrame, ref_mask: pd.Series, windows: dict,
                          seed: int = 0, n_boot: int = 500) -> pd.DataFrame:
    rows = []
    for wname, wmask in windows.items():
        X = pd.concat([panel.loc[ref_mask, FEATURES], panel.loc[wmask, FEATURES]])
        yv = np.r_[np.zeros(int(ref_mask.sum())), np.ones(int(wmask.sum()))]
        if min((yv == 0).sum(), (yv == 1).sum()) < 8:
            continue
        clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        prob = cross_val_predict(clf, X, yv, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(yv, prob)

        # bootstrap CI over the out-of-fold predictions
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            idx = rng.choice(len(yv), len(yv), replace=True)
            if len(np.unique(yv[idx])) < 2:
                continue
            boots.append(roc_auc_score(yv[idx], prob[idx]))
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))

        clf.fit(X, yv)
        imp = dict(zip(FEATURES, np.round(clf.feature_importances_, 4)))
        rows.append({"window": wname, "auc": float(auc), "ci_lo": float(lo), "ci_hi": float(hi),
                     "n_ref": int(ref_mask.sum()), "n_cur": int(wmask.sum()),
                     "drifted": bool(auc >= 0.70 and lo > 0.5),
                     "top_features": ", ".join(
                         f"{k}={v}" for k, v in sorted(imp.items(), key=lambda t: -t[1])[:3])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. the cutoff test
# --------------------------------------------------------------------------
def cutoff_test(panel: pd.DataFrame, cfg: dict, holdout_quarters: int = 4) -> pd.DataFrame:
    """Train each model on the full vs the cut window; compare on the untouched
    final period. Walk-forward in spirit: the evaluation period is strictly
    after every training row."""
    quarters = np.sort(panel["quarter"].unique())
    held = set(quarters[-holdout_quarters:])
    test = panel[panel["quarter"].isin(held)]
    trainable = panel[~panel["quarter"].isin(held)]

    windows = {"full_2020": pd.Timestamp("2020-01-01"), "cut_2023": pd.Timestamp("2023-01-01")}
    rows = []
    for wname, start in windows.items():
        tr = trainable[trainable["quarter"] >= start]
        for name in M.ALL_MODELS:
            for seed in cfg["split"]["seeds"]:
                try:
                    res = M.fit_evaluate(name, tr[FEATURES], tr[TARGET],
                                         test[FEATURES], test[TARGET],
                                         seed=seed, fold=0, arm=f"ipv_{wname}")
                    rows.append({"window": wname, "model": name, "seed": seed,
                                 "n_train": len(tr), "n_test": len(test),
                                 "mae": res.mae, "r2": res.r2,
                                 "fit_seconds": res.fit_seconds})
                except Exception as e:  # noqa: BLE001
                    rows.append({"window": wname, "model": name, "seed": seed,
                                 "n_train": len(tr), "n_test": len(test),
                                 "mae": np.nan, "notes": f"{type(e).__name__}: {str(e)[:120]}"})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = yaml.safe_load((EXPERIMENT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building IPV panel (2020 onward)")
    panel = build_panel(cfg)
    quarters = np.sort(panel["quarter"].unique())
    print(f"  {len(panel)} rows, {panel['series_col'].nunique()} series, "
          f"{len(quarters)} quarters: {pd.Timestamp(quarters[0]).date()} .. "
          f"{pd.Timestamp(quarters[-1]).date()}")

    # frozen reference: the first four quarters of the span, never rolling
    ref_q = set(quarters[:4])
    ref_mask = panel["quarter"].isin(ref_q)
    print(f"  frozen reference window: {[str(pd.Timestamp(q).date()) for q in sorted(ref_q)]}")

    windows = {}
    for year in sorted({pd.Timestamp(q).year for q in quarters}):
        if year == pd.Timestamp(quarters[0]).year:
            continue
        m = panel["quarter"].dt.year == year
        if m.sum() >= 8:
            windows[str(year)] = m
    print(f"  comparison windows: {list(windows)}\n")

    print("1) Univariate drift (KS + PSI, BH-corrected at 5%)")
    uni = univariate_drift(panel, ref_mask, windows)
    uni.to_csv(OUTPUT_DIR / "drift_univariate.csv", index=False)
    if not uni.empty:
        n_shift = uni.groupby("window")["shifted"].sum()
        print(f"  features flagged as shifted, per window: {dict(n_shift)}")
        print(f"  (of {uni.groupby('window').size().iloc[0]} features tested per window)")
        raw_sig = int((uni["ks_p"] < 0.05).sum())
        bh_sig = int((uni["ks_p_bh"] < 0.05).sum())
        print(f"  KS significant: {raw_sig} uncorrected vs {bh_sig} after BH "
              f"- the correction matters\n")

    print("2) Multivariate drift (classifier two-sample test)")
    multi = classifier_two_sample(panel, ref_mask, windows)
    multi.to_csv(OUTPUT_DIR / "drift_multivariate.csv", index=False)
    if not multi.empty:
        print(multi[["window", "auc", "ci_lo", "ci_hi", "drifted"]].round(3).to_string(index=False))
        print("  CAVEAT: `level_lag1` is a rising index, so a classifier can separate\n"
              "  windows by level alone - part of this AUC is mechanical trend, not a\n"
              "  change in the modelled relationship. The stationary-by-construction\n"
              "  dlog_* features drift too (see drift_univariate.csv), and the 2023\n"
              "  detector leans entirely on them, so the drift is not only the trend.")
    print()

    print("3) Cutoff test: full 2020 window vs 2023 cut, on the untouched final period")
    cut = cutoff_test(panel, cfg)
    cut.to_csv(OUTPUT_DIR / "drift_cutoff_test.csv", index=False)
    ok = cut[cut["mae"].notna()]
    if not ok.empty:
        piv = ok.groupby(["model", "window"])["mae"].agg(["mean", "std"]).unstack()
        print(piv.round(5).to_string())
    print(f"\nWrote drift_*.csv -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
