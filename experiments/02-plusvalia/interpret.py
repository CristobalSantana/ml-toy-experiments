"""
interpret.py -- Phase 7: recover structure, not just error.

Three views of what the model actually learned:

1. Partial dependence and accumulated local effects (ALE) for the strongest
   features. PDP and ALE are both shown because they disagree when features
   are correlated: PDP averages over a marginal that includes combinations
   that never occur (a 20 m2 house in Las Condes), while ALE only ever
   compares nearby points. Where they diverge, ALE is the one to trust.

2. SHAP on the best tree model - with the caveat that a SHAP value is a local
   linear attribution, not the model. It says how this prediction was reached,
   not what the model is.

3. Symbolic regression (PySR) for an explicit closed form, reported with its
   error against the black-box model and its length. A forty-term expression
   that fits well and means nothing is itself the finding, and gets reported
   as such rather than quietly dropped.

Interpretability runs on a declared subsample: exhaustive SHAP over 645k rows
of a deep forest buys no insight that 20k rows do not already give, and PySR
is an evolutionary search that needs small data by design.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

import features as F  # noqa: E402
import models as M  # noqa: E402

OUTPUT_DIR = EXPERIMENT_ROOT / "outputs"
N_INTERPRET = 20_000     # rows for fitting the model we interpret
N_SHAP = 2_000           # rows explained (SHAP is quadratic-ish in practice)
N_SYMBOLIC = 5_000       # PySR is an evolutionary search; small by design
BEST_TREE_MODEL = "random_forest"   # lowest MAE in the full arm


def subsample(X, y, groups, n, seed=0):
    """Whole manzanas, so the interpreted sample keeps the grouping structure."""
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.sort(groups.unique()))
    sizes = groups.value_counts()
    take, total = [], 0
    for g in uniq:
        if total >= n:
            break
        take.append(g); total += int(sizes[g])
    m = groups.isin(set(take))
    return X[m], y[m], groups[m]


# --------------------------------------------------------------------------
# ALE
# --------------------------------------------------------------------------
def ale_1d(predict, X: pd.DataFrame, feature: str, bins: int = 20) -> pd.DataFrame:
    """First-order accumulated local effects.

    Within each quantile bin, replace the feature with the bin's edges and
    average the *difference* in prediction. Only local, in-support comparisons
    are made, which is what makes ALE safe under correlated features where PDP
    silently extrapolates.
    """
    x = X[feature].to_numpy(dtype=float)
    ok = np.isfinite(x)
    edges = np.unique(np.nanquantile(x[ok], np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return pd.DataFrame(columns=["x", "ale"])

    idx = np.clip(np.searchsorted(edges, x, side="left") - 1, 0, len(edges) - 2)
    deltas = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1)

    for b in range(len(edges) - 1):
        sel = (idx == b) & ok
        n = int(sel.sum())
        if n == 0:
            continue
        lo, hi = X[sel].copy(), X[sel].copy()
        lo[feature], hi[feature] = edges[b], edges[b + 1]
        deltas[b] = float(np.mean(predict(hi) - predict(lo)))
        counts[b] = n

    acc = np.cumsum(deltas)
    centers = (edges[:-1] + edges[1:]) / 2
    # centre so the weighted mean effect is zero (ALE is defined up to a constant)
    acc = acc - np.average(acc, weights=np.clip(counts, 1e-9, None))
    return pd.DataFrame({"x": centers, "ale": acc, "n": counts})


# --------------------------------------------------------------------------
def main() -> None:
    cfg = yaml.safe_load((EXPERIMENT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    X, y, groups, holdout, _ = F.build(cfg, verbose=False)
    dev = ~holdout
    Xs, ys, gs = subsample(X[dev], y[dev], groups[dev], N_INTERPRET, seed=cfg["seed"])
    print(f"interpretability sample: {len(Xs)} rows, {gs.nunique()} manzanas "
          f"(declared subsample of the {int(dev.sum())} development rows)")

    # ---- fit the model we interpret -------------------------------------
    est = M.build_estimator(BEST_TREE_MODEL, cfg["seed"], Xs)
    est.fit(Xs, ys)
    pred_fn = lambda d: est.predict(d)  # noqa: E731
    base_mae = float(np.mean(np.abs(ys - pred_fn(Xs))))
    print(f"{BEST_TREE_MODEL} in-sample MAE on this subsample: {base_mae:.4f}")

    # ---- 2. SHAP --------------------------------------------------------
    print("\nSHAP (tree explainer)")
    import shap
    prep = est.named_steps["prep"]
    inner = est.named_steps["model"]
    Xt = prep.transform(Xs.iloc[:N_SHAP])
    names = list(prep.get_feature_names_out())
    # SHAP over a deep forest costs several minutes; cache the raw values so a
    # later stage failing does not force recomputing them.
    cache = OUTPUT_DIR / "shap_values.npy"
    t0 = time.perf_counter()
    if cache.exists() and np.load(cache).shape == (len(Xt), len(names)):
        sv = np.load(cache)
        print("  reusing cached SHAP values")
    else:
        sv = shap.TreeExplainer(inner).shap_values(Xt, check_additivity=False)
        np.save(cache, sv)
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(axis=0)}) \
            .sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(OUTPUT_DIR / "shap_importance.csv", index=False)
    print(f"  {N_SHAP} rows explained in {time.perf_counter()-t0:.0f}s")
    print(imp.head(6).to_string(index=False))
    report["shap_top"] = imp.head(6).to_dict("records")

    fig = plt.figure(figsize=(8, 5))
    shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=10)
    plt.title("SHAP: local attributions, not the model itself", fontsize=10)
    plt.tight_layout(); fig.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150); plt.close(fig)

    # ---- 1. PDP + ALE for the strongest raw features --------------------
    raw_numeric = [c for c in Xs.columns if pd.api.types.is_numeric_dtype(Xs[c])]
    # Restrict to genuinely continuous features: a PDP/ALE curve over a binary
    # flag is two points and conveys nothing a group mean would not.
    continuous = [c for c in raw_numeric if Xs[c].nunique(dropna=True) > 10]
    ranked = [c for c in
              (imp["feature"].str.replace("num__", "", regex=False)
                             .str.replace("cat__", "", regex=False))
              if c in continuous]
    top = list(dict.fromkeys(ranked))[:3] or continuous[:3]
    print(f"\nPDP + ALE for: {top}  (continuous features only)")

    from sklearn.inspection import partial_dependence
    fig, axes = plt.subplots(1, len(top), figsize=(5 * len(top), 4))
    axes = np.atleast_1d(axes)
    ale_all = {}
    for ax, feat in zip(axes, top):
        panel = Xs.iloc[:5000]
        # PDP over rows where the feature actually exists. log10_sup_terreno_m2
        # is NaN for 86% of rows (apartments hold no land of their own), and a
        # percentile grid over a mostly-NaN column comes back all-NaN and plots
        # nothing at all. Restricting to houses also makes the curve honest:
        # it is a statement about properties that have land.
        pdp_rows = panel[panel[feat].notna()]
        # percentiles (0.01, 0.99) instead of sklearn's default (0.05, 0.95) so
        # the PDP spans nearly the same range as ALE - comparing two curves
        # drawn over different x-ranges is worse than not comparing them.
        pd_res = partial_dependence(est, pdp_rows, [feat], kind="average",
                                    grid_resolution=20, percentiles=(0.01, 0.99))
        gx = np.asarray(pd_res["grid_values"][0]); gy = np.asarray(pd_res["average"][0])
        ok = np.isfinite(gx) & np.isfinite(gy)
        if ok.sum() >= 2:
            ax.plot(gx[ok], gy[ok] - gy[ok].mean(), label="PDP (centred)", lw=2)
        else:
            print(f"  WARNING: PDP unavailable for {feat} (grid degenerate)")

        a = ale_1d(pred_fn, panel, feat)
        ale_all[feat] = a.to_dict("list")
        if not a.empty:
            ax.plot(a["x"], a["ale"], label="ALE", lw=2, ls="--")
        ax.set_xlabel(feat); ax.set_ylabel("effect on log10(UF/m²)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Partial dependence vs accumulated local effects")
    plt.tight_layout(); fig.savefig(OUTPUT_DIR / "pdp_ale.png", dpi=150); plt.close(fig)
    (OUTPUT_DIR / "ale_curves.json").write_text(json.dumps(ale_all, indent=2, default=float))

    # ---- 3. symbolic regression ----------------------------------------
    print("\nSymbolic regression (PySR)")
    from pysr import PySRRegressor
    # Held-out comparison, grouped by manzana. Scoring the expression against
    # the forest's *in-sample* error would flatter the forest enormously (its
    # in-sample MAE is ~0.004 against ~0.039 in cross-validation) and make the
    # ratio a statement about tree overfitting rather than about the
    # expression. Both are scored on manzanas neither of them was fit on.
    sym_pool = Xs[raw_numeric].iloc[:N_SYMBOLIC].fillna(Xs[raw_numeric].median())
    y_pool = ys.iloc[:N_SYMBOLIC]
    g_pool = gs.iloc[:N_SYMBOLIC]
    uniq_sym = np.sort(g_pool.unique())
    rng = np.random.default_rng(cfg["seed"]); rng.shuffle(uniq_sym)
    test_g = set(uniq_sym[: max(1, len(uniq_sym) // 4)])
    te_m = g_pool.isin(test_g)
    Xsym, ysym = sym_pool[~te_m], y_pool[~te_m]
    Xsym_te, ysym_te = sym_pool[te_m], y_pool[te_m]
    print(f"  symbolic fit on {len(Xsym)} rows, scored on {len(Xsym_te)} held-out rows")
    sr = PySRRegressor(
        niterations=40, binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "exp", "sqrt"],
        maxsize=25, populations=15, progress=False, verbosity=0,
        random_state=cfg["seed"], deterministic=True, parallelism="serial",
        temp_equation_file=True,
    )
    t0 = time.perf_counter()
    sr.fit(Xsym.to_numpy(), ysym.to_numpy(), variable_names=list(Xsym.columns))
    sym_s = time.perf_counter() - t0

    best = sr.get_best()
    expr = str(best["equation"])
    complexity = int(best["complexity"])

    # Apples-to-apples reference: a forest fit on exactly PySR's training rows
    # and the same numeric feature set, scored on the same held-out manzanas.
    # Reusing the earlier forest would score it on rows it was trained on, and
    # its in-sample MAE (~0.003) is an order of magnitude below its honest
    # cross-validated error - the ratio would then measure tree overfitting,
    # not how much structure the expression gives up.
    ref = M.build_estimator(BEST_TREE_MODEL, cfg["seed"], Xsym)
    ref.fit(Xsym, ysym)
    sym_mae = float(np.mean(np.abs(ysym_te - sr.predict(Xsym_te.to_numpy()))))
    bb_mae = float(np.mean(np.abs(ysym_te - ref.predict(Xsym_te))))

    print(f"  searched in {sym_s:.0f}s")
    print(f"  expression (complexity {complexity}): {expr}")
    print(f"  held-out MAE: symbolic {sym_mae:.4f} vs {BEST_TREE_MODEL} {bb_mae:.4f} "
          f"({sym_mae/bb_mae:.2f}x)")
    readable = complexity <= 15
    print(f"  short enough to read? {'yes' if readable else 'NO - reported as the '
          'informative negative result it is'}")

    sr.equations_.to_csv(OUTPUT_DIR / "symbolic_equations.csv", index=False)
    report["symbolic"] = {"expression": expr, "complexity": complexity,
                          "mae": sym_mae, "blackbox_mae": bb_mae,
                          "ratio": sym_mae / bb_mae, "readable": readable,
                          "search_seconds": sym_s, "n_rows_fit": int(len(Xsym)), "n_rows_scored": int(len(Xsym_te))}
    report["interpret_sample"] = {"n_rows": int(len(Xs)), "n_manzanas": int(gs.nunique()),
                                  "model": BEST_TREE_MODEL, "in_sample_mae": base_mae}
    (OUTPUT_DIR / "interpretability.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nWrote shap_*, pdp_ale.png, symbolic_equations.csv, interpretability.json -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
