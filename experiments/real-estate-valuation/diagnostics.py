"""
diagnostics.py -- Feature correlation and collinearity, before any modelling.

Two questions this answers, both of which change how the results should be
read:

1. **Correlation** - which features move together, and which move with the
   target. A tree can use correlated features happily, but their individual
   importances stop being separable, so SHAP and partial dependence become
   harder to read.

2. **Collinearity (VIF)** - whether a feature is nearly a linear combination of
   the others. This is what makes Ridge coefficients unstable and swings
   attribution between near-duplicate columns. Variance Inflation Factor above
   ~5 is the usual warning line, above ~10 the usual red line.

Writes one figure with both panels plus a CSV, so the numbers behind the plot
stay checkable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

import features as F  # noqa: E402

OUTPUT_DIR = EXPERIMENT_ROOT / "outputs"
N_SAMPLE = 100_000


def vif(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor per feature: 1 / (1 - R²) from regressing that
    feature on all the others. Computed from the correlation matrix inverse,
    which is equivalent and far cheaper than fitting one regression per column.
    """
    Z = X.dropna()
    Z = Z.loc[:, Z.std() > 0]
    corr = np.corrcoef(Z.to_numpy(), rowvar=False)
    try:
        inv = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(corr)   # perfectly collinear -> pseudo-inverse
    return (pd.DataFrame({"feature": Z.columns, "vif": np.diag(inv)})
              .sort_values("vif", ascending=False))


def main() -> None:
    cfg = yaml.safe_load((EXPERIMENT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, groups, holdout, _ = F.build(cfg, verbose=False)
    dev = ~holdout
    # Diagnostics use development rows only: the holdout stays untouched even
    # for something as innocuous as a correlation matrix, because "look, then
    # decide" is exactly what the pre-registration forbids.
    Xd = X[dev].select_dtypes(include=[np.number])
    yd = y[dev]
    if len(Xd) > N_SAMPLE:
        idx = np.random.default_rng(cfg["seed"]).choice(len(Xd), N_SAMPLE, replace=False)
        Xd, yd = Xd.iloc[idx], yd.iloc[idx]

    frame = Xd.copy()
    frame[y.name] = yd
    corr = frame.corr(method="spearman")   # Spearman: monotone, no linearity assumed

    v = vif(Xd)
    v.to_csv(OUTPUT_DIR / "collinearity_vif.csv", index=False)
    corr.to_csv(OUTPUT_DIR / "feature_correlation.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    # --- correlation heatmap ---
    ax = axes[0]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(val) > 0.55 else "black")
    ax.set_title("Spearman correlation (target in the last row/column)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="ρ")

    # --- VIF ---
    ax = axes[1]
    colors = ["#d62728" if x >= 10 else "#ff7f0e" if x >= 5 else "#1f77b4" for x in v["vif"]]
    ax.barh(v["feature"], v["vif"], color=colors)
    # Log scale: an exactly-collinear pair produces a VIF around 1e15, which on
    # a linear axis flattens every other bar to invisibility.
    ax.set_xscale("log")
    ax.set_xlim(1, max(v["vif"].max() * 3, 100))
    ax.axvline(5, ls="--", c="gray", lw=1)
    ax.axvline(10, ls="--", c="red", lw=1)
    ax.set_xlabel("Variance Inflation Factor (log scale)")
    ax.set_title("Collinearity\n(dashed: 5 = watch, 10 = problem)", fontsize=10)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")
    for i, (f_, x_) in enumerate(zip(v["feature"], v["vif"])):
        if x_ >= 100:
            ax.text(x_, i, "  exactly collinear", va="center", fontsize=8, color="#d62728")

    fig.suptitle("Feature diagnostics before modelling", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_correlation_collinearity.png", dpi=150)
    plt.close(fig)

    print("Correlation with the target (Spearman):")
    tcorr = corr[y.name].drop(y.name).sort_values(key=abs, ascending=False)
    print(tcorr.round(3).to_string())
    print("\nTop VIF:")
    print(v.head(5).round(2).to_string(index=False))
    flagged = v[v["vif"] >= 5]
    print(f"\n{len(flagged)} feature(s) at VIF >= 5"
          + (f": {list(flagged['feature'])}" if len(flagged) else ""))
    print(f"\nWrote fig_correlation_collinearity.png, feature_correlation.csv, "
          f"collinearity_vif.csv -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
