"""
make_figures.py -- The result figures, built from the saved CSVs.

Reads only outputs/*.csv, so it is cheap to re-run while iterating on styling
and never retrains anything.

Deliberately few figures, each answering a different question:

  fig_model_comparison.png     which model, and at what cost
  fig_predicted_vs_actual.png  does the best model track reality
  fig_errors.png               how the error is distributed and where it lands
  fig_drift.png                did the distribution move, and did cutting help

(`fig_correlation_collinearity.png` comes from diagnostics.py, and the
interpretability plots from interpret.py.)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
COLORS = {"ridge": "#8c8c8c", "random_forest": "#1f77b4", "lightgbm": "#2ca02c",
          "catboost": "#ff7f0e", "mlp": "#9467bd", "tabpfn": "#d62728"}


def model_comparison() -> None:
    """Error against cost, in both arms - the comparison the experiment is about."""
    df = pd.read_csv(OUTPUT_DIR / "cv_results.csv")
    df = df[df["mae"].notna()].copy()
    df["total_s"] = df["fit_seconds"] + df["predict_seconds"]
    g = df.groupby(["arm", "model"]).agg(
        mae=("mae", "mean"), sd=("mae", "std"), t=("total_s", "mean"),
        mem=("peak_memory_mb", "mean")).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for ax, arm, title in [
        (axes[0], "full", "Full data (645k training rows)\nTabPFN cannot run at this size"),
        (axes[1], "regime_limited", "Regime-limited (~4.7k rows)\ninside TabPFN's CPU envelope"),
    ]:
        a = g[g["arm"] == arm]
        for _, r in a.iterrows():
            size = 60 + 240 * (r["mem"] / max(g["mem"].max(), 1e-9)) ** 0.5
            ax.scatter(r["t"], r["mae"], s=size, color=COLORS.get(r["model"], "k"),
                       alpha=0.85, edgecolor="k", lw=0.6, zorder=3)
            ax.errorbar(r["t"], r["mae"], yerr=r["sd"], fmt="none",
                        ecolor=COLORS.get(r["model"], "k"), capsize=4, alpha=0.7)
            ax.annotate(r["model"], (r["t"], r["mae"]), textcoords="offset points",
                        xytext=(11, 5), fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("total time: fit + predict (s, log scale)")
        ax.set_ylabel("MAE on log10(UF/m²)  —  lower is better")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Error against cost  (marker size = peak memory; bars = ±1 sd over folds)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "fig_model_comparison.png", dpi=150)
    plt.close(fig)


def predicted_vs_actual() -> None:
    """Best model's predictions against the truth, on the frozen holdout."""
    p = OUTPUT_DIR / "final_holdout_predictions.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    summary = pd.read_csv(OUTPUT_DIR / "final_holdout_summary.csv")
    best = summary.sort_values("mae").iloc[0]["model"]
    col = f"pred_{best}"
    if col not in d:
        return

    actual = 10 ** d["actual_log10"].to_numpy()
    pred = 10 ** d[col].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    ax = axes[0]
    ax.hexbin(actual, pred, gridsize=60, bins="log", cmap="Blues", mincnt=1)
    lim = (min(actual.min(), pred.min()), np.percentile(actual, 99.9))
    ax.plot(lim, lim, "r--", lw=1.5, label="perfect prediction")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("actual assessed value (UF/m²)")
    ax.set_ylabel("predicted (UF/m²)")
    ax.set_title(f"{best} on the frozen holdout\n{len(d):,} properties never seen in training",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    pct = 100 * (pred - actual) / actual
    ax.hexbin(actual, pct, gridsize=60, bins="log", cmap="Blues", mincnt=1)
    ax.axhline(0, c="r", ls="--", lw=1.5)
    ax.set_xlim(lim); ax.set_ylim(-60, 60)
    ax.set_xlabel("actual assessed value (UF/m²)")
    ax.set_ylabel("relative error (%)")
    ax.set_title("Error against price level\n(is the model worse for cheap or expensive property?)",
                 fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("Predicted vs actual, and where the error sits", fontsize=12)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "fig_predicted_vs_actual.png", dpi=150)
    plt.close(fig)


def errors() -> None:
    """Residual distributions per model, plus error by comuna for the best one."""
    p = OUTPUT_DIR / "final_holdout_predictions.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    models = [c[5:] for c in d.columns if c.startswith("pred_")]
    if not models:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    data = [(10 ** d[f"pred_{m}"] - 10 ** d["actual_log10"]) / 10 ** d["actual_log10"] * 100
            for m in models]
    order = np.argsort([np.median(np.abs(x)) for x in data])
    bp = ax.boxplot([data[i] for i in order], tick_labels=[models[i] for i in order],
                    showfliers=False, patch_artist=True)
    for patch, i in zip(bp["boxes"], order):
        patch.set_facecolor(COLORS.get(models[i], "#cccccc")); patch.set_alpha(0.75)
    ax.axhline(0, c="r", ls="--", lw=1)
    ax.set_ylabel("relative error (%)")
    ax.set_title("Error distribution per model (holdout)\nsorted by median absolute error",
                 fontsize=10)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    summary = pd.read_csv(OUTPUT_DIR / "final_holdout_summary.csv")
    best = summary.sort_values("mae").iloc[0]["model"]
    err = np.abs(d[f"pred_{best}"] - d["actual_log10"])
    ax.hist(err, bins=60, color=COLORS.get(best, "#1f77b4"), alpha=0.85)
    ax.set_xlim(0, err.quantile(0.999))
    top = ax.get_ylim()[1]
    for q in (0.5, 0.9, 0.99):
        v = err.quantile(q)
        ax.axvline(v, ls="--", lw=1.2, c="k", alpha=0.7)
        # 0.72 of the axis height, not 0.92: at 0.92 the rotated labels ran into
        # the title.
        ax.text(v, top * 0.72, f" p{int(q*100)}={v:.3f}", fontsize=8, rotation=90)
    ax.set_xlabel("absolute error on log10(UF/m²)")
    ax.set_ylabel("properties")
    ax.set_title(f"{best}: absolute error distribution\nthe long tail is where valuation fails",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("How the error is distributed", fontsize=12)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "fig_errors.png", dpi=150)
    plt.close(fig)


def drift() -> None:
    """One figure for the whole drift question: did it move, and did cutting help."""
    mp, cp = OUTPUT_DIR / "drift_multivariate.csv", OUTPUT_DIR / "drift_cutoff_test.csv"
    if not (mp.exists() and cp.exists()):
        return
    m, c = pd.read_csv(mp), pd.read_csv(cp)
    c = c[c["mae"].notna()]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    ax = axes[0]
    ax.plot(m["window"], m["auc"], "o-", lw=2, color="#d62728", label="detector AUC")
    ax.fill_between(m["window"], m["ci_lo"], m["ci_hi"], alpha=0.2, color="#d62728",
                    label="95% bootstrap CI")
    ax.axhline(0.70, ls="--", c="k", lw=1, label="pre-registered threshold")
    ax.axhline(0.50, ls=":", c="gray", lw=1, label="no drift")
    ax.set_ylim(0.4, 1.03)
    ax.set_xlabel("window vs the frozen 2020 reference")
    ax.set_ylabel("two-sample classifier ROC-AUC")
    ax.set_title("Did the distribution move?\nEvery window drifts (part of it is trend - see README)",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    g = c.groupby(["model", "window"])["mae"].agg(["mean", "std"]).reset_index()
    piv = g.pivot(index="model", columns="window", values="mean").sort_values("full_2020")
    err = g.pivot(index="model", columns="window", values="std").reindex(piv.index)
    x = np.arange(len(piv)); w = 0.38
    ax.bar(x - w/2, piv["full_2020"], w, yerr=err["full_2020"], capsize=3,
           label="train from 2020 (full)", color="#c9c9c9", edgecolor="k", lw=0.5)
    ax.bar(x + w/2, piv["cut_2023"], w, yerr=err["cut_2023"], capsize=3,
           label="train from 2023 (cut)", color="#1f77b4", edgecolor="k", lw=0.5)
    ax.set_xticks(x, piv.index, rotation=20)
    ax.set_ylabel("MAE on the untouched final 4 quarters")
    ax.set_title("Did cutting the drifted span help?\nPre-registered hypothesis, tested not assumed",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Distribution drift in the housing price index", fontsize=12)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "fig_drift.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    model_comparison(); predicted_vs_actual(); errors(); drift()
    print(f"Wrote fig_*.png -> {OUTPUT_DIR}")
