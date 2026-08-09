"""
make_figures.py -- Phase 8 figures, built from the saved result CSVs.

Reads only outputs/*.csv, so it is cheap to re-run while iterating on styling
and never retrains anything.

  fig_error_vs_cost.png       error against training time, both arms
  fig_drift_cutoff.png        full vs drift-cut training window
  fig_drift_auc.png           the detector's AUC over time
  (pdp_ale.png and shap_summary.png come from interpret.py)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
COLORS = {"ridge": "#888888", "random_forest": "#1f77b4", "lightgbm": "#2ca02c",
          "catboost": "#ff7f0e", "mlp": "#9467bd", "tabpfn": "#d62728"}


def error_vs_cost() -> None:
    df = pd.read_csv(OUTPUT_DIR / "cv_results.csv")
    df = df[df["mae"].notna()].copy()
    df["total_s"] = df["fit_seconds"] + df["predict_seconds"]
    g = df.groupby(["arm", "model"]).agg(
        mae=("mae", "mean"), sd=("mae", "std"), t=("total_s", "mean")).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, arm, title in [
        (axes[0], "full", "Full data (645k training rows)\nTabPFN cannot run here"),
        (axes[1], "regime_limited", "Regime-limited (~4.7k rows)\ninside TabPFN's CPU envelope"),
    ]:
        a = g[g["arm"] == arm]
        for _, r in a.iterrows():
            ax.errorbar(r["t"], r["mae"], yerr=r["sd"], fmt="o", ms=11,
                        color=COLORS.get(r["model"], "k"), capsize=4, alpha=0.9)
            ax.annotate(r["model"], (r["t"], r["mae"]), textcoords="offset points",
                        xytext=(9, 4), fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("total time: fit + predict (s, log scale)")
        ax.set_ylabel("MAE on log10(UF/m²)   — lower is better")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Error against cost: the comparison the experiment is about", fontsize=12)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "fig_error_vs_cost.png", dpi=150); plt.close(fig)


def drift_cutoff() -> None:
    p = OUTPUT_DIR / "drift_cutoff_test.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df = df[df["mae"].notna()]
    g = df.groupby(["model", "window"])["mae"].agg(["mean", "std"]).reset_index()
    piv = g.pivot(index="model", columns="window", values="mean").sort_values("full_2020")
    err = g.pivot(index="model", columns="window", values="std").reindex(piv.index)

    x = np.arange(len(piv)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, piv["full_2020"], w, yerr=err["full_2020"], capsize=3,
           label="train from 2020 (full window)", color="#c6c6c6", edgecolor="k", lw=0.5)
    ax.bar(x + w/2, piv["cut_2023"], w, yerr=err["cut_2023"], capsize=3,
           label="train from 2023 (drift-cut)", color="#1f77b4", edgecolor="k", lw=0.5)
    ax.set_xticks(x, piv.index, rotation=20)
    ax.set_ylabel("MAE on the untouched final 4 quarters")
    ax.set_title("Does cutting the drifted 2020-2022 span help?\n"
                 "Pre-registered hypothesis: yes. Result: yes for 5 of 6 models.", fontsize=11)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "fig_drift_cutoff.png", dpi=150); plt.close(fig)


def drift_auc() -> None:
    p = OUTPUT_DIR / "drift_multivariate.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(d["window"], d["auc"], "o-", lw=2, color="#d62728", label="classifier AUC")
    ax.fill_between(d["window"], d["ci_lo"], d["ci_hi"], alpha=0.2, color="#d62728",
                    label="95% bootstrap CI")
    ax.axhline(0.70, ls="--", c="k", lw=1, label="pre-registered threshold (0.70)")
    ax.axhline(0.50, ls=":", c="gray", lw=1, label="no drift (0.50)")
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("window compared against the frozen 2020 reference")
    ax.set_ylabel("two-sample classifier ROC-AUC")
    ax.set_title("Multivariate drift over time\n"
                 "Caveat: level_lag1 trends, so part of this is mechanical", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "fig_drift_auc.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    error_vs_cost(); drift_cutoff(); drift_auc()
    print(f"Wrote fig_*.png -> {OUTPUT_DIR}")
