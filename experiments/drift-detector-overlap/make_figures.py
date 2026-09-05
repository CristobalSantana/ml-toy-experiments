"""
make_figures.py -- Three figures, one per arm.

    python make_figures.py

  fig_degradation.png   the synthetic curve: where the detector stops working
  fig_real_vs_curve.png the taxi pairs dropped onto that curve
  fig_sample_size.png   the same shift through windows of growing size

Same palette as the other experiments in this repository, so a reader moving
between them is not also relearning the colours.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

BG = "#0e0e0e"
FG = "#f5f0e8"
ACCENT = "#0284C7"
GREY = "#8f8f8f"
WARN = "#e69f00"
RED = "#e05555"
# Three clearly separate hues. Three shades of the same blue were
# indistinguishable at figure size, which hid the whole point of the panel.
N_COLOURS = {500: "#7dd3fc", 5000: "#e69f00", 50000: "#4caf7d"}


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=GREY, labelsize=9)
    ax.set_title(title, color=FG, fontsize=11)
    ax.set_xlabel(xlabel, color=GREY, fontsize=9)
    ax.set_ylabel(ylabel, color=GREY, fontsize=9)
    ax.grid(True, color="#2a2a2a", lw=0.7)
    ax.set_axisbelow(True)


def degradation() -> None:
    """MCC against overlap, one line per window, one panel per feature count."""
    df = pd.read_csv(OUT / "synthetic.csv")
    dims = sorted(df["n_features"].unique())
    fig, axes = plt.subplots(1, len(dims), figsize=(12.5, 4.8), facecolor=BG,
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, d in zip(axes, dims):
        sub = df[df["n_features"] == d]
        for n, g in sub.groupby("n_per_side"):
            g = g.sort_values("overlap")
            ax.plot(g["overlap"], g["mcc"], "o-", ms=4, lw=2,
                    color=N_COLOURS.get(int(n), ACCENT), label=f"n = {int(n):,}")
        ax.axhline(0, color=GREY, lw=1, ls=":")
        label = ("1 feature, nothing to hide behind" if d == 1
                 else f"{d} features, the shift in one of them")
        _style(ax, label, "overlapping coefficient", "detector MCC")
        ax.set_ylim(-0.15, 1.05)
    axes[0].legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=9)

    fig.suptitle("Where the classifier two-sample test stops working",
                 color=FG, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_degradation.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_degradation.png")


def real_vs_curve() -> None:
    """The taxi pairs against the synthetic curve, on one comparable quantity.

    Both axes are held-out classifier accuracy - 0.5 is indistinguishable, 1.0
    is perfectly separable. An earlier version put synthetic MCC on the same
    axis as a rescaling of real accuracy; those are different quantities and
    the comparison was not one.

    x is 1 - overlap on a log scale. The five real pairs sit between 0.866 and
    0.988 overlap, which on a linear 0-1 axis is a pile-up in the last 15% of
    the plot with the labels on top of each other.
    """
    p = OUT / "real_pairs.csv"
    if not p.exists():
        return
    real = pd.read_csv(p)
    syn = pd.read_csv(OUT / "synthetic.csv")
    ref = syn[(syn["n_features"] == 8) & (syn["n_per_side"] == 50000)].copy()
    ref = ref[ref["overlap"] < 1.0].sort_values("overlap")

    fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor=BG)
    ax.plot(1 - ref["overlap"], ref["mean_accuracy_drifted"], "-", lw=2,
            color=ACCENT, label="synthetic, 8 features, n = 50,000")
    ax.axhline(0.5, color=GREY, lw=1, ls=":")
    ax.text(0.9, 0.505, "indistinguishable", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", color=GREY, fontsize=8.5)

    # pairs 0-2 sit within 0.007 of each other on x; stacking their labels
    # vertically put each one on the next marker, so they fan out instead
    offsets = [(-12, -6, "right"), (-10, 16, "right"), (14, -2, "left"),
               (0, 18, "center"), (0, 18, "center")]
    for (_, r), (dx, dy, ha) in zip(real.iterrows(), offsets):
        colour = GREY if r["is_null"] else (WARN if r["drift_called"] else RED)
        x = max(1 - r["mean_overlap"], 1e-4)
        ax.plot(x, r["accuracy"], "D", ms=10, color=colour, zorder=4)
        ax.annotate(f"{int(r['pair'])}. {r['name']}", xy=(x, r["accuracy"]),
                    xytext=(dx, dy), textcoords="offset points", ha=ha,
                    fontsize=9, color=FG)

    ax.set_xscale("log")
    _style(ax, "Does the synthetic curve predict real drift?",
           "distance from identical:  1 - mean overlapping coefficient  (log)",
           "held-out classifier accuracy")
    ax.set_ylim(0.44, 1.06)
    ax.set_xlim(6e-3, 1.6)
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_real_vs_curve.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_real_vs_curve.png")


def sample_size() -> None:
    """The count of 'significant' features against the size of the window."""
    p = OUT / "sample_size.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(9.5, 5.0), facecolor=BG)

    for arm, colour, label in (("drifted", WARN, "two adjacent months"),
                               ("control", ACCENT, "one month split in half")):
        g = df[df["arm"] == arm].sort_values("n_per_side")
        if g.empty:
            continue
        # Spread as an error bar at each point, not as connected min/max
        # lines: those cross each other between arms and read as four series
        # rather than two with a range.
        lo = g["mean_flagged"] - g["min_flagged"]
        hi = g["max_flagged"] - g["mean_flagged"]
        ax.errorbar(g["n_per_side"], g["mean_flagged"], yerr=[lo, hi],
                    fmt="o-", lw=2, ms=6, color=colour, label=label,
                    elinewidth=1.0, capsize=3, alpha=0.95)

    n_feat = int(df["n_features"].iloc[0])
    ax.axhline(n_feat, color=GREY, ls="--", lw=1.2)
    ax.text(df["n_per_side"].max(), n_feat + 0.12, "all features flagged",
            ha="right", va="bottom", color=GREY, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_ylim(-0.3, n_feat + 0.7)
    _style(ax, "The drift never changed. Only the window did.",
           "rows per side (log scale)",
           f"features called different, of {n_feat}")
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_sample_size.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_sample_size.png")


if __name__ == "__main__":
    degradation()
    real_vs_curve()
    sample_size()
    print(f"-> {OUT}")
