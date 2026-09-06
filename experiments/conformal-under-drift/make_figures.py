"""
make_figures.py -- Three figures.

    python make_figures.py

  fig_coverage.png       what was promised against what was delivered
  fig_silent_failure.png the width never moved while the coverage collapsed
  fig_overlap.png        why watching the inputs would have ranked it backwards
"""

from __future__ import annotations

import json
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
GREEN = "#4caf7d"
RED = "#e05555"


def _style(ax, title="", xlabel="", ylabel=""):
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


def coverage() -> None:
    """Promised 90%, delivered anywhere from 90% to 15%."""
    c = pd.read_csv(OUT / "coverage.csv")
    fig, ax = plt.subplots(figsize=(11, 5.4), facecolor=BG)
    x = np.arange(len(c))
    w = 0.38

    b1 = ax.bar(x - w / 2, c["coverage_frozen"], w, color=ACCENT,
                label="calibrated once, on 2024-06")
    b2 = ax.bar(x + w / 2, c["coverage_recal"], w, color=GREEN, alpha=0.9,
                label="recalibrated on 2,000 fresh trips")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{b.get_height():.3f}", ha="center", color=FG, fontsize=8.5)

    ax.axhline(0.9, color=WARN, lw=2, ls="--")
    # below the line, not above: above it collides with the value labels of
    # the bars that reach 0.90
    ax.text(len(c) - 0.45, 0.845, "what the method guarantees", ha="right",
            color=WARN, fontsize=9.5)
    ax.set_xticks(x, [f"{m}\noverlap {o:.3f}" for m, o in
                      zip(c["month"], c["mean_overlap"])])
    for lbl in ax.get_xticklabels():
        lbl.set_color(FG)
    ax.set_ylim(0, 1.05)
    _style(ax, "A distribution-free guarantee, in a world that drifts",
           "", "share of true fares inside the interval")
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=9,
              loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_coverage.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_coverage.png")


def silent_failure() -> None:
    """The output looked identical the whole way down.

    This is the figure the experiment exists for. The interval width is a
    property of the calibration set, so it cannot move when the world does -
    which means nothing the method emits changes as the guarantee stops
    holding. A monitor watching interval widths would have seen a flat line.
    """
    c = pd.read_csv(OUT / "coverage.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor=BG, sharex=True)
    x = np.arange(len(c))

    axes[0].bar(x, c["width_frozen"], 0.55, color=GREY)
    for i, v in enumerate(c["width_frozen"]):
        axes[0].text(i, v + 0.06, f"${v:.2f}", ha="center", color=FG, fontsize=9)
    axes[0].set_ylim(0, max(c["width_frozen"]) * 1.35)
    _style(axes[0], "What the method reported: interval width", "",
           "mean interval width (dollars)")

    axes[1].bar(x, c["coverage_frozen"], 0.55,
                color=[GREEN if v >= 0.88 else RED for v in c["coverage_frozen"]])
    for i, v in enumerate(c["coverage_frozen"]):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", color=FG, fontsize=9)
    axes[1].axhline(0.9, color=WARN, lw=2, ls="--")
    axes[1].set_ylim(0, 1.08)
    _style(axes[1], "What was actually happening: coverage", "",
           "share of true fares inside the interval")

    for ax in axes:
        ax.set_xticks(x, c["month"])
        for lbl in ax.get_xticklabels():
            lbl.set_color(FG)

    fig.suptitle("The width never moved. The guarantee did.", color=FG, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_silent_failure.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_silent_failure.png")


def overlap() -> None:
    """Coverage against feature overlap - the ranking a drift monitor would give.

    If watching the inputs were enough, this would be a rising line. It is not:
    the month with the *better* feature overlap has the worse coverage.
    """
    c = pd.read_csv(OUT / "coverage.csv")
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=BG)

    ok = c["coverage_frozen"] >= 0.88
    ax.scatter(c.loc[ok, "mean_overlap"], c.loc[ok, "coverage_frozen"],
               s=150, color=GREEN, zorder=4, label="guarantee held")
    ax.scatter(c.loc[~ok, "mean_overlap"], c.loc[~ok, "coverage_frozen"],
               s=150, color=RED, zorder=4, label="guarantee failed")

    # The three surviving months sit within 0.01 of each other on x, so their
    # labels are fanned out by hand rather than stacked on top of one another.
    offsets = {"2024-06": (30, 16, "left"), "2024-05": (0, -42, "center"),
               "2023-06": (-30, 16, "right"), "2020-04": (12, 24, "left"),
               "2019-06": (-12, 24, "right")}
    for _, r in c.iterrows():
        dx, dy, ha = offsets.get(r["month"], (0, 20, "center"))
        ax.annotate(f"{r['month']}\nmean fare ${r['mean_fare']:.2f}",
                    xy=(r["mean_overlap"], r["coverage_frozen"]),
                    xytext=(dx, dy), textcoords="offset points", ha=ha,
                    fontsize=9, color=FG)

    ax.axhline(0.9, color=WARN, lw=1.6, ls="--")
    ax.annotate("", xy=(0.947, 0.20), xytext=(0.811, 0.30),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6, ls=":"))
    ax.text(0.879, 0.38, "more similar inputs,\nworse coverage",
            ha="center", color=RED, fontsize=9.5)

    ax.set_ylim(0, 1.12)
    ax.set_xlim(0.79, 1.005)
    _style(ax, "Feature overlap does not predict whether the guarantee holds",
           "mean overlapping coefficient with the calibration month",
           "coverage")
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=9,
              loc="center left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_overlap.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_overlap.png")


if __name__ == "__main__":
    coverage()
    silent_failure()
    overlap()
    print(f"-> {OUT}")
