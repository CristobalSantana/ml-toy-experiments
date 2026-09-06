"""
make_figures.py -- Three figures.

    python make_figures.py

  fig_cost_of_accuracy.png  error against operations, everything on one plane
  fig_sweep.png             what more timesteps buy, and what they cost
  fig_accounting.png        the same two networks under two conventions
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
GREY = "#8f8f8f"
ACCENT = "#0284C7"      # the dense network
WARN = "#e69f00"        # direct coding
RED = "#e05555"         # rate coding
GREEN = "#4caf7d"       # the cheap classical references
VIOLET = "#9a7fd1"


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


def _load():
    sw = pd.read_csv(OUT / "sweep.csv")
    base = pd.read_csv(OUT / "baselines.csv")
    acc = json.loads((OUT / "accounting.json").read_text(encoding="utf-8"))
    be = (pd.read_csv(OUT / "best_effort.csv")
          if (OUT / "best_effort.csv").exists() else None)
    med = (sw.groupby(["encoding", "n_steps"]).median(numeric_only=True)
           .reset_index())
    return sw, base, acc, med, be


def cost_of_accuracy() -> None:
    """Error against operations. Down and left is better.

    Both axes at once, because a claim about energy is only interesting at a
    given accuracy, and a claim about accuracy is only interesting at a given
    cost. Putting them on one plane means neither can be quoted alone.
    """
    _, base, acc, med, be = _load()
    fig, ax = plt.subplots(figsize=(11, 6.0), facecolor=BG)

    for enc, colour in (("direct", WARN), ("rate", RED)):
        g = med[med["encoding"] == enc].sort_values("n_steps")
        ax.plot(g["ops_total"], g["rmse"], "o-", lw=1.8, ms=6, color=colour,
                label=f"spiking, {enc} coding")
        for _, r in g.iterrows():
            ax.annotate(f"T={int(r['n_steps'])}",
                        (r["ops_total"], r["rmse"]), textcoords="offset points",
                        xytext=(0, 9), ha="center", color=colour, fontsize=8)

    mlp = base[base["model"] == "mlp"].iloc[0]
    ax.scatter(mlp["ops_total"], mlp["rmse"], s=220, marker="*", color=ACCENT,
               zorder=5, label="dense network (same width)")
    ridge = base[base["model"] == "ridge"].iloc[0]
    ax.scatter(ridge["ops_total"], ridge["rmse"], s=150, marker="D",
               color=GREEN, zorder=5, label="ridge regression")

    if be is not None:
        b = be.median(numeric_only=True)
        ax.scatter(b["ops_total"], b["rmse"], s=150, marker="P", color=VIOLET,
                   zorder=5, label="spiking, best effort (post-hoc)")

    sp = base[base["model"] == "smart_persistence"].iloc[0]["rmse"]
    ax.axhline(sp, color=GREY, lw=1.3, ls="--")
    # in axes coordinates, so the label cannot run off the right edge
    ax.text(0.985, sp, "smart persistence, which costs nothing",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            color=GREY, fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    # not "up and to the right": T=1 sits marginally left of the dense
    # network at 900 operations against 960, and the title should not
    # overstate what the figure shows
    _style(ax, "More error at every setting, and more operations at all but one",
           "operations per prediction (log scale)",
           "daylight RMSE (log scale, lower is better)")
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=9,
              loc="upper right")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(FG)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cost_of_accuracy.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_cost_of_accuracy.png")


def sweep() -> None:
    """What more timesteps buy on the left, what they cost on the right."""
    _, base, acc, med, _ = _load()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor=BG)
    mlp = base[base["model"] == "mlp"].iloc[0]

    # log y: rate coding is an order of magnitude worse than direct, and a
    # linear axis would flatten the direct curve into a straight line
    for enc, colour in (("direct", WARN), ("rate", RED)):
        g = med[med["encoding"] == enc].sort_values("n_steps")
        axes[0].plot(g["n_steps"], g["rmse"], "o-", lw=2, ms=6, color=colour,
                     label=f"{enc} coding")
    # the dense network and P2's 2% tolerance are within a hair of each other,
    # so they are drawn as one band rather than two colliding lines
    axes[0].axhspan(mlp["rmse"], mlp["rmse"] * 1.02, color=GREEN, alpha=0.45,
                    lw=0)
    axes[0].axhline(mlp["rmse"], color=ACCENT, lw=2, ls="--")
    axes[0].annotate("dense network, and the 2% band\nP2 asked to be reached "
                     "by T=16",
                     xy=(4, mlp["rmse"]), xytext=(0, -34),
                     textcoords="offset points", ha="center", color=ACCENT,
                     fontsize=9)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_ylim(mlp["rmse"] * 0.55, med["rmse"].max() * 1.5)
    _style(axes[0], "No T came close, and rate coding was never in the running",
           "timesteps per prediction (T)", "daylight RMSE (log scale)")
    axes[0].legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=9,
                   loc="center right")

    # lines, not stacked bars: on a log axis a bar's length means nothing,
    # and these two quantities differ by more than two orders of magnitude
    g = med[med["encoding"] == "direct"].sort_values("n_steps")
    axes[1].plot(g["n_steps"], g["ops_input"], "o-", lw=2.2, ms=6, color=GREY,
                 label="input layer (dense, on every timestep)")
    axes[1].plot(g["n_steps"], g["ops_hidden"], "s-", lw=2.2, ms=6, color=WARN,
                 label="hidden layer (sparse, event-driven)")
    axes[1].axhline(mlp["ops_total"], color=ACCENT, lw=2, ls="--")
    axes[1].text(1.05, mlp["ops_total"] * 1.25,
                 f"the whole dense network: {mlp['ops_total']:,.0f}",
                 color=ACCENT, fontsize=9)
    share = g["ops_input"].iloc[-1] / g["ops_total"].iloc[-1]
    axes[1].annotate(f"the sparse layer is {1 - share:.1%} of the total",
                     xy=(g["n_steps"].iloc[-1], g["ops_hidden"].iloc[-1]),
                     xytext=(-10, -30), textcoords="offset points", ha="right",
                     color=WARN, fontsize=9)
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    # headroom, so the legend sits in empty space rather than on the last point
    axes[1].set_ylim(g["ops_hidden"].min() * 0.5, g["ops_input"].max() * 12)
    _style(axes[1], "and the part that is not sparse is the part that grows",
           "timesteps per prediction (T)",
           "operations per prediction (log scale)")
    axes[1].legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=8.5,
                   loc="upper left")

    for ax in axes:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(FG)

    fig.suptitle("Direct coding: the first layer is dense on every one of the "
                 "T timesteps", color=FG, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sweep.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_sweep.png")


def accounting() -> None:
    """Cost relative to the dense network, under both conventions, at every T.

    Drawn as ratios across the whole sweep rather than as bars at one chosen
    T, because the two conventions do not disagree everywhere - they disagree
    over a range, and which range is the finding. Picking a single T would
    have let the figure say whatever that T happened to say.
    """
    _, base, acc, med, _ = _load()
    mlp = base[base["model"] == "mlp"].iloc[0]
    mlp_hidden = acc["mlp_ops_hidden"]

    fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=BG)
    for enc, ls, mk in (("direct", "-", "o"), ("rate", "--", "s")):
        g = med[med["encoding"] == enc].sort_values("n_steps")
        ax.plot(g["n_steps"], g["ops_total"] / mlp["ops_total"], ls, marker=mk,
                lw=2.2, ms=6, color=GREY,
                label=f"counting every operation, {enc} coding")
        ax.plot(g["n_steps"], g["ops_hidden"] / mlp_hidden, ls, marker=mk,
                lw=2.2, ms=6, color=WARN,
                label=f"counting only the sparse layer, {enc} coding")

    ax.axhline(1.0, color=ACCENT, lw=2)
    # placed in gaps between the four series rather than at a fixed corner
    ax.text(31, 1.15, "parity with the dense network", ha="right",
            color=ACCENT, fontsize=9.5)
    ax.text(8, 0.055, "spiking cheaper", ha="center", color=GREEN, fontsize=10)
    ax.text(1.05, 3.2, "spiking more expensive", color=RED, fontsize=10)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _style(ax, "The same networks. Two ways of counting. Answers on opposite "
               "sides of parity.",
           "timesteps per prediction (T)",
           "operations relative to the dense network (log scale)")
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=8.5,
              loc="upper left", ncol=2)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(FG)
    fig.text(0.5, 0.015,
             "Not a matched comparison at any point on this axis: no spiking "
             "configuration tested reached the dense network's accuracy.",
             ha="center", color=GREY, fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(OUT / "fig_accounting.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_accounting.png")


if __name__ == "__main__":
    cost_of_accuracy()
    sweep()
    accounting()
    print(f"-> {OUT}")
