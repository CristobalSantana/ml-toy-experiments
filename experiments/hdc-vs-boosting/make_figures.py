"""
make_figures.py -- Three figures, one per arm.

    python make_figures.py

  fig_learning_curve.png  ranking quality against how much data was available
  fig_robustness.png      what each method does when its inputs, or its own
                          representation, are damaged
  fig_cost.png            what the accuracy cost, in time and in stored numbers
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
GREY = "#8f8f8f"
COLOURS = {"majority class": GREY, "logistic regression": "#7dd3fc",
           "gradient boosting": "#0284C7", "hyperdimensional": "#e69f00"}


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


def learning_curve() -> None:
    """AUC against training size. The two ends answer different questions.

    The left end is HDC's own claim - that a class prototype means something
    after very few records. The right end is whether that still matters once
    data is cheap.
    """
    lc = pd.read_csv(OUT / "learning_curve.csv")
    g = lc.groupby(["model", "n_train"])["auc"]
    med, lo, hi = g.median(), g.min(), g.max()

    fig, ax = plt.subplots(figsize=(10.5, 5.6), facecolor=BG)
    for model in COLOURS:
        if model not in med.index.get_level_values(0):
            continue
        m = med.loc[model].sort_index()
        ax.plot(m.index, m.values, "o-", lw=2, ms=5,
                color=COLOURS[model], label=model)
        ax.fill_between(m.index, lo.loc[model].sort_index().values,
                        hi.loc[model].sort_index().values,
                        color=COLOURS[model], alpha=0.13, lw=0)

    ax.axhline(0.5, color=GREY, lw=1, ls=":")
    ax.text(lc.n_train.max(), 0.505, "no ranking at all", ha="right",
            va="bottom", color=GREY, fontsize=8.5)
    ax.set_xscale("log")
    _style(ax, "Ranking quality against how much data there was",
           "training rows (log scale)", "ROC AUC on a fixed test month")
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=9,
              loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_learning_curve.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_learning_curve.png")


def robustness() -> None:
    """Two damage models, side by side, because they are not the same test.

    Left: the inputs are wrong, which can happen to anything. Right: the
    method's own representation is partly erased, which only means something
    for a method that spreads information across every dimension.
    """
    rb = pd.read_csv(OUT / "robustness.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor=BG)

    inp = rb[rb["kind"] == "input"]
    for model, g in inp.groupby("model"):
        g = g.sort_values("fraction")
        axes[0].plot(g["fraction"] * 100, g["auc"], "o-", lw=2, ms=6,
                     color=COLOURS.get(model, GREY), label=model)
    _style(axes[0], "Inputs corrupted  (both methods)",
           "share of feature values replaced (%)", "ROC AUC")
    axes[0].legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=9)

    dd = rb[rb["kind"] == "dimension_dropout"].sort_values("fraction")
    axes[1].plot(dd["fraction"] * 100, dd["balanced_accuracy"], "o-", lw=2,
                 ms=7, color=COLOURS["hyperdimensional"])
    base = float(dd["balanced_accuracy"].iloc[0])
    axes[1].axhline(base, color=GREY, lw=1, ls=":")
    axes[1].text(2, base + 0.0004, "undamaged", color=GREY, fontsize=8.5)
    _style(axes[1], "HDC's own representation erased  (no equivalent for trees)",
           "share of hypervector dimensions switched off (%)",
           "balanced accuracy")

    fig.suptitle("Two kinds of damage", color=FG, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_robustness.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_robustness.png")


def cost() -> None:
    """What the ranking quality cost: seconds to fit, numbers to keep."""
    c = pd.read_csv(OUT / "cost.csv")
    c = c[c["model"] != "majority class"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=BG)

    for ax, col, title, xlabel in (
            (axes[0], "fit_seconds", "Time to fit", "seconds (log scale)"),
            (axes[1], "n_parameters", "What has to be kept afterwards",
             "stored numbers (log scale)")):
        y = np.arange(len(c))
        ax.barh(y, c[col], color=[COLOURS.get(m, GREY) for m in c["model"]],
                height=0.55)
        for i, (v, a) in enumerate(zip(c[col], c["auc"])):
            ax.text(v * 1.15, i, f"{v:,.0f}" if v > 10 else f"{v:.2f}",
                    va="center", color=FG, fontsize=9)
        ax.set_yticks(y, [f"{m}\nAUC {a:.3f}" for m, a in zip(c["model"], c["auc"])])
        for lbl in ax.get_yticklabels():
            lbl.set_color(FG)
        ax.set_xscale("log")
        _style(ax, title, xlabel)

    fig.suptitle("The price of the ranking, at the largest training size",
                 color=FG, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cost.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_cost.png")


if __name__ == "__main__":
    learning_curve()
    robustness()
    cost()
    print(f"-> {OUT}")
