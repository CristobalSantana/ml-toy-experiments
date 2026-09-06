"""
make_figures.py -- Three figures.

    python make_figures.py

  fig_cliff.png        where recovery stops, and how far smoothing moves it
  fig_failure_mode.png what it does at the moment it stops
  fig_extrapolation.png the equation against two black boxes
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
VIOLET = "#9a7fd1"

# sigma = 0 has no place on a log axis, and dropping it would hide the one
# case everybody assumes. It is drawn at this position and labelled "0".
ZERO_AT = 3e-10


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


def _x(sigma: pd.Series) -> np.ndarray:
    return np.where(sigma == 0, ZERO_AT, sigma)


def cliff() -> None:
    """The whole result in one line: recovery is a cliff, not a slope."""
    raw = pd.read_csv(OUT / "sweep.csv")
    sm = pd.read_csv(OUT / "smoothed.csv")
    fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=BG)

    series = [("raw finite differences", raw, ACCENT, 2.6, "o")]
    colours = {5: GREEN, 11: WARN, 21: VIOLET, 41: RED}
    for w, c in colours.items():
        series.append((f"Savitzky-Golay, window {w}", sm[sm["window"] == w],
                       c, 1.7, "s"))

    for label, d, colour, lw, mk in series:
        g = d.groupby("sigma")["recovered"].mean().reset_index()
        ax.plot(_x(g["sigma"]), g["recovered"], marker=mk, ms=5, lw=lw,
                color=colour, label=label)

    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.12)
    ax.set_yticks([0, 0.5, 1.0], ["0", "half", "always"])
    ticks = [ZERO_AT, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    ax.set_xticks(ticks, ["0", "1e-9", "1e-8", "1e-7", "1e-6", "1e-5",
                          "1e-4", "1e-3", "1e-2"])
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(FG)
    ax.axvspan(ZERO_AT * 0.5, 6e-10, color="#1c1c1c", zorder=0)
    # 21 and 41 lie exactly on top of each other along the bottom, which is
    # the finding rather than a drawing problem; say so instead of nudging
    # one of them off its true value.
    ax.text(2e-5, 0.07, "windows 21 and 41 never recover, at any noise level\n"
                        "- their smoothing bias alone exceeds the tolerance",
            ha="center", color=GREY, fontsize=9)

    _style(ax, "Recovery of the diffusion equation is a cliff, not a slope",
           "noise added to the field, as a fraction of its range",
           "share of 5 seeds that recovered  u_t = 1.3333 u_xx")
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=9,
              loc="center left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cliff.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_cliff.png")


def failure_mode() -> None:
    """What it does at the moment it stops working.

    The two panels are the point. The term count explodes at the cliff; the
    coefficient stays inside the 5% tolerance for another decade and a half,
    and `u_xx` is still in the model nearly four decades later. Anyone
    reporting only the coefficient would be quoting a number good to a
    fraction of a percent from a model with ten terms in it.
    """
    raw = pd.read_csv(OUT / "sweep.csv")
    s = json.loads((OUT / "summary.json").read_text())
    cl = s["cliff_raw"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor=BG)

    g = raw.groupby("sigma").agg(
        n_terms=("n_terms", "median"),
        kept=("kept_true_term", "mean"),
        err=("coef_rel_error", "median")).reset_index()
    x = _x(g["sigma"])

    axes[0].plot(x, g["n_terms"], marker="o", ms=5, lw=2.4, color=ACCENT,
                 label="terms in the discovered equation")
    axes[0].axhline(1, color=GREY, lw=1.2, ls=":")
    axes[0].text(4e-10, 1.45, "the truth has one term", color=GREY, fontsize=9)
    axes[0].set_ylim(0, 11.8)
    _style(axes[0], "It buries the right term rather than dropping it",
           "noise, as a fraction of the field's range", "number of terms")

    # a real second axis, not a rescaled line pretending to have one
    tw = axes[0].twinx()
    tw.set_xscale("log")          # before the shared ticks are set below
    tw.plot(x, g["kept"], marker="s", ms=5, lw=1.8, color=GREEN, ls="--",
            label="share of seeds still containing u_xx")
    tw.set_ylim(-0.02, 1.07)
    tw.set_ylabel("share of seeds still containing u_xx", color=GREEN,
                  fontsize=9)
    tw.tick_params(colors=GREEN, labelsize=9)
    for sp in tw.spines.values():
        sp.set_visible(False)
    tw.spines["right"].set_visible(True)
    tw.spines["right"].set_color(GREEN)
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = tw.get_legend_handles_labels()
    axes[0].legend(h1 + h2, l1 + l2, facecolor=BG, edgecolor=GREY,
                   labelcolor=FG, fontsize=8.5, loc="center left")

    axes[1].plot(x, g["err"], marker="o", ms=5, lw=2.4, color=WARN)
    axes[1].axhline(s["tolerance"], color=GREEN, lw=1.6, ls="--")
    axes[1].text(4e-10, s["tolerance"] * 1.4, "5% tolerance", color=GREEN,
                 fontsize=9)
    axes[1].set_yscale("log")
    _style(axes[1], "The number itself stays right well past that point",
           "noise, as a fraction of the field's range",
           "relative error of the u_xx coefficient")

    for ax in (axes[0], axes[1]):
        ax.set_xscale("log")
        ax.axvline(cl, color=RED, lw=1.4, ls="-.")
        ax.set_xticks([ZERO_AT, 1e-8, 1e-6, 1e-4, 1e-2],
                      ["0", "1e-8", "1e-6", "1e-4", "1e-2"])
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(FG)
    axes[1].text(cl * 1.6, 3e-6, "recovery\nstops here", color=RED, fontsize=9)

    fig.suptitle("The equation stops being identifiable long before its "
                 "coefficient goes wrong", color=FG, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_failure_mode.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_failure_mode.png")


def extrapolation() -> None:
    """The prediction that failed, drawn honestly.

    Points on a log axis rather than bars: the errors span a factor of 50, so
    a linear scale would flatten the three small ones into nothing, and a bar
    on a log axis has a length that means nothing at all.
    """
    e = pd.read_csv(OUT / "extrapolation.csv")
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=BG)

    halves = ["trained on", "held out (later)"]
    models = [("equation", "the discovered equation (one term)", ACCENT, "o"),
              ("gbdt_library", "gradient booster, given the same library",
               WARN, "s"),
              ("gbdt_raw", "gradient booster, given only raw u", RED, "^")]

    for row, half in enumerate(halves):
        d = e[e["half"] == half]
        vals = [d[c].median() for c, _, _, _ in models]
        ax.plot([min(vals), max(vals)], [row, row], color="#2a2a2a", lw=2,
                zorder=1)
        # two of these land on top of each other, which is the result; drop
        # the colliding label below the line rather than moving the point
        order = np.argsort(vals)
        below = set()
        for a, b in zip(order[:-1], order[1:]):
            if vals[b] / vals[a] < 1.3 and a not in below:
                below.add(b)
        for i, ((col, label, colour, mk), v) in enumerate(zip(models, vals)):
            ax.scatter(v, row, s=170, color=colour, marker=mk, zorder=3,
                       label=label if row == 0 else None)
            ax.text(v, row + (-0.30 if i in below else 0.17), f"{v:.4f}",
                    ha="center", color=FG, fontsize=9)

    ax.set_xscale("log")
    ax.set_xlim(1.4e-3, 1.7e-1)
    ax.set_ylim(-0.55, 1.8)
    ax.set_yticks([0, 1], ["first half of the time domain\n(fitted here)",
                           "second half\n(never seen)"])
    for lbl in ax.get_yticklabels() + ax.get_xticklabels():
        lbl.set_color(FG)
    _style(ax, "P5 predicted the black box would win here and lose there. "
               "It did neither.",
           "relative RMSE against the clean du/dt  (lower is better, log scale)",
           "")
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=9,
              loc="upper center", ncol=1)
    fig.tight_layout()
    fig.savefig(OUT / "fig_extrapolation.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_extrapolation.png")


if __name__ == "__main__":
    cliff()
    failure_mode()
    extrapolation()
    print(f"-> {OUT}")
