"""
make_figures.py -- Three figures.

    python make_figures.py

  fig_two_framings.png  the same result reported two ways, side by side
  fig_week.png          a week of the held-out year, hour by hour
  fig_clear_sky.png     the envelope every baseline is measured against

Same palette as the rest of the repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATASET = HERE.parents[1] / "datasets" / "de" / "opsd_solar"

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


def two_framings() -> None:
    """The headline: R2 squeezes everything against 1.0, skill spreads it out.

    Both panels show the same six predictors on the same held-out year. The
    left one is what solar-forecasting papers report; the right one is what
    the forecast is worth against the benchmark it has to beat.
    """
    s = pd.read_csv(OUT / "scores.csv")
    s = s[s["model"] != "B0 train mean"]      # off the scale, and not a claim
    order = ["B1 climatology", "B2 smart persistence", "M ridge", "M gbdt", "M esn"]
    colours = {"B1 climatology": GREY, "B2 smart persistence": WARN,
               "M ridge": "#7dd3fc", "M gbdt": ACCENT, "M esn": GREEN}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=BG)
    width = 0.38

    for ax, (col, title, xlabel) in zip(axes, [
            ("r2_all", "What gets reported:  R² over all hours",
             "R²   (1.0 is perfect)"),
            ("skill_vs_persistence",
             "What it is worth:  skill against smart persistence",
             "share of the benchmark's error removed")]):
        y = np.arange(len(order))
        for i, h in enumerate((1, 24)):
            vals = [float(s[(s.horizon == h) & (s.model == m)][col].iloc[0]) for m in order]
            off = (i - 0.5) * width
            # Clip long negative bars to the axis and say so. Climatology's
            # skill at 1 h is -5.75; drawn to scale it would flatten every
            # other bar, and drawn clipped without a mark it would be a lie
            # about its length.
            floor = -0.60 if col == "skill_vs_persistence" else 0.0
            drawn = [max(v, floor) for v in vals]
            bars = ax.barh(y + off, drawn, height=width,
                           color=[colours[m] for m in order],
                           alpha=1.0 if h == 1 else 0.45, edgecolor="none")
            for b, v, dv in zip(bars, vals, drawn):
                clipped = v < floor
                txt = f"{v:.3f}" + ("  ◀ off scale" if clipped else "")
                ax.text(max(dv, 0) + 0.012 if not clipped else floor + 0.02,
                        b.get_y() + b.get_height() / 2, txt,
                        va="center", ha="left", color=RED if clipped else FG,
                        fontsize=8)
        ax.set_yticks(y, [m.split(" ", 1)[1] for m in order])
        for lbl in ax.get_yticklabels():
            lbl.set_color(FG)
        ax.axvline(0, color=GREY, lw=1)
        _style(ax, title, xlabel)

    axes[0].set_xlim(0, 1.13)
    axes[1].set_xlim(-0.63, 1.05)
    # solid = 1 hour ahead, faded = 24
    axes[1].text(0.98, 0.03, "solid: 1 h ahead     faded: 24 h ahead",
                 transform=axes[1].transAxes, ha="right", va="bottom",
                 color=GREY, fontsize=9)
    fig.suptitle("The same forecasts, scored two ways", color=FG, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_two_framings.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_two_framings.png")


def week() -> None:
    """One week of the held-out year, so the numbers have a shape."""
    p = OUT / "week_h1.csv"
    if not p.exists():
        return
    w = pd.read_csv(p, parse_dates=["time"])
    fig, ax = plt.subplots(figsize=(13, 5.0), facecolor=BG)

    ax.fill_between(w["time"], 0, w["actual"], color=FG, alpha=0.16, lw=0)
    ax.plot(w["time"], w["actual"], color=FG, lw=2.2, label="measured")
    ax.plot(w["time"], w["B1 climatology"], color=GREY, lw=1.6, ls="--",
            label="climatology (knows only the calendar)")
    ax.plot(w["time"], w["B2 smart persistence"], color=WARN, lw=1.6,
            label="smart persistence (last hour carried forward)")
    ax.plot(w["time"], w["M esn"], color=GREEN, lw=1.8,
            label="echo state network, 1 h ahead")

    _style(ax, "A week of the held-out year, 10-17 June 2019",
           "", "capacity factor")
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=9, loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "fig_week.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_week.png")


def clear_sky() -> None:
    """Output against sun elevation, with the envelope every baseline uses.

    The vertical spread at a fixed elevation is the weather - the only part of
    this that has to be forecast. The curve through the top of it is what the
    sun alone would give, and it is free.
    """
    sys.path.insert(0, str(DATASET))
    from load import load as load_solar

    df = load_solar(verbose=False)
    cs = pd.read_csv(OUT / "clear_sky.csv")
    rng = np.random.default_rng(0)
    sel = rng.choice(len(df), size=min(12000, len(df)), replace=False)
    d = df.iloc[sel]

    fig, ax = plt.subplots(figsize=(10.5, 5.4), facecolor=BG)
    ax.scatter(d["elevation"], d["cf"], s=3, color=ACCENT, alpha=0.20,
               edgecolors="none", label="one hour")
    ax.plot(cs["elevation"], cs["clear_sky_cf"], color=WARN, lw=2.4,
            label="clear-sky envelope (90th percentile per degree)")
    ax.axvline(5, color=GREY, lw=1.2, ls=":")
    ax.text(5.6, 0.63, "sun above 5°\n= 'daylight'", color=GREY, fontsize=9,
            va="top")

    _style(ax, "Solar output against the sun's elevation, 2015-2019",
           "sun elevation (degrees)", "capacity factor")
    ax.set_xlim(-10, 62)
    ax.set_ylim(-0.02, 0.72)
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_clear_sky.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_clear_sky.png")


if __name__ == "__main__":
    two_framings()
    week()
    clear_sky()
    print(f"-> {OUT}")
