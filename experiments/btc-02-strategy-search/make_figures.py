"""
make_figures.py -- Three figures.

    python make_figures.py

  fig_null_comparison.png  what a search this size produces when nothing is there
  fig_grid.png             where the optimum was, and where it went
  fig_dev_vs_holdout.png   how much a good backtest tells you about next year
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
ACCENT = "#0284C7"
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


def _load():
    real = pd.read_csv(OUT / "real.csv")
    nulls = pd.read_csv(OUT / "nulls.csv")
    v = json.loads((OUT / "verdict.json").read_text(encoding="utf-8"))
    return real, nulls, v


def null_comparison() -> None:
    """The figure the experiment exists for.

    The real search's best result, against the distribution of best results
    from the identical search run on data where no timing rule can work.
    """
    _, nulls, v = _load()
    real = v["real"]["best_dev_excess"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), facecolor=BG,
                             sharey=True)

    titles = {
        "shuffle": "Bars shuffled\n(no serial structure at all)",
        "block": "20-day blocks resampled\n(volatility clustering preserved)",
    }
    for ax, kind, colour in zip(axes, ("shuffle", "block"), (ACCENT, WARN)):
        s = nulls[nulls["kind"] == kind]["best_excess"]
        # clipped so one extreme surrogate cannot flatten the whole histogram;
        # the count above the clip is stated on the axis instead
        hi = max(float(s.quantile(0.99)), real * 1.15)
        ax.hist(np.clip(s, None, hi), bins=34, color=colour, alpha=0.85)
        over = int((s > hi).sum())
        ax.axvline(real, color=RED, lw=2.4)
        ax.axvline(v[kind]["best_excess_q95"], color=GREEN, lw=1.6, ls="--")
        ax.set_xlim(min(float(s.min()), 0) - 0.2, hi * 1.02)
        _style(ax, titles[kind],
               "best strategy's final equity, relative to buy-and-hold"
               + (f"  ({over} surrogate(s) beyond the axis)" if over else ""),
               "surrogates" if kind == "shuffle" else "")
        ax.text(real, ax.get_ylim()[1] * 0.92, "  real Bitcoin", color=RED,
                fontsize=9.5, va="top")
        ax.text(v[kind]["best_excess_q95"], ax.get_ylim()[1] * 0.72,
                "  95th percentile", color=GREEN, fontsize=9, va="top")
        ax.text(0.97, 0.5, f"p = {v[kind]['p_value']:.3f}",
                transform=ax.transAxes, ha="right", color=FG, fontsize=13)

    fig.suptitle("The same search, on data where nothing can be found",
                 color=FG, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_null_comparison.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_null_comparison.png")


def grid() -> None:
    """Where the optimum was, and where it was a year later."""
    real, _, v = _load()
    bh_d, bh_h = v["buy_and_hold"]["dev"], v["buy_and_hold"]["holdout"]
    bf, bs = v["real"]["best_params"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=BG)
    for ax, col, bh, title in (
            (axes[0], "dev_return", bh_d, "Development (2017-2023)"),
            (axes[1], "holdout_return", bh_h, "Holdout (2024-2026)")):
        piv = real.pivot(index="slow", columns="fast", values=col)
        # centred on buy-and-hold: red is losing to simply holding, green is
        # beating it, and the two panels use their own period's benchmark
        span = float(np.nanmax(np.abs(piv.to_numpy() - bh)))
        m = ax.pcolormesh(piv.columns, piv.index, piv.to_numpy() - bh,
                          cmap="RdYlGn", vmin=-span, vmax=span, shading="auto")
        ax.scatter([bf], [bs], s=180, facecolors="none", edgecolors="#111111",
                   linewidths=2.6, zorder=5)
        ax.scatter([bf], [bs], s=180, facecolors="none", edgecolors=FG,
                   linewidths=1.2, zorder=6)
        cb = fig.colorbar(m, ax=ax)
        cb.set_label("total return minus buy-and-hold", color=GREY, fontsize=9)
        cb.ax.tick_params(colors=GREY, labelsize=8)
        _style(ax, title, "fast window (days)", "slow window (days)")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(FG)
    box = dict(boxstyle="round,pad=0.3", facecolor="#111111", alpha=0.85,
               edgecolor="none")
    axes[0].text(bf + 2, bs, "best in development", color=FG, fontsize=9,
                 va="center", bbox=box)
    axes[1].text(bf + 2, bs, "the same cell", color=FG, fontsize=9,
                 va="center", bbox=box)

    fig.suptitle("Green beats buy-and-hold. The green shrinks.",
                 color=FG, fontsize=12.5)
    # each panel is scaled to its own period, because the development period
    # returned 891% and the holdout 49%: one shared scale would render the
    # right-hand panel a single flat colour. Compare the *sign* across
    # panels, not the intensity.
    fig.text(0.5, 0.015,
             "Colour scales are per panel - development spans "
             f"+/-{float(np.nanmax(np.abs(real['dev_return'] - bh_d))):.0f} "
             f"and the holdout +/-"
             f"{float(np.nanmax(np.abs(real['holdout_return'] - bh_h))):.1f}. "
             "Compare the sign between panels, not the intensity.",
             ha="center", color=GREY, fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(OUT / "fig_grid.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_grid.png")


def dev_vs_holdout() -> None:
    """Every strategy, twice: how it did then and how it did next."""
    real, _, v = _load()
    bh_d, bh_h = v["buy_and_hold"]["dev"], v["buy_and_hold"]["holdout"]
    fig, ax = plt.subplots(figsize=(10.5, 6.0), facecolor=BG)

    won_dev = real["dev_return"] > bh_d
    ax.scatter(real.loc[~won_dev, "dev_return"],
               real.loc[~won_dev, "holdout_return"], s=13, color=GREY,
               alpha=0.55, label="lost to buy-and-hold in development")
    ax.scatter(real.loc[won_dev, "dev_return"],
               real.loc[won_dev, "holdout_return"], s=16, color=ACCENT,
               alpha=0.8, label="beat buy-and-hold in development")

    ax.axvline(bh_d, color=WARN, lw=1.6, ls="--")
    ax.axhline(bh_h, color=WARN, lw=1.6, ls="--")
    best = real.loc[real["dev_return"].idxmax()]
    ax.scatter([best["dev_return"]], [best["holdout_return"]], s=170,
               marker="*", color=RED, zorder=6,
               label="the one you would have picked")

    both = int(((real["dev_return"] > bh_d)
                & (real["holdout_return"] > bh_h)).sum())
    n_dev = int(won_dev.sum())
    ax.text(0.98, 0.97,
            f"{n_dev} of {len(real):,} beat buy-and-hold in development\n"
            f"{both} of those {n_dev} did so again in the holdout\n"
            f"rank correlation between the two axes: "
            f"{real['dev_return'].corr(real['holdout_return'], method='spearman'):+.2f}",
            transform=ax.transAxes, ha="right", va="top", color=FG,
            fontsize=9.5)
    ax.text(bh_d, ax.get_ylim()[0], "  buy-and-hold, development", color=WARN,
            fontsize=8.5, rotation=90, va="bottom")
    ax.text(ax.get_xlim()[0], bh_h, "  buy-and-hold, holdout", color=WARN,
            fontsize=8.5, ha="left", va="bottom")

    # plain log: every development return is positive (the worst strategy
    # still ended up 71%), so symlog would spend half the axis on empty space
    ax.set_xscale("log")
    _style(ax, "A good backtest and the year that follows",
           "total return, development 2017-2023 (log scale)",
           "total return, holdout 2024-2026")
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=9,
              loc="lower left")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(FG)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dev_vs_holdout.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_dev_vs_holdout.png")


if __name__ == "__main__":
    null_comparison()
    grid()
    dev_vs_holdout()
    print(f"-> {OUT}")
