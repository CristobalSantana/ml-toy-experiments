"""
make_figures.py -- Figures for the BTC divergence experiment.

Three, each answering a different question:

  fig_equity.png    what would have happened to the money
  fig_grid.png      does any parameter choice survive out of sample
  fig_signals.png   what the strategy actually does on the chart
  fig_timeframes.png daily vs hourly, and what costs ate
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import backtest as B  # noqa: E402
import data_loader  # noqa: E402
import strategy as S  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DEV_END, HOLDOUT_START = "2023-12-31", "2024-01-01"

BG = "#0e0e0e"
FG = "#f5f0e8"
ACCENT = "#0284C7"
GREY = "#8f8f8f"
WARN = "#e05555"


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GREY)
    ax.tick_params(colors=GREY, labelsize=9)
    ax.set_title(title, color=FG, fontsize=11)
    ax.set_xlabel(xlabel, color=GREY, fontsize=9)
    ax.set_ylabel(ylabel, color=GREY, fontsize=9)
    ax.grid(alpha=0.15, color=GREY)


def equity() -> None:
    df = data_loader.load()
    p = S.Params()                      # the middle of the grid, not the best cell
    sig = S.divergence_signals(df, p)
    pos = S.positions_from_signals(sig)
    strat = B.run(df, pos)
    bh = B.buy_and_hold(df)

    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=BG)
    ax.plot(bh.equity.index, bh.equity, color=FG, lw=2, label="buy and hold")
    ax.plot(strat.equity.index, strat.equity, color=ACCENT, lw=2,
            label="RSI divergence (default parameters)")
    ax.axvline(pd.Timestamp(HOLDOUT_START), color=WARN, ls="--", lw=1.4)
    ax.text(pd.Timestamp(HOLDOUT_START), ax.get_ylim()[1] * 0.6,
            "  held-out period\n  (untouched until the end)", color=WARN, fontsize=9)
    ax.set_yscale("log")
    _style(ax, "Growth of 1 USDT in BTC, 2017-2026",
           ylabel="equity, log scale (1 = starting capital)")
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_equity.png", dpi=150, facecolor=BG)
    plt.close(fig)


def grid() -> None:
    g = pd.read_csv(OUT / "grid_results.csv")
    bh = pd.read_csv(OUT / "buy_and_hold.csv")
    key = ["rsi_period", "pivot_window", "max_pivot_gap"]
    d = g[g.period == "development"].set_index(key)
    h = g[g.period == "holdout"].set_index(key)
    j = d[["sharpe"]].join(h[["sharpe"]], lsuffix="_dev", rsuffix="_hold")
    bh_dev, bh_hold = float(bh.sharpe.iloc[0]), float(bh.sharpe.iloc[1])

    fig, ax = plt.subplots(figsize=(8.2, 6.4), facecolor=BG)
    ax.axhline(bh_hold, color=FG, ls="--", lw=1.4)
    ax.axvline(bh_dev, color=FG, ls="--", lw=1.4)
    ax.text(bh_dev, ax.get_ylim()[1], " buy and hold ", color=FG, fontsize=8, rotation=90,
            va="top", ha="right")
    ax.scatter(j.sharpe_dev, j.sharpe_hold, s=90, color=ACCENT, alpha=0.85,
               edgecolor=FG, lw=0.6, zorder=3)

    best = j.sharpe_dev.idxmax()
    ax.scatter([j.loc[best, "sharpe_dev"]], [j.loc[best, "sharpe_hold"]], s=230,
               facecolor="none", edgecolor=WARN, lw=2.2, zorder=4)
    ax.annotate("best cell in development:\nbeats buy-and-hold here,\nloses to it out of sample",
                (j.loc[best, "sharpe_dev"], j.loc[best, "sharpe_hold"]),
                textcoords="offset points", xytext=(-165, -78), color=WARN, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=WARN, lw=1))

    # the quadrant that would matter
    ax.axhspan(bh_hold, max(j.sharpe_hold.max(), bh_hold) + 0.2,
               xmin=0, xmax=1, color=ACCENT, alpha=0.05)
    _style(ax, "Does any parameter choice survive out of sample?\n"
               "18 combinations, none in the upper-right quadrant",
           xlabel="Sharpe on development (2017-2023)",
           ylabel="Sharpe on held-out (2024-2026)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_grid.png", dpi=150, facecolor=BG)
    plt.close(fig)


def signals() -> None:
    df = data_loader.load()
    recent = df[df["date"] >= "2023-01-01"].reset_index(drop=True)
    p = S.Params()
    sig = S.divergence_signals(recent, p)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), facecolor=BG, sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    ax = axes[0]
    ax.plot(sig["date"], sig["close"], color=FG, lw=1.3)
    bull, bear = sig[sig.bullish], sig[sig.bearish]
    ax.scatter(bull["date"], bull["close"], marker="^", s=70, color=ACCENT,
               label="bullish divergence confirmed", zorder=3)
    ax.scatter(bear["date"], bear["close"], marker="v", s=70, color=WARN,
               label="bearish divergence confirmed", zorder=3)
    _style(ax, "What the strategy actually trades (2023 onward)", ylabel="BTC/USDT")
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=8)

    ax = axes[1]
    ax.plot(sig["date"], sig["rsi"], color=ACCENT, lw=1.2)
    ax.axhline(70, color=GREY, ls=":", lw=1)
    ax.axhline(30, color=GREY, ls=":", lw=1)
    _style(ax, "", ylabel=f"RSI({p.rsi_period})")
    fig.tight_layout()
    fig.savefig(OUT / "fig_signals.png", dpi=150, facecolor=BG)
    plt.close(fig)


def timeframes() -> None:
    """Both arms side by side, and where the hourly one goes wrong.

    Only drawn once the 1h arm has been run (outputs/grid_results_1h.csv).
    """
    p1h = OUT / "grid_results_1h.csv"
    if not p1h.exists():
        return
    g = pd.concat([pd.read_csv(OUT / "grid_results.csv").assign(arm="daily"),
                   pd.read_csv(p1h).assign(arm="hourly")])
    bh = {("daily", "development"): 0.862, ("daily", "holdout"): 0.557,
          ("hourly", "development"): 0.852, ("hourly", "holdout"): 0.559}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), facecolor=BG)

    ax = axes[0]
    labels, data, marks = [], [], []
    for arm in ("daily", "hourly"):
        for per in ("development", "holdout"):
            sub = g[(g.arm == arm) & (g.period == per)]
            labels.append(f"{arm}\n{per[:4]}")
            data.append(sub["sharpe"].to_numpy())
            marks.append(bh[(arm, per)])
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(ACCENT); patch.set_alpha(0.65)
    for e in ("whiskers", "caps", "medians"):
        for item in bp[e]:
            item.set_color(FG)
    for i, m in enumerate(marks, start=1):
        ax.plot([i - 0.32, i + 0.32], [m, m], color=WARN, lw=2.2,
                label="buy and hold" if i == 1 else None)
    _style(ax, "Sharpe across the 18-cell grid\nred line = buy and hold",
           ylabel="Sharpe")
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=8)

    ax = axes[1]
    cost = g.groupby(["arm", "period"])["total_cost_drag"].median() * 100
    trades = g.groupby(["arm", "period"])["n_trades"].median()
    keys = [("daily", "development"), ("daily", "holdout"),
            ("hourly", "development"), ("hourly", "holdout")]
    xs = np.arange(len(keys))
    ax.bar(xs, [cost[k] for k in keys], color=[ACCENT, ACCENT, WARN, WARN], alpha=0.85)
    for x, k in zip(xs, keys):
        ax.text(x, cost[k] + 1.5, f"{trades[k]:.0f} trades", ha="center",
                color=FG, fontsize=8)
    ax.set_xticks(xs, [f"{a}\n{p[:4]}" for a, p in keys])
    _style(ax, "What trading costs consumed\n0.30% per round trip",
           ylabel="cumulative cost drag (% of capital)")

    fig.suptitle("Daily vs hourly: trading ten times as often does not help",
                 color=FG, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_timeframes.png", dpi=150, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    equity(); grid(); signals(); timeframes()
    print(f"Wrote fig_equity.png, fig_grid.png, fig_signals.png -> {OUT}")
