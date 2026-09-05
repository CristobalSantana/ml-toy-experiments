"""
run_experiment.py -- The pre-registered experiment, end to end.

    python run_experiment.py

1. loads (or reuses) the BTC daily history
2. verifies the look-ahead checks fire
3. runs the full 18-cell parameter grid on the development period
4. evaluates the same grid, once, on the untouched held-out period
5. applies the decision rules from CRITERIA.md

The headline number is the *median across the grid*, never the best cell. With
18 combinations, the best one is a lottery winner, and reporting it is how a
strategy that does nothing is made to look profitable.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B
import data_loader
import leakage
import strategy as S

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

DEV_END = "2023-12-31"          # frozen in CRITERIA.md
HOLDOUT_START = "2024-01-01"

GRID = {
    "rsi_period": [7, 14, 21],
    "pivot_window": [3, 5, 8],
    "max_pivot_gap": [30, 60],
}


def grid_params() -> list[S.Params]:
    keys = list(GRID)
    return [S.Params(**dict(zip(keys, combo))) for combo in itertools.product(*GRID.values())]


def evaluate(df: pd.DataFrame, params: list[S.Params], period: str) -> pd.DataFrame:
    rows = []
    for p in params:
        sig = S.divergence_signals(df, p)
        pos = S.positions_from_signals(sig)
        leakage.check_execution_lag(sig, pos)        # verified on every cell
        res = B.run(df, pos, label="divergence")
        rows.append({"period": period, "rsi_period": p.rsi_period,
                     "pivot_window": p.pivot_window, "max_pivot_gap": p.max_pivot_gap,
                     **{k: v for k, v in res.metrics.items() if k != "label"}})
    return pd.DataFrame(rows)


def summarise(grid: pd.DataFrame, bh: dict, period: str) -> str:
    q = grid["sharpe"].quantile
    lines = [
        f"### {period}",
        f"  buy-and-hold : sharpe {bh['sharpe']:.3f}  return {bh['total_return']*100:>9,.0f}%"
        f"  maxDD {bh['max_drawdown']*100:>6.1f}%",
        f"  divergence   : sharpe {grid['sharpe'].median():.3f} (median of "
        f"{len(grid)} cells, p25 {q(0.25):.3f}, p75 {q(0.75):.3f})",
        f"                 return {grid['total_return'].median()*100:>9,.0f}% (median)"
        f"  maxDD {grid['max_drawdown'].median()*100:>6.1f}%",
        f"                 trades {grid['n_trades'].median():.0f}  exposure "
        f"{grid['exposure'].median()*100:.0f}%  hit rate "
        f"{grid['hit_rate'].median()*100:.0f}%",
        f"  best cell    : sharpe {grid['sharpe'].max():.3f}   "
        f"worst cell: sharpe {grid['sharpe'].min():.3f}",
    ]
    return "\n".join(lines)


def decide(grid: pd.DataFrame, bh: dict) -> str:
    med = float(grid["sharpe"].median())
    p25 = float(grid["sharpe"].quantile(0.25))
    s_bh = float(bh["sharpe"])
    dd_better = float(grid["max_drawdown"].median()) > float(bh["max_drawdown"])

    if med > s_bh and p25 > s_bh:
        verdict = ("**The strategy works.** Median grid Sharpe beats buy-and-hold, and so "
                   "does the 25th percentile - it is not one lucky corner of the grid.")
    elif med <= s_bh:
        verdict = ("**The strategy fails.** Median grid Sharpe does not beat buy-and-hold. "
                   "This was the expected outcome and is reported as plainly as the "
                   "alternative would have been.")
    else:
        verdict = ("**Inconclusive.** The median beats buy-and-hold but the 25th percentile "
                   "does not, so the answer depends on which parameters you happened to "
                   "pick - which is itself the finding.")

    return "\n".join([
        f"- median grid Sharpe : {med:.3f}",
        f"- grid p25 Sharpe    : {p25:.3f}",
        f"- buy-and-hold Sharpe: {s_bh:.3f}",
        f"- max drawdown: divergence {grid['max_drawdown'].median()*100:.1f}% vs "
        f"buy-and-hold {bh['max_drawdown']*100:.1f}%"
        f" -> {'smaller (in the strategy''s favour)' if dd_better else 'not smaller'}",
        "", verdict,
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--interval", default="1d",
                    choices=list(data_loader.BARS_PER_YEAR),
                    help="candle size; the 1h arm is pre-registered in CRITERIA-1h.md")
    interval = ap.parse_args().interval
    tag = "" if interval == "1d" else f"_{interval}"

    OUT.mkdir(parents=True, exist_ok=True)
    df = data_loader.load(interval=interval)
    # Sharpe is annualised by sqrt(periods per year); with hourly bars the
    # daily constant would inflate every ratio by sqrt(24).
    B.set_periods_per_year(data_loader.BARS_PER_YEAR[interval])
    print(f"annualising by sqrt({data_loader.BARS_PER_YEAR[interval]}) for {interval} bars")

    print("\nLook-ahead checks")
    leakage.run_all(df, S.Params(), verbose=True)

    dev = df[df["date"] <= DEV_END].reset_index(drop=True)
    hold = df[df["date"] >= HOLDOUT_START].reset_index(drop=True)
    print(f"\ndevelopment {dev['date'].min().date()} -> {dev['date'].max().date()} "
          f"({len(dev):,} bars)")
    print(f"held out    {hold['date'].min().date()} -> {hold['date'].max().date()} "
          f"({len(hold):,} bars)  -- untouched until now")

    params = grid_params()
    print(f"\nrunning {len(params)} parameter combinations on each period ...")

    dev_grid = evaluate(dev, params, "development")
    hold_grid = evaluate(hold, params, "holdout")
    bh_dev = B.buy_and_hold(dev).metrics
    bh_hold = B.buy_and_hold(hold).metrics

    pd.concat([dev_grid, hold_grid]).to_csv(OUT / f"grid_results{tag}.csv", index=False)
    pd.DataFrame([bh_dev, bh_hold]).to_csv(OUT / f"buy_and_hold{tag}.csv", index=False)

    print("\n" + summarise(dev_grid, bh_dev, "DEVELOPMENT (2017-2023)"))
    print("\n" + summarise(hold_grid, bh_hold, "HELD OUT (2024-today)"))

    decision = decide(hold_grid, bh_hold)
    print("\n" + "=" * 66 + "\nPRE-REGISTERED DECISION (held-out period)\n" + "=" * 66)
    print(decision)

    (OUT / f"decision{tag}.md").write_text(
        "# Pre-registered decision\n\nRules fixed in CRITERIA.md before any backtest ran.\n\n"
        + decision + "\n", encoding="utf-8")
    print(f"\nWrote grid_results.csv, buy_and_hold.csv, decision.md -> {OUT}")


if __name__ == "__main__":
    main()
