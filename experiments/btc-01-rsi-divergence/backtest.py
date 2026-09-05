"""
backtest.py -- Turn positions into an equity curve, with costs, and score it
against buy-and-hold.

Execution follows CRITERIA.md exactly: a signal at the close of bar i is filled
at the *open* of bar i+1. So the bar on which a position opens earns only
open->close, not the whole close-to-close move that the signal was made of.
Getting this wrong is worth several percent a year of imaginary return.

Costs are charged on entry and on exit: 0.10% fee + 0.05% slippage per side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEE = 0.0010          # Binance spot taker
SLIPPAGE = 0.0005
COST_PER_SIDE = FEE + SLIPPAGE
PERIODS_PER_YEAR = 365   # set per timeframe; 365 daily, 8760 hourly


@dataclass
class Result:
    equity: pd.Series
    returns: pd.Series
    position: pd.Series
    metrics: dict


def _bar_returns(df: pd.DataFrame, pos: pd.Series) -> np.ndarray:
    """Return actually earned on each bar, honouring open-fill execution."""
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    p = pos.to_numpy()
    n = len(df)
    r = np.zeros(n)
    for i in range(1, n):
        prev, cur = p[i - 1], p[i]
        if prev == 0 and cur == 1:        # entered at this bar's open
            r[i] = c[i] / o[i] - 1
        elif prev == 1 and cur == 1:      # already in
            r[i] = c[i] / c[i - 1] - 1
        elif prev == 1 and cur == 0:      # exited at this bar's open
            r[i] = o[i] / c[i - 1] - 1
    return r


def set_periods_per_year(n: int) -> None:
    """Sharpe is annualised by sqrt(periods per year); leaving this at the daily
    value while feeding hourly bars would inflate the ratio by sqrt(24)."""
    global PERIODS_PER_YEAR
    PERIODS_PER_YEAR = n


def metrics(returns: pd.Series, position: pd.Series | None = None,
            label: str = "") -> dict:
    eq = (1 + returns).cumprod()
    n = len(returns)
    years = n / PERIODS_PER_YEAR
    total = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    sd = float(returns.std())
    sharpe = float(returns.mean() / sd * np.sqrt(PERIODS_PER_YEAR)) if sd > 0 else np.nan
    dd = float((eq / eq.cummax() - 1).min())

    out = {"label": label, "total_return": total, "cagr": cagr, "sharpe": sharpe,
           "max_drawdown": dd, "n_bars": int(n)}

    if position is not None:
        changes = position.diff().fillna(position.iloc[0])
        entries = int((changes > 0).sum())
        out["n_trades"] = entries
        out["exposure"] = float(position.mean())
        # per-trade outcome, for the hit rate
        wins, tot, cur = 0, 0, None
        p = position.to_numpy()
        e = (1 + returns).cumprod().to_numpy()
        for i in range(1, len(p)):
            if p[i] == 1 and p[i - 1] == 0:
                cur = e[i - 1]
            elif p[i] == 0 and p[i - 1] == 1 and cur is not None:
                tot += 1
                wins += int(e[i] > cur)
                cur = None
        if cur is not None:                     # still open at the end
            tot += 1
            wins += int(e[-1] > cur)
        out["hit_rate"] = wins / tot if tot else np.nan
    return out


def run(df: pd.DataFrame, position: pd.Series, label: str = "strategy") -> Result:
    gross = _bar_returns(df, position)

    # charge a side every time the position changes
    changes = position.diff().fillna(position.iloc[0]).abs().to_numpy()
    costs = changes * COST_PER_SIDE
    net = gross - costs

    returns = pd.Series(net, index=df["date"], name="returns")
    equity = (1 + returns).cumprod().rename("equity")
    m = metrics(returns, position.set_axis(df["date"]), label)
    m["total_cost_drag"] = float(costs.sum())
    return Result(equity=equity, returns=returns, position=position.set_axis(df["date"]),
                  metrics=m)


def buy_and_hold(df: pd.DataFrame) -> Result:
    """The benchmark that matters. Bought at the first open, held throughout,
    charged one entry cost so the comparison is not rigged in its favour."""
    c = df["close"].to_numpy()
    o = df["open"].to_numpy()
    r = np.zeros(len(df))
    r[0] = c[0] / o[0] - 1 - COST_PER_SIDE
    r[1:] = c[1:] / c[:-1] - 1
    returns = pd.Series(r, index=df["date"], name="returns")
    pos = pd.Series(np.ones(len(df)), index=df["date"], name="position")
    m = metrics(returns, pos, "buy_and_hold")
    m["n_trades"] = 1
    m["total_cost_drag"] = COST_PER_SIDE
    return Result(equity=(1 + returns).cumprod().rename("equity"),
                  returns=returns, position=pos, metrics=m)
