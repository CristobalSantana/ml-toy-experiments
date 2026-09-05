"""
strategy.py -- RSI, pivot detection and divergence signals, built so that no
signal can see the future.

The trap this file exists to avoid: a pivot low at bar `t` is a low that is
lower than the `N` bars before it *and the N bars after it*. Those later bars
have not happened yet at `t`. Detect pivots with a centred window, mark the
signal at `t`, and the backtest earns money it could never have earned - the
classic way a divergence strategy is made to look profitable.

Here a pivot found at `t` is only *published* at `t + N`, and every signal is
executed on the following bar's open. `leakage.py` verifies this empirically
rather than trusting the comment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Params:
    rsi_period: int = 14
    pivot_window: int = 5        # bars required either side of a pivot
    max_pivot_gap: int = 60      # max bars between the two pivots of a divergence
    min_pivot_gap: int = 5       # too-close pivots are noise, not structure


def rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI - the smoothing the indicator was defined with, and what
    charting platforms show. A plain rolling mean gives visibly different
    values and would make these results incomparable to anyone else's."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0).where(avg_loss.notna(), np.nan)


def find_pivots(values: np.ndarray, window: int, kind: str) -> np.ndarray:
    """Index of each pivot. A pivot at i is the strict extreme of
    [i-window, i+window]. Returned as *positions*, not signals - publishing
    them is the caller's job, and that is where the lag is applied.
    """
    n = len(values)
    out = []
    for i in range(window, n - window):
        seg = values[i - window: i + window + 1]
        v = values[i]
        if np.isnan(v) or np.isnan(seg).any():
            continue
        if kind == "low" and v == seg.min() and (seg == v).sum() == 1:
            out.append(i)
        elif kind == "high" and v == seg.max() and (seg == v).sum() == 1:
            out.append(i)
    return np.array(out, dtype=int)


def divergence_signals(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Return the frame with `rsi`, `bullish` and `bearish` columns.

    A signal on row i means: as of the close of bar i, a divergence has been
    confirmed. The backtest acts on bar i+1's open.
    """
    out = df.copy()
    out["rsi"] = rsi(out["close"], p.rsi_period)

    close = out["close"].to_numpy()
    low = out["low"].to_numpy()
    high = out["high"].to_numpy()
    r = out["rsi"].to_numpy()

    lows = find_pivots(low, p.pivot_window, "low")
    highs = find_pivots(high, p.pivot_window, "high")

    bullish = np.zeros(len(out), dtype=bool)
    bearish = np.zeros(len(out), dtype=bool)

    # Regular bullish divergence: price makes a lower low, RSI a higher low.
    for a, b in zip(lows[:-1], lows[1:]):
        gap = b - a
        if not (p.min_pivot_gap <= gap <= p.max_pivot_gap):
            continue
        if low[b] < low[a] and r[b] > r[a]:
            publish = b + p.pivot_window      # the pivot at b is only known here
            if publish < len(out):
                bullish[publish] = True

    # Regular bearish divergence: price makes a higher high, RSI a lower high.
    for a, b in zip(highs[:-1], highs[1:]):
        gap = b - a
        if not (p.min_pivot_gap <= gap <= p.max_pivot_gap):
            continue
        if high[b] > high[a] and r[b] < r[a]:
            publish = b + p.pivot_window
            if publish < len(out):
                bearish[publish] = True

    out["bullish"] = bullish
    out["bearish"] = bearish
    return out


def positions_from_signals(sig: pd.DataFrame) -> pd.Series:
    """Long-only position held from the bar AFTER a bullish signal until the
    bar AFTER the next bearish one.

    Returned as the position *held during* each bar, so combining it with that
    bar's return needs no further shifting - and cannot accidentally credit the
    strategy with the move that produced the signal.
    """
    n = len(sig)
    pos = np.zeros(n)
    in_market = False
    bull = sig["bullish"].to_numpy()
    bear = sig["bearish"].to_numpy()
    for i in range(n - 1):
        if not in_market and bull[i]:
            in_market = True
        elif in_market and bear[i]:
            in_market = False
        pos[i + 1] = 1.0 if in_market else 0.0
    return pd.Series(pos, index=sig.index, name="position")
