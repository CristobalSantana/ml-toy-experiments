"""
search.py -- Backtest ~1,500 moving-average crossovers at once, and build
price series in which none of them can possibly work.

Why this is vectorised
----------------------
The experiment runs the whole grid on 400 surrogate series, which is about
600,000 backtests. One at a time with pandas `rolling` that takes hours; as
array arithmetic it takes a couple of minutes. Moving averages come from a
cumulative sum, so a 200-day window costs what a 2-day window costs.

The execution and cost model is btc-01's, exactly
-------------------------------------------------
Not approximately. A position entered is filled at that bar's **open**, a
position held earns **close to close**, and a position exited is filled at
that bar's **open**. 0.10% fee plus 0.05% slippage per side, charged on every
change. `test_search.py` asserts that this file reproduces btc-01's published
buy-and-hold return to four decimal places on the same data and the same
split - if it does not, the two experiments are not comparable and the
comparison this one is built on means nothing.

The surrogates
--------------
Shuffling a close-price series would destroy the relationship between a
bar's open and its close, which is exactly what the execution model above
depends on. So the surrogates permute **whole bars**, each carrying its own
overnight gap and its own open-to-close move. Serial structure is destroyed;
everything inside a bar survives untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEE = 0.0010          # Binance spot taker, as in btc-01
SLIPPAGE = 0.0005
COST_PER_SIDE = FEE + SLIPPAGE


def grid(fast_max: int = 40, slow_max: int = 200, slow_step: int = 5
         ) -> np.ndarray:
    """Every (fast, slow) pair with slow > fast. Shape (K, 2)."""
    fasts = np.arange(2, fast_max + 1)
    slows = np.arange(slow_step, slow_max + 1, slow_step)
    return np.array([(f, s) for f in fasts for s in slows if s > f],
                    dtype=np.int64)


def moving_averages(price: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """(n, W) trailing means, NaN until each window is filled.

    The NaN prefix is not padding. It marks bars where the average does not
    exist yet, and the caller must stay flat there rather than trade on a
    partially filled window.
    """
    n = len(price)
    cs = np.concatenate([[0.0], np.cumsum(price)])
    out = np.full((n, len(windows)), np.nan)
    for j, w in enumerate(windows):
        if w > n:
            continue
        idx = np.arange(w - 1, n)
        out[idx, j] = (cs[idx + 1] - cs[idx + 1 - w]) / w
    return out


def positions(close: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """(n, K) position held during each bar, decided on the previous close.

    The shift is the whole game. `signal` is what the rule says looking at
    bar t; `pos` is what is actually held over bar t+1. Dropping the shift
    produces a backtest that trades on information it could not have had, and
    `test_search.py` builds that version deliberately to show what it buys.
    """
    windows = np.unique(pairs)
    ma = moving_averages(close, windows)
    col = {w: j for j, w in enumerate(windows)}
    fast = ma[:, [col[w] for w in pairs[:, 0]]]
    slow = ma[:, [col[w] for w in pairs[:, 1]]]
    signal = np.where(np.isnan(fast) | np.isnan(slow), 0.0,
                      (fast > slow).astype(np.float64))
    return np.vstack([np.zeros((1, signal.shape[1])), signal[:-1]])


@dataclass
class GridResult:
    total_return: np.ndarray      # (K,)
    n_trades: np.ndarray          # (K,)
    exposure: np.ndarray          # (K,)


def run_positions(open_: np.ndarray, close: np.ndarray, pos: np.ndarray
                  ) -> GridResult:
    """Net return of every column of `pos`, on one price series.

    The three transition returns below are properties of the bar, not of the
    strategy, so they are computed once and selected between - which is what
    makes a 1,500-wide grid cost the same as a single backtest.
    """
    n = len(close)
    r_enter = np.zeros(n)
    r_hold = np.zeros(n)
    r_exit = np.zeros(n)
    r_enter[1:] = close[1:] / open_[1:] - 1.0        # filled at this open
    r_hold[1:] = close[1:] / close[:-1] - 1.0        # already in
    r_exit[1:] = open_[1:] / close[:-1] - 1.0        # out at this open

    prev, cur = pos[:-1], pos[1:]
    gross = np.zeros_like(pos)
    gross[1:] = np.where((prev == 0) & (cur == 1), r_enter[1:, None],
                np.where((prev == 1) & (cur == 1), r_hold[1:, None],
                np.where((prev == 1) & (cur == 0), r_exit[1:, None], 0.0)))

    turnover = np.abs(np.diff(pos, axis=0,
                              prepend=np.zeros((1, pos.shape[1]))))
    net = gross - turnover * COST_PER_SIDE
    return GridResult(total_return=np.prod(1.0 + net, axis=0) - 1.0,
                      n_trades=turnover.sum(axis=0),
                      exposure=pos.mean(axis=0))


def run_grid(open_: np.ndarray, close: np.ndarray, pairs: np.ndarray
             ) -> GridResult:
    return run_positions(open_, close, positions(close, pairs))


def buy_and_hold(open_: np.ndarray, close: np.ndarray) -> float:
    """btc-01's benchmark: bought at the first open, held to the last close,
    charged exactly one entry so the comparison is not rigged in its favour.
    """
    r = np.zeros(len(close))
    r[0] = close[0] / open_[0] - 1.0 - COST_PER_SIDE
    r[1:] = close[1:] / close[:-1] - 1.0
    return float(np.prod(1.0 + r) - 1.0)


# --------------------------------------------------------------------------
# surrogates: price series where no strategy can have an edge
# --------------------------------------------------------------------------

def _bar_factors(open_: np.ndarray, close: np.ndarray):
    """Each bar as (overnight gap, open-to-close move), both multiplicative.

    Bar 0 has no previous close, so its gap is defined as 1 and it carries
    only its own intrabar move.
    """
    gap = np.ones(len(close))
    gap[1:] = open_[1:] / close[:-1]
    intra = close / open_
    return gap, intra


def _rebuild(first_open: float, gap: np.ndarray, intra: np.ndarray):
    """Turn a sequence of bar factors back into (open, close) arrays."""
    n = len(gap)
    open_ = np.empty(n)
    close = np.empty(n)
    open_[0] = first_open
    close[0] = open_[0] * intra[0]
    for i in range(1, n):
        open_[i] = close[i - 1] * gap[i]
        close[i] = open_[i] * intra[i]
    return open_, close


def shuffle_surrogate(open_: np.ndarray, close: np.ndarray,
                      rng: np.random.Generator):
    """Whole bars, in a random order.

    Every serial structure is destroyed and every bar's internal shape is
    preserved. The multiset of bar returns is untouched, so the asset still
    went up by the same amount over the period - only the order in which it
    did so has changed, and order is the only thing a timing rule can use.
    """
    gap, intra = _bar_factors(open_, close)
    perm = rng.permutation(len(gap))
    return _rebuild(open_[0], gap[perm], intra[perm])


def block_surrogate(open_: np.ndarray, close: np.ndarray,
                    rng: np.random.Generator, block: int = 20):
    """Moving-block bootstrap: keeps volatility clustering, breaks the rest.

    A harder null than `shuffle`, because trending and quiet stretches
    survive inside each block. Blocks are drawn with replacement, so the
    asset's total move is no longer fixed and buy-and-hold has to be
    recomputed for every surrogate.
    """
    gap, intra = _bar_factors(open_, close)
    n = len(gap)
    starts = rng.integers(0, max(n - block, 1), size=int(np.ceil(n / block)))
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    idx = np.clip(idx, 0, n - 1)
    return _rebuild(open_[0], gap[idx], intra[idx])
