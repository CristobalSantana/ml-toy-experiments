"""
test_leakage.py -- Show the look-ahead checks actually fire.

A check that never triggers is decoration. These deliberately build the broken
version of the strategy - pivots published at the bar they occur, and entries
executed on the signal bar - and assert that each check catches it.

    python test_leakage.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import data_loader
import leakage
import strategy as S


def leaky_signals(df: pd.DataFrame, p: S.Params) -> pd.DataFrame:
    """The naive implementation: a pivot is marked at the bar where it occurs.

    This is what most divergence backtests do, and it is not implementable -
    at that bar nobody yet knows the following `window` bars failed to break it.
    """
    out = df.copy()
    out["rsi"] = S.rsi(out["close"], p.rsi_period)
    low, high = out["low"].to_numpy(), out["high"].to_numpy()
    r = out["rsi"].to_numpy()
    lows = S.find_pivots(low, p.pivot_window, "low")
    highs = S.find_pivots(high, p.pivot_window, "high")

    bull = np.zeros(len(out), dtype=bool)
    bear = np.zeros(len(out), dtype=bool)
    for a, b in zip(lows[:-1], lows[1:]):
        if p.min_pivot_gap <= b - a <= p.max_pivot_gap and low[b] < low[a] and r[b] > r[a]:
            bull[b] = True                      # published at the pivot itself
    for a, b in zip(highs[:-1], highs[1:]):
        if p.min_pivot_gap <= b - a <= p.max_pivot_gap and high[b] > high[a] and r[b] < r[a]:
            bear[b] = True
    out["bullish"], out["bearish"] = bull, bear
    return out


def same_bar_positions(sig: pd.DataFrame) -> pd.Series:
    """Enter and exit on the signal bar - the other classic inflation."""
    n = len(sig)
    pos = np.zeros(n)
    in_market = False
    bull, bear = sig["bullish"].to_numpy(), sig["bearish"].to_numpy()
    for i in range(n):
        if not in_market and bull[i]:
            in_market = True
        elif in_market and bear[i]:
            in_market = False
        pos[i] = 1.0 if in_market else 0.0
    return pd.Series(pos, index=sig.index, name="position")


def main() -> int:
    df = data_loader.load()
    p = S.Params()
    ok = []

    print("1. the real implementation passes both checks")
    try:
        leakage.run_all(df, p, verbose=True)
        print("   PASS")
        ok.append(True)
    except leakage.LookAheadError as e:
        print(f"   FAIL - the honest version should pass: {e}")
        ok.append(False)

    print("\n2. pivots published at the pivot bar must be caught")
    orig = S.divergence_signals
    S.divergence_signals = leaky_signals          # type: ignore[assignment]
    try:
        leakage.check_causal(df, p)
        print("   FAIL - look-ahead went undetected")
        ok.append(False)
    except leakage.LookAheadError as e:
        print(f"   PASS - {str(e).splitlines()[0]}")
        ok.append(True)
    finally:
        S.divergence_signals = orig               # type: ignore[assignment]

    print("\n3. entering on the signal bar must be caught")
    sig = S.divergence_signals(df, p)
    try:
        leakage.check_execution_lag(sig, same_bar_positions(sig))
        print("   FAIL - same-bar execution went undetected")
        ok.append(False)
    except leakage.LookAheadError as e:
        print(f"   PASS - {str(e).splitlines()[0][:96]}")
        ok.append(True)

    print("\n" + ("ALL CHECKS BEHAVE CORRECTLY" if all(ok) else "SOME CHECKS ARE BROKEN"))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
