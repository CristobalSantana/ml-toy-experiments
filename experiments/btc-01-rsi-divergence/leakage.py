"""
leakage.py -- Prove the signals cannot see the future.

The decisive test is not reading the code, it is this: compute the signals on
the full history, then recompute them on the history truncated at bar k. If
any signal at or before k changes, the original run was using data that had not
happened yet.

A comment claiming "this is causal" is worth nothing. This is worth something.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import strategy as S


class LookAheadError(AssertionError):
    """Raised when a signal depends on data later than its own bar."""


def check_causal(df: pd.DataFrame, p: S.Params, cuts: int = 12, seed: int = 0) -> dict:
    """Recompute signals on truncated histories and compare.

    Returns a small report; raises LookAheadError on any mismatch.
    """
    full = S.divergence_signals(df, p)
    n = len(df)
    rng = np.random.default_rng(seed)
    # cut points spread over the second half, where enough history exists
    points = sorted(rng.choice(np.arange(int(n * 0.5), n - 1), size=cuts, replace=False))

    mismatches = []
    for k in points:
        partial = S.divergence_signals(df.iloc[: k + 1], p)
        for col in ("bullish", "bearish"):
            a = full[col].to_numpy()[: k + 1]
            b = partial[col].to_numpy()
            if not np.array_equal(a, b):
                where = np.flatnonzero(a != b)
                mismatches.append((int(k), col, where[:5].tolist(), int(len(where))))

    if mismatches:
        lines = [f"  cut at bar {k}: '{col}' differs at bars {w} ({m} bar(s) total)"
                 for k, col, w, m in mismatches[:6]]
        raise LookAheadError(
            "Signals change when later data is removed, so they were using it:\n"
            + "\n".join(lines))

    return {"cuts_tested": len(points),
            "bullish_signals": int(full["bullish"].sum()),
            "bearish_signals": int(full["bearish"].sum())}


def check_execution_lag(sig: pd.DataFrame, pos: pd.Series) -> None:
    """A position must never change on the same bar as its own signal.

    Entering on the bar that produced the signal quietly books the move the
    signal was made of, which is the second way these backtests inflate.
    """
    changes = pos.diff().fillna(0.0).to_numpy()
    bull = sig["bullish"].to_numpy()
    bear = sig["bearish"].to_numpy()
    same_bar = np.flatnonzero(((changes > 0) & bull) | ((changes < 0) & bear))
    if len(same_bar):
        raise LookAheadError(
            f"{len(same_bar)} position change(s) happen on the signal bar itself "
            f"(e.g. bars {same_bar[:5].tolist()}); execution must be the next bar.")


def run_all(df: pd.DataFrame, p: S.Params, verbose: bool = True) -> dict:
    report = check_causal(df, p)
    sig = S.divergence_signals(df, p)
    check_execution_lag(sig, S.positions_from_signals(sig))
    if verbose:
        print(f"  causality: signals unchanged across {report['cuts_tested']} truncations")
        print(f"  execution: no position changes on the signal bar")
        print(f"  signals: {report['bullish_signals']} bullish, "
              f"{report['bearish_signals']} bearish")
    return report
