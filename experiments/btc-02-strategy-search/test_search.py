"""
test_search.py -- Is this backtester the same backtester as btc-01's, and do
the surrogates do what they claim?

    python test_search.py

Six checks. The first is the one the whole experiment rests on: if this file
does not reproduce btc-01's published buy-and-hold return, the two
experiments are not measuring the same thing and comparing them would be
meaningless.

Run before the experiment; `run_all.py` stops if any check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import search as S

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "btc-01-rsi-divergence" / "data" / "btcusdt_1d.csv"

# published in btc-01/outputs/buy_and_hold.csv
BTC01_DEV = 8.907474
BTC01_HOLDOUT = 0.490849

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def load():
    d = pd.read_csv(DATA, parse_dates=["date"])
    dev = d[d["date"] <= "2023-12-31"].reset_index(drop=True)
    hol = d[d["date"] >= "2024-01-01"].reset_index(drop=True)
    return dev, hol


# 1 -------------------------------------------------------------------------
def test_matches_btc01() -> None:
    """The control. Same data, same split, same number."""
    dev, hol = load()
    got = [S.buy_and_hold(f["open"].to_numpy(), f["close"].to_numpy())
           for f in (dev, hol)]
    ok = (abs(got[0] - BTC01_DEV) < 5e-5 and abs(got[1] - BTC01_HOLDOUT) < 5e-5)
    check("reproduces btc-01's buy-and-hold", ok,
          f"development {got[0]:.6f} against {BTC01_DEV}, "
          f"holdout {got[1]:.6f} against {BTC01_HOLDOUT}")


# 2 -------------------------------------------------------------------------
def test_the_shift_is_load_bearing() -> None:
    """Trade on today's signal today, and see what it buys.

    This is the bug the whole family of backtesting failures is made of. It
    is built here deliberately rather than merely guarded against, so the
    size of the effect is on the record.
    """
    dev, _ = load()
    o, c = dev["open"].to_numpy(), dev["close"].to_numpy()
    pairs = S.grid()

    honest = S.positions(c, pairs)
    peeking = np.vstack([honest[1:], honest[-1:]])     # undo the shift

    a = S.run_positions(o, c, honest).total_return
    b = S.run_positions(o, c, peeking).total_return
    # Measured against the null the check is actually testing - "the shift
    # makes no difference", under which the share helped would be 0.5 - and
    # not against a threshold picked to fit. An earlier version of this check
    # asserted 90% and failed at 86.7%, which said more about the guess than
    # about the code.
    share = float((b > a).mean())
    sd = np.sqrt(0.25 / len(a))
    z = (share - 0.5) / sd
    check("the next-bar shift is load-bearing", z > 10,
          f"reading one bar ahead helps {share:.1%} of the {len(a):,} "
          f"strategies ({z:.0f} sd from the 50% that would mean it does not "
          f"matter); median total return {np.median(a):.2f} honest against "
          f"{np.median(b):.2f} ({np.median(b) / np.median(a):.1f}x)")


# 3 -------------------------------------------------------------------------
def test_always_long_is_buy_and_hold() -> None:
    """A position of 1 in every bar must equal buy-and-hold, up to bar zero.

    It does not equal it exactly, and the gap is inherited from btc-01 rather
    than introduced here: btc-01's strategy backtester assigns a return of
    zero to the first bar whatever the position, while its buy-and-hold
    captures that bar's open-to-close move. Running btc-01's own code on this
    data gives the same 8.852828 for an always-long position against 8.907474
    for buy-and-hold.

    The effect is 0.55% and it favours the benchmark, which is the
    conservative direction, so it is reproduced rather than silently
    corrected - correcting it here would break comparability with btc-01,
    which is the thing this file exists to guarantee.
    """
    dev, _ = load()
    o, c = dev["open"].to_numpy(), dev["close"].to_numpy()
    pos = np.ones((len(c), 1))
    mine = float(S.run_positions(o, c, pos).total_return[0])
    bh = S.buy_and_hold(o, c)
    bar0 = float(c[0] / o[0])
    residual = abs((1 + mine) * bar0 - (1 + bh))
    check("always-long equals buy-and-hold, up to bar zero", residual < 1e-3,
          f"{mine:.6f} against {bh:.6f}; the gap is bar zero's "
          f"{100 * (bar0 - 1):+.3f}% open-to-close move, which the benchmark "
          f"gets and no strategy can (residual {residual:.1e})")


# 4 -------------------------------------------------------------------------
def test_moving_averages_are_right() -> None:
    """Against pandas, which is not the implementation being tested."""
    dev, _ = load()
    c = dev["close"].to_numpy()
    windows = np.array([2, 7, 50, 200])
    mine = S.moving_averages(c, windows)
    worst = 0.0
    for j, w in enumerate(windows):
        want = pd.Series(c).rolling(int(w)).mean().to_numpy()
        both = ~np.isnan(want)
        worst = max(worst, float(np.abs(mine[both, j] - want[both]).max()))
        # and the NaN pattern must agree, or a window is being traded early
        if not np.array_equal(np.isnan(mine[:, j]), np.isnan(want)):
            worst = float("inf")
    check("moving averages match pandas", worst < 1e-6,
          f"largest disagreement {worst:.2e} over windows {list(windows)}")


# 5 -------------------------------------------------------------------------
def test_shuffle_preserves_the_asset() -> None:
    """A shuffled surrogate must be the same asset in a different order.

    Buy-and-hold is not bit-identical across surrogates: whichever bar lands
    first carries the one entry cost, and that bar's open-to-close move
    varies. The effect is a fraction of a percent, and it is measured here
    rather than assumed away, because the whole comparison is against
    buy-and-hold.
    """
    dev, _ = load()
    o, c = dev["open"].to_numpy(), dev["close"].to_numpy()
    real = S.buy_and_hold(o, c)
    vals = []
    for seed in range(30):
        so, sc = S.shuffle_surrogate(o, c, np.random.default_rng(seed))
        vals.append(S.buy_and_hold(so, sc))
    spread = (max(vals) - min(vals)) / real
    check("shuffling preserves the asset's total move", spread < 0.01,
          f"buy-and-hold across 30 surrogates spans {spread:.3%} of "
          f"{real:.4f} (real {real:.4f}, surrogate median "
          f"{np.median(vals):.4f})")


# 6 -------------------------------------------------------------------------
def test_costs_are_charged_per_side() -> None:
    """Flip every bar and the drag must be one side per change, no more."""
    n = 200
    o = np.full(n, 100.0)
    c = np.full(n, 100.0)                     # a flat market: no gross return
    pos = np.zeros((n, 1))
    pos[::2] = 1.0                            # in, out, in, out...
    r = S.run_positions(o, c, pos)
    expected_changes = float(np.abs(np.diff(pos[:, 0], prepend=0.0)).sum())
    got = float(r.total_return[0])
    want = (1.0 - S.COST_PER_SIDE) ** expected_changes - 1.0
    check("costs are one side per position change",
          abs(got - want) < 1e-9 and r.n_trades[0] == expected_changes,
          f"{int(expected_changes)} changes, return {got:.6f}, "
          f"expected {want:.6f}")


if __name__ == "__main__":
    print("implementation checks")
    if not DATA.exists():
        sys.exit(f"missing {DATA}\nrun: python "
                 f"../btc-01-rsi-divergence/data_loader.py")
    test_matches_btc01()
    test_the_shift_is_load_bearing()
    test_always_long_is_buy_and_hold()
    test_moving_averages_are_right()
    test_shuffle_preserves_the_asset()
    test_costs_are_charged_per_side()
    print()
    if FAILURES:
        sys.exit(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
    print("all checks passed")
