"""
test_leakage.py -- The feature that was removed, and what it would have bought.

    python test_leakage.py

This experiment's leak is not temporal. Nothing reads the future, and a
chronological split does not catch it. `tip_amount` is zero for every cash
trip because the TLC records card tips and cannot see cash ones - the label is
written into the feature by the way the data was collected.

That kind of leak is only ever caught by knowing where the numbers came from.
So rather than assert it, this file measures it: fit the same model with the
feature and without, and report the gap.

Four checks. run_all.py stops if any fail.
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score

import data as D

SEED = 20260905
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if detail:
        print(f"        {detail}")
    if not ok:
        FAILURES.append(label)


def _fit_score(tr, te, cols):
    m = HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                       early_stopping=False, random_state=SEED)
    m.fit(tr[cols].to_numpy(), tr["is_cash"].to_numpy())
    p = m.predict(te[cols].to_numpy())
    return balanced_accuracy_score(te["is_cash"].to_numpy(), p)


def main() -> None:
    print("Leakage demonstration\n")
    tr, te = D.build("2024-05", "2024-06", test_size=60000, seed=SEED,
                     with_leaky=True, verbose=False)
    tr = D.subsample(tr, 60000, SEED)

    # ---- 1. the mechanism, stated as a measurement -----------------------
    cash_tip_zero = float((tr.loc[tr.is_cash == 1, "tip_amount"] == 0).mean())
    card_tip_zero = float((tr.loc[tr.is_cash == 0, "tip_amount"] == 0).mean())
    check("tip_amount is zero for every cash trip and few card trips",
          cash_tip_zero > 0.999 and card_tip_zero < 0.20,
          f"zero tip: {cash_tip_zero:.3%} of cash trips, "
          f"{card_tip_zero:.1%} of card trips")

    # ---- 2. the excluded feature is not in the task ----------------------
    check("the task does not expose tip_amount or total_amount",
          all(c not in D.FEATURES for c in D.LEAKY),
          f"features: {D.FEATURES}")

    # ---- 3. what the leak buys ------------------------------------------
    honest = _fit_score(tr, te, D.FEATURES)
    leaked = _fit_score(tr, te, D.FEATURES + ["tip_amount"])
    check("including tip_amount turns the problem into a lookup",
          leaked > honest + 0.15,
          f"balanced accuracy {honest:.3f} honest vs {leaked:.3f} with "
          f"tip_amount - a gap of {leaked - honest:+.3f}")

    # ---- 4. total_amount carries it second-hand -------------------------
    second = _fit_score(tr, te, D.FEATURES + ["total_amount"])
    check("total_amount carries the same contamination",
          second > honest + 0.03,
          f"balanced accuracy {honest:.3f} honest vs {second:.3f} with "
          f"total_amount (= fare + extras + tip)")

    print()
    if FAILURES:
        sys.exit(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
    print("All checks passed. The leak is real, measured, and excluded.")


if __name__ == "__main__":
    main()
