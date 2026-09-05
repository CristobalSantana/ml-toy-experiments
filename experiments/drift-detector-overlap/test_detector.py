"""
test_detector.py -- Prove the detector is a test and not a rubber stamp.

    python test_detector.py

Five checks. The middle one is the reason this file exists: it builds the
in-sample version of the detector on purpose - the same mistake anyone makes
by scoring a classifier on its training rows - and asserts that it reports
drift between two samples drawn from the same distribution. A detector that
cannot be caught doing this is not being tested.

If any check fails, `run_all.py` stops. Numbers from a detector that flags
identical samples are not worth computing.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier

from detector import c2st, delta_for_overlap, empirical_overlap, gaussian_pair, mcc

SEED = 20260905
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if detail:
        print(f"        {detail}")
    if not ok:
        FAILURES.append(label)


def test_overlap_formula() -> None:
    """delta_for_overlap must invert the empirical overlap it claims to set."""
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for target in (0.2, 0.5, 0.8, 0.95):
        d = delta_for_overlap(target)
        a = rng.normal(0.0, 1.0, 400_000)
        b = rng.normal(d, 1.0, 400_000)
        got = empirical_overlap(a, b, bins=200)
        worst = max(worst, abs(got - target))
    check("analytic overlap matches the measured overlap", worst < 0.01,
          f"largest gap {worst:.4f} across OVL in (0.2 .. 0.95)")


def test_identical_not_flagged() -> None:
    """The null. Two samples from one distribution must not be called drift."""
    rng = np.random.default_rng(SEED + 1)
    flagged = 0
    trials = 40
    for i in range(trials):
        a = rng.normal(0.0, 1.0, (2000, 1))
        b = rng.normal(0.0, 1.0, (2000, 1))
        flagged += c2st(a, b, seed=SEED + i).drift
    rate = flagged / trials
    check("identical distributions are not flagged", rate <= 0.20,
          f"false-alarm rate {rate:.0%} over {trials} trials at alpha = 0.05")


def test_obvious_drift_is_flagged() -> None:
    """The other end. A separation this large must be caught every time."""
    rng = np.random.default_rng(SEED + 2)
    caught = 0
    trials = 20
    for i in range(trials):
        a, b = gaussian_pair(2000, 1, ovl=0.10, rng=rng)
        caught += c2st(a, b, seed=SEED + i).drift
    check("a large, obvious shift is always flagged", caught == trials,
          f"caught {caught}/{trials} at OVL = 0.10")


def test_in_sample_scoring_is_caught() -> None:
    """The deliberate mistake.

    Score the classifier on the rows it trained on and it separates two
    identical samples almost perfectly. This is the failure the held-out split
    exists to prevent, so it is built here and asserted to be visible.
    """
    rng = np.random.default_rng(SEED + 3)
    a = rng.normal(0.0, 1.0, (2000, 8))
    b = rng.normal(0.0, 1.0, (2000, 8))
    X = np.vstack([a, b])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]

    model = HistGradientBoostingClassifier(max_iter=100, max_depth=6,
                                           early_stopping=False, random_state=SEED)
    model.fit(X, y)
    correct = int((model.predict(X) == y).sum())
    in_acc = correct / len(y)
    in_p = float(stats.binomtest(correct, len(y), 0.5, alternative="greater").pvalue)

    out = c2st(a, b, seed=SEED)

    # The assertion is on the *decision*, not on an accuracy threshold. An
    # earlier version of this test demanded in-sample accuracy above 0.75 and
    # failed at 0.586 - a number picked by guess. What matters is that the
    # in-sample version calls drift on two identical samples while the
    # held-out version does not.
    check("scoring in sample fakes drift where there is none",
          in_p < 0.05 and not out.drift,
          f"in-sample {in_acc:.3f} (p={in_p:.2g}, drift called) vs "
          f"held-out {out.accuracy:.3f} (p={out.p_value:.2f}, no drift) "
          f"- both samples from the SAME distribution")


def test_mcc_rejects_always_yes() -> None:
    """A detector that always says drift scores 0.5 accuracy and 0 MCC."""
    truth = np.r_[np.ones(50), np.zeros(50)].astype(bool)
    always = np.ones(100, dtype=bool)
    perfect = truth.copy()
    check("MCC gives no credit to a detector that always says drift",
          abs(mcc(truth, always)) < 1e-9 and mcc(truth, perfect) == 1.0,
          f"always-yes MCC {mcc(truth, always):.3f}, perfect MCC "
          f"{mcc(truth, perfect):.3f}")


def main() -> None:
    print("Detector self-checks\n")
    test_overlap_formula()
    test_identical_not_flagged()
    test_obvious_drift_is_flagged()
    test_in_sample_scoring_is_caught()
    test_mcc_rejects_always_yes()

    print()
    if FAILURES:
        sys.exit(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
    print("All checks passed. The detector holds its false-alarm rate, catches "
          "a real shift,\nand the in-sample version of itself is visibly broken.")


if __name__ == "__main__":
    main()
