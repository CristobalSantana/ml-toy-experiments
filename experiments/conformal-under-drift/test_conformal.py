"""
test_conformal.py -- Check the guarantee holds where it is supposed to.

    python test_conformal.py

Four checks on synthetic data, where exchangeability is true by construction
and any failure is the implementation rather than the world.

The third is the one worth reading: it builds the version without the
finite-sample correction - the one most people write first - and measures how
much coverage that costs at a small calibration set. The theorem is exact; the
common implementation of it is not.

run_all.py stops if any fail.
"""

from __future__ import annotations

import sys

import numpy as np

from conformal import calibrate, coverage_report

SEED = 20260905
ALPHA = 0.1
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if detail:
        print(f"        {detail}")
    if not ok:
        FAILURES.append(label)


def _draw(n: int, rng, heavy: bool = False):
    """y = f(x) + noise. Heavy-tailed on request, to check the guarantee does
    not quietly depend on the noise being Gaussian."""
    x = rng.uniform(0, 10, (n, 3))
    signal = 3 * x[:, 0] + 0.5 * x[:, 1] ** 2 - 2 * x[:, 2]
    noise = rng.standard_t(2, n) * 2.0 if heavy else rng.normal(0, 3, n)
    return x, signal + noise


def _fit_predict(rng, n_cal: int, heavy: bool = False, finite_sample=True):
    from sklearn.ensemble import HistGradientBoostingRegressor
    Xf, yf = _draw(4000, rng, heavy)
    Xc, yc = _draw(n_cal, rng, heavy)
    Xe, ye = _draw(20000, rng, heavy)
    m = HistGradientBoostingRegressor(max_iter=150, random_state=SEED).fit(Xf, yf)
    q = calibrate(yc, m.predict(Xc), ALPHA, finite_sample=finite_sample)
    return coverage_report(ye, m.predict(Xe), q, ALPHA)


def test_covers_when_exchangeable() -> None:
    """The core claim, on data that satisfies the assumption."""
    rng = np.random.default_rng(SEED)
    covs = [_fit_predict(rng, 5000)["coverage"] for _ in range(5)]
    mean = float(np.mean(covs))
    check("coverage holds at 90% when the data is exchangeable",
          0.885 <= mean <= 0.915,
          f"mean coverage {mean:.4f} over 5 runs "
          f"({', '.join(f'{c:.3f}' for c in covs)})")


def test_holds_under_heavy_tails() -> None:
    """Distribution-free means distribution-free: no Gaussian assumption."""
    rng = np.random.default_rng(SEED + 1)
    covs = [_fit_predict(rng, 5000, heavy=True)["coverage"] for _ in range(5)]
    mean = float(np.mean(covs))
    check("coverage survives heavy-tailed noise (Student t, 2 df)",
          0.88 <= mean <= 0.92,
          f"mean coverage {mean:.4f} - the guarantee does not assume a shape")


def test_finite_sample_correction_matters() -> None:
    """What dropping the (n+1) correction costs, where it is felt.

    At n = 50,000 the two versions differ by a rounding error. At n = 50 the
    naive quantile under-covers, which is precisely the regime where somebody
    reached for a guarantee because they had little data.
    """
    rng = np.random.default_rng(SEED + 2)
    small_ok = np.mean([_fit_predict(rng, 50, finite_sample=True)["coverage"]
                        for _ in range(20)])
    small_naive = np.mean([_fit_predict(rng, 50, finite_sample=False)["coverage"]
                           for _ in range(20)])
    check("the finite-sample correction is what makes small calibration sets valid",
          small_ok > small_naive,
          f"n_cal = 50: corrected {small_ok:.4f} vs naive {small_naive:.4f} "
          f"(target 0.900) - a gap of {small_ok - small_naive:+.4f}")


def test_too_few_points_refuses() -> None:
    """Below 1/alpha calibration points the level cannot be certified at all.

    The honest answer is an infinite interval. Returning the largest observed
    residual instead would look like a valid interval and would not be one.
    """
    rng = np.random.default_rng(SEED + 3)
    q = calibrate(rng.normal(size=5), np.zeros(5), alpha=0.1)
    check("too few calibration points returns an infinite width, not a guess",
          np.isinf(q), f"n_cal = 5 at alpha = 0.1 -> half-width {q}")


def main() -> None:
    print("Conformal implementation checks\n")
    test_covers_when_exchangeable()
    test_holds_under_heavy_tails()
    test_finite_sample_correction_matters()
    test_too_few_points_refuses()
    print()
    if FAILURES:
        sys.exit(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
    print("All checks passed. The guarantee holds where its assumption does.")


if __name__ == "__main__":
    main()
