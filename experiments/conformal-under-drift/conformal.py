"""
conformal.py -- Split conformal prediction, with the correction that makes the
guarantee exact.

The method is three lines of arithmetic on top of any regressor:

    residuals = |y_cal - yhat_cal|
    q         = the ceil((n+1)(1-alpha))/n empirical quantile of them
    interval  = yhat +- q

and it gives, for exchangeable data, coverage of at least 1 - alpha in finite
samples, for any distribution and any model.

The ceiling is not decoration
-----------------------------
The finite-sample guarantee needs the ((n+1)(1-alpha))-th smallest residual,
not the plain (1-alpha) quantile. With n = 50,000 the difference is a fraction
of a cent and nobody would notice; with n = 200 it is the difference between a
theorem and a heuristic. `test_conformal.py` builds the naive version and
measures the gap at small calibration sizes, because that is where the people
who need the guarantee most are operating.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Interval:
    lower: np.ndarray
    upper: np.ndarray
    half_width: float

    def covers(self, y: np.ndarray) -> np.ndarray:
        return (y >= self.lower) & (y <= self.upper)


def conformal_quantile(residuals: np.ndarray, alpha: float,
                       finite_sample: bool = True) -> float:
    """The half-width that makes the interval valid.

    finite_sample=False drops the (n+1) correction. It exists so the test
    file can measure what dropping it costs rather than assert that it
    matters.
    """
    r = np.sort(np.asarray(residuals, dtype=float))
    n = len(r)
    if n == 0:
        raise ValueError("no calibration residuals")
    if finite_sample:
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        if k > n:
            # too few calibration points to certify this alpha at all; the
            # honest answer is an infinite interval, not a silently capped one
            return float("inf")
        return float(r[k - 1])
    return float(np.quantile(r, 1.0 - alpha))


def calibrate(y_cal: np.ndarray, pred_cal: np.ndarray, alpha: float,
              finite_sample: bool = True) -> float:
    return conformal_quantile(np.abs(np.asarray(y_cal) - np.asarray(pred_cal)),
                              alpha, finite_sample)


def predict_interval(pred: np.ndarray, half_width: float,
                     lower_bound: float | None = None) -> Interval:
    """Interval around a point prediction.

    `lower_bound` clips the lower edge, for targets that cannot go below some
    value - a fare cannot be negative. It defaults to None, and the default
    matters: raising the lower edge **narrows** the interval and can only
    remove coverage. An earlier version defaulted it to 0 with a comment
    claiming the opposite, and the synthetic check caught it immediately -
    coverage came out at 0.798 instead of 0.900, because roughly a tenth of
    the test targets were negative and were being excluded by the clip.

    Safe only when no true value can fall below the bound, which is a fact
    about the task and therefore belongs at the call site.
    """
    pred = np.asarray(pred, dtype=float)
    lo = pred - half_width
    if lower_bound is not None:
        lo = np.maximum(lo, lower_bound)
    return Interval(lower=lo, upper=pred + half_width, half_width=half_width)


def coverage_report(y: np.ndarray, pred: np.ndarray, half_width: float,
                    alpha: float, lower_bound: float | None = None) -> dict:
    iv = predict_interval(pred, half_width, lower_bound)
    cov = float(iv.covers(np.asarray(y)).mean())
    return {"coverage": cov,
            "target": 1.0 - alpha,
            "coverage_gap": (1.0 - alpha) - cov,
            "mean_width": float((iv.upper - iv.lower).mean()),
            "half_width": float(half_width),
            "n": int(len(y))}
