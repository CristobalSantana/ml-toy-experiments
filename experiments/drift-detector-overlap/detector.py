"""
detector.py -- The classifier two-sample test, and the overlap measures it is
plotted against.

The detector trains a model to tell a reference sample from a current one and
asks whether it does better than chance. Everything that makes that a test
rather than a demonstration lives in one place here, so it is the same object
in all three arms.

The one rule that cannot be relaxed
-----------------------------------
The classifier is scored **only on rows it did not train on**. Scored in
sample, a gradient-boosted tree separates two samples drawn from the *same*
distribution nearly perfectly, and the detector reports drift everywhere. That
is this experiment's version of look-ahead, and `test_detector.py` builds the
in-sample version on purpose to confirm it is caught.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------
# overlap
# --------------------------------------------------------------------------

def delta_for_overlap(ovl: float, sigma: float = 1.0) -> float:
    """Mean separation of two equal-variance Gaussians with a given overlap.

    OVL = 2 * Phi(-delta / 2 sigma), so delta = -2 sigma * Phi^-1(OVL / 2).
    Setting the overlap exactly is what makes the synthetic arm a controlled
    sweep rather than a search: at OVL = 1 the separation is 0, and the two
    samples are the same distribution.
    """
    if not 0.0 < ovl <= 1.0:
        raise ValueError(f"overlap must be in (0, 1], got {ovl}")
    return float(-2.0 * sigma * stats.norm.ppf(ovl / 2.0))


def empirical_overlap(x: np.ndarray, y: np.ndarray, bins: int = 100) -> float:
    """Overlapping coefficient of two samples: the area their densities share.

    Used on real data, where no analytic form applies. Both samples are
    histogrammed on one shared range so the bin edges cannot differ between
    them - separate ranges would make the two densities incomparable and the
    result meaningless.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")

    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    hx, _ = np.histogram(x, bins=edges, density=True)
    hy, _ = np.histogram(y, bins=edges, density=True)
    return float(np.minimum(hx, hy).sum() * (edges[1] - edges[0]))


# --------------------------------------------------------------------------
# the test
# --------------------------------------------------------------------------

@dataclass
class C2STResult:
    accuracy: float          # held-out
    p_value: float           # one-sided binomial against 0.5
    drift: bool              # p < alpha
    n_test: int


def c2st(a: np.ndarray, b: np.ndarray, *, seed: int, max_iter: int = 100,
         max_depth: int = 6, learning_rate: float = 0.1,
         early_stopping: bool = False, test_size: float = 0.5,
         alpha: float = 0.05) -> C2STResult:
    """Classifier two-sample test on two samples of the same feature set.

    Returns held-out accuracy and the p-value of a one-sided binomial test
    against chance. Under the null the classifier cannot beat 0.5, so the
    number of correct held-out predictions is Binomial(n_test, 0.5).
    """
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    if a.ndim == 2 and a.shape[0] == 1 and a.shape[1] != b.shape[1]:
        a, b = a.T, b.T
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"feature mismatch: {a.shape[1]} vs {b.shape[1]}")

    X = np.vstack([a, b])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)

    model = HistGradientBoostingClassifier(
        max_iter=max_iter, max_depth=max_depth, learning_rate=learning_rate,
        early_stopping=early_stopping, random_state=seed)
    model.fit(Xtr, ytr)

    correct = int((model.predict(Xte) == yte).sum())
    n_test = int(len(yte))
    acc = correct / n_test
    # one-sided: only "better than chance" is evidence of drift
    p = float(stats.binomtest(correct, n_test, 0.5, alternative="greater").pvalue)
    return C2STResult(accuracy=acc, p_value=p, drift=bool(p < alpha), n_test=n_test)


def mcc(truth: np.ndarray, called: np.ndarray) -> float:
    """Matthews correlation coefficient of a detector's binary decisions.

    Accuracy is not enough here: the trials are balanced by construction, so a
    detector that answers "drift" every time still scores 0.5. MCC sends that
    to 0, which is the honest description of a detector that has learned to
    always say yes.
    """
    t = np.asarray(truth, dtype=bool)
    c = np.asarray(called, dtype=bool)
    tp = int((t & c).sum())
    tn = int((~t & ~c).sum())
    fp = int((~t & c).sum())
    fn = int((t & ~c).sum())
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0.0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def gaussian_pair(n: int, d: int, ovl: float, rng: np.random.Generator
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Two samples whose shifted feature has exactly the requested overlap.

    The shift goes into feature 0 only. With d > 1 the remaining features are
    identical noise, so the marginal overlap is unchanged and the difference
    between the two sweeps is purely the cost of having to find the signal
    among distractors.
    """
    delta = delta_for_overlap(ovl)
    a = rng.normal(0.0, 1.0, size=(n, d))
    b = rng.normal(0.0, 1.0, size=(n, d))
    b[:, 0] += delta
    return a, b
