"""
models.py -- The baselines, the models, and the clear-sky envelope they share.

The baselines are not filler. B1 (climatology) is the instrument that measures
how much of a solar forecast is the calendar, and B2 (smart persistence) is
the denominator of every skill score. The learned models are here to be
compared against them, not the other way round.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------
# clear-sky envelope
# --------------------------------------------------------------------------

@dataclass
class ClearSky:
    """The capacity factor a cloudless hour would produce, by sun elevation.

    Empirical rather than a physical model: the high quantile of observed
    output within each elevation bin. That needs no site parameters, no
    aerosol assumptions and no tilt geometry, and it absorbs whatever the
    fleet's real orientation happens to be.

    Fitted on the training split only. Fitting it on everything would let the
    test year set the level of the baseline it is later scored against, which
    is a quiet way of grading a model against itself.
    """
    bin_deg: float = 1.0
    quantile: float = 0.90
    _bins: np.ndarray = field(default=None, repr=False)
    _vals: np.ndarray = field(default=None, repr=False)

    def fit(self, elevation: np.ndarray, cf: np.ndarray) -> "ClearSky":
        lo = np.floor(elevation.min() / self.bin_deg) * self.bin_deg
        hi = np.ceil(elevation.max() / self.bin_deg) * self.bin_deg
        self._bins = np.arange(lo, hi + self.bin_deg, self.bin_deg)
        idx = np.clip(np.digitize(elevation, self._bins) - 1, 0, len(self._bins) - 2)

        vals = np.zeros(len(self._bins) - 1)
        for b in range(len(vals)):
            sel = cf[idx == b]
            vals[b] = np.quantile(sel, self.quantile) if sel.size >= 20 else np.nan
        # bins with too few hours borrow from their neighbours rather than
        # producing a hole that later divides by nan
        s = pd.Series(vals).interpolate(limit_direction="both")
        self._vals = np.maximum(s.to_numpy(), 0.0)
        return self

    def predict(self, elevation: np.ndarray) -> np.ndarray:
        idx = np.clip(np.digitize(elevation, self._bins) - 1, 0, len(self._vals) - 1)
        out = self._vals[idx]
        return np.where(elevation <= 0.0, 0.0, out)

    def index(self, elevation: np.ndarray, cf: np.ndarray) -> np.ndarray:
        """Clear-sky index: how much of a cloudless hour actually arrived.

        Undefined at night, where the denominator is zero. Filled with 1.0 -
        "as clear as it can be" - because the alternative, 0, would tell a
        model that every night is heavily overcast and let it carry that
        through sunrise.
        """
        cs = self.predict(elevation)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(cs > 1e-6, cf / np.maximum(cs, 1e-6), 1.0)
        return np.clip(np.nan_to_num(k, nan=1.0), 0.0, 1.5)


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------

def baseline_train_mean(train_cf: np.ndarray, n: int) -> np.ndarray:
    """B0. The constant that makes R2 exactly zero by definition, so the rest
    of the table has a floor to be read against."""
    return np.full(n, float(train_cf.mean()))


def baseline_climatology(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """B1. The average capacity factor for that day-of-year and hour.

    Contains no forecast information whatsoever: it is the same prediction for
    2019-06-15 at noon as it was for 2015-06-15 at noon, and it cannot react to
    a cloud. If this scores well, the metric is measuring the calendar.
    """
    key = ["doy", "hour"]
    tbl = train.groupby(key)["cf"].mean()
    # a 7-day rolling smooth over day-of-year, so a single overcast 14 June in
    # the training years does not become the climatology of every 14 June
    full = tbl.unstack("hour").sort_index()
    full = (pd.concat([full.iloc[-3:], full, full.iloc[:3]])
            .rolling(7, center=True, min_periods=1).mean().iloc[3:-3])
    lookup = full.stack()
    out = pd.MultiIndex.from_arrays([target["doy"], target["hour"]])
    return lookup.reindex(out).to_numpy(dtype=float)


def baseline_smart_persistence(cf_now: np.ndarray, cs_now: np.ndarray,
                               cs_target: np.ndarray) -> np.ndarray:
    """B2. Carry the clear-sky index forward and rescale it to the target hour.

    The standard operational benchmark. Plain persistence - repeating the last
    capacity factor - would predict darkness at noon after a night, so it is
    not a serious comparison and is not used.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(cs_now > 1e-6, cf_now / np.maximum(cs_now, 1e-6), 1.0)
    k = np.clip(np.nan_to_num(k, nan=1.0), 0.0, 1.5)
    return np.clip(k * cs_target, 0.0, 1.0)


# --------------------------------------------------------------------------
# learned models
# --------------------------------------------------------------------------

class RidgeModel:
    name = "ridge"

    def __init__(self, alpha: float = 1.0, **_):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def predict(self, X):
        return np.clip(self.model.predict(self.scaler.transform(X)), 0.0, 1.0)


class GBDTModel:
    name = "gbdt"

    def __init__(self, max_iter=300, max_depth=6, learning_rate=0.06,
                 early_stopping=False, seed=0, **_):
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter, max_depth=max_depth, learning_rate=learning_rate,
            early_stopping=early_stopping, random_state=seed)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X), 0.0, 1.0)


class ESNModel:
    """Echo state network: a fixed random recurrent reservoir, linear readout.

    The cheap end of the recurrent family. The reservoir weights are drawn once
    and never trained - only the readout is fitted, by ridge regression - so
    there is no backpropagation through time and no gradient to tune. The
    spectral radius is rescaled to keep the reservoir on the edge of stability,
    which is what gives it memory without letting its state blow up.
    """
    name = "esn"

    def __init__(self, n_reservoir=400, spectral_radius=0.9, leak_rate=0.3,
                 input_scale=0.5, ridge_alpha=1.0, washout=200, seed=0, **_):
        self.n = n_reservoir
        self.rho = spectral_radius
        self.leak = leak_rate
        self.input_scale = input_scale
        self.washout = washout
        self.scaler = StandardScaler()
        self.readout = Ridge(alpha=ridge_alpha)
        self.rng = np.random.default_rng(seed)
        self.W = None
        self.Win = None

    def _build(self, n_inputs: int) -> None:
        W = self.rng.normal(0.0, 1.0, (self.n, self.n))
        # sparse: each unit listens to about a tenth of the others, which is
        # the usual recipe and keeps the dynamics rich rather than uniform
        W *= (self.rng.random((self.n, self.n)) < 0.1)
        eig = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (self.rho / eig)
        self.Win = self.rng.uniform(-1.0, 1.0, (self.n, n_inputs + 1)) * self.input_scale

    def _run(self, X: np.ndarray) -> np.ndarray:
        states = np.zeros((len(X), self.n))
        s = np.zeros(self.n)
        for i, x in enumerate(X):
            u = np.r_[1.0, x]
            s = (1 - self.leak) * s + self.leak * np.tanh(self.Win @ u + self.W @ s)
            states[i] = s
        return states

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        if self.W is None:
            self._build(Xs.shape[1])
        states = self._run(Xs)
        # the reservoir starts from zero and needs time to forget it; the
        # first `washout` rows are transient, not signal
        w = min(self.washout, len(states) // 10)
        self.readout.fit(np.hstack([states[w:], Xs[w:]]), y[w:])
        return self

    def predict(self, X):
        Xs = self.scaler.transform(X)
        states = self._run(Xs)
        return np.clip(self.readout.predict(np.hstack([states, Xs])), 0.0, 1.0)


MODELS = {"ridge": RidgeModel, "gbdt": GBDTModel, "esn": ESNModel}
