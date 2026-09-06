"""
data.py -- The forecasting task, kept deliberately small.

Next-hour German solar capacity factor. The interesting question about solar
forecasting - how much of a reported R^2 is the sun rising on schedule - is
answered in `solar-forecast-skill`. Here the forecast is only a task on which
a spiking network and a dense one can be matched in accuracy, so that their
operation counts can be compared at the same level of usefulness.

Everything here is observable at hour `t`, except the position of the sun at
`t+1`, which is known centuries in advance. The clear-sky envelope is fitted
on the training years only: fitting it on all the data would let the test
year into the baseline it is scored against.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent.parent / "datasets" / "de" / "opsd_solar"

LAGS = [1, 2, 3, 24]          # hours back, 1 meaning "now"
DAYLIGHT_DEG = 5.0


def _loader():
    sys.path.insert(0, str(DATASET))
    import load                                     # noqa: E402
    return load


class ClearSky:
    """The 90th percentile of capacity factor in each 1-degree elevation bin.

    An empirical envelope rather than a physical model: it absorbs panel
    orientation, soiling and the fleet's geographic spread, none of which a
    clear-sky irradiance formula knows about.
    """

    def __init__(self, bin_deg: float = 1.0, quantile: float = 0.90):
        self.bin_deg, self.quantile = bin_deg, quantile
        self.centres: np.ndarray | None = None
        self.values: np.ndarray | None = None

    def fit(self, elevation: np.ndarray, cf: np.ndarray) -> "ClearSky":
        e = np.asarray(elevation)
        bins = np.floor(e / self.bin_deg).astype(int)
        d = pd.DataFrame({"b": bins, "cf": np.asarray(cf)})
        g = d.groupby("b")["cf"].quantile(self.quantile)
        self.centres = (g.index.to_numpy() + 0.5) * self.bin_deg
        # monotone in elevation by construction of the physics; enforcing it
        # stops a sparse high-elevation bin from dipping below its neighbour
        self.values = np.maximum.accumulate(g.to_numpy())
        return self

    def __call__(self, elevation: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(elevation), self.centres, self.values)


@dataclass
class Task:
    Xtr: np.ndarray
    ytr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    features: list[str]
    test_frame: pd.DataFrame          # for the baselines and the day mask
    day_te: np.ndarray

    @property
    def n_features(self) -> int:
        return self.Xtr.shape[1]


def build(splits: dict, verbose: bool = True) -> tuple[pd.DataFrame, ClearSky]:
    df = _loader().load(verbose=False)
    d = df.sort_values("time").reset_index(drop=True).copy()

    train_mask = d["time"] <= splits["train_end"]
    cs_model = ClearSky().fit(d.loc[train_mask, "elevation"].to_numpy(),
                              d.loc[train_mask, "cf"].to_numpy())
    d["cs"] = cs_model(d["elevation"].to_numpy())
    d["k"] = d["cf"] / np.maximum(d["cs"], 1e-3)

    out = pd.DataFrame({"time": d["time"]})
    out["y"] = d["cf"].shift(-1)
    out["target_elevation"] = d["elevation"].shift(-1)
    out["target_cs"] = d["cs"].shift(-1)
    for lag in LAGS:
        out[f"cf_lag{lag}"] = d["cf"].shift(lag - 1)
        out[f"k_lag{lag}"] = d["k"].shift(lag - 1)
    out["elevation_now"] = d["elevation"]
    out["cs_now"] = d["cs"]

    tt = pd.DatetimeIndex(d["time"].shift(-1))
    out["target_hour_sin"] = np.sin(2 * np.pi * tt.hour / 24)
    out["target_hour_cos"] = np.cos(2 * np.pi * tt.hour / 24)
    out["target_doy_sin"] = np.sin(2 * np.pi * tt.dayofyear / 365.25)
    out["target_doy_cos"] = np.cos(2 * np.pi * tt.dayofyear / 365.25)

    out = out.dropna().reset_index(drop=True)
    if verbose:
        print(f"  {len(out):,} forecast rows, "
              f"{out['time'].min().date()} to {out['time'].max().date()}")
    return out, cs_model


def make_task(splits: dict, scaling: str = "standard",
              verbose: bool = True) -> Task:
    """Split by calendar year, never shuffled, and scale on the train split.

    `scaling` is 'standard' for the dense network and the direct-coded SNN,
    and 'minmax' for the rate-coded one, whose encoder needs values in [0, 1]
    to read as Bernoulli probabilities. Both are fitted on the training years
    alone.
    """
    frame, _ = build(splits, verbose=verbose)
    feats = [c for c in frame.columns
             if c not in ("time", "y", "target_elevation", "target_cs")]

    t = frame["time"]
    tr = t <= splits["train_end"]
    va = (t > splits["train_end"]) & (t <= splits["val_end"])
    te = t > splits["val_end"]

    X = frame[feats].to_numpy(dtype=np.float32)
    y = frame["y"].to_numpy(dtype=np.float32)

    if scaling == "standard":
        mu, sd = X[tr].mean(0), X[tr].std(0)
        Xs = (X - mu) / np.maximum(sd, 1e-6)
    elif scaling == "minmax":
        lo, hi = X[tr].min(0), X[tr].max(0)
        Xs = (X - lo) / np.maximum(hi - lo, 1e-6)
    else:
        raise ValueError(f"unknown scaling {scaling!r}")

    day_te = (frame.loc[te, "target_elevation"].to_numpy() > DAYLIGHT_DEG)
    if verbose:
        print(f"  train {int(tr.sum()):,}  val {int(va.sum()):,}  "
              f"test {int(te.sum()):,}  ({day_te.sum():,} daylight hours)")

    return Task(Xtr=Xs[tr], ytr=y[tr], Xva=Xs[va], yva=y[va],
                Xte=Xs[te], yte=y[te], features=feats,
                test_frame=frame.loc[te].reset_index(drop=True),
                day_te=day_te)


def baselines(task: Task) -> dict[str, np.ndarray]:
    """Persistence and smart persistence, on the test split.

    Smart persistence carries the *clear-sky index* forward rather than the
    output, so it knows the sun is about to set. It is the baseline a solar
    forecast has to beat to have done anything.
    """
    f = task.test_frame
    return {
        "persistence": f["cf_lag1"].to_numpy(),
        "smart_persistence": (f["k_lag1"].to_numpy()
                              * f["target_cs"].to_numpy()).clip(0.0, 1.0),
    }
