"""
features.py -- Build the design matrix, and refuse to build a leaky one.

Every column here is either an observation from hour *t* or earlier, or a
property of the sun at hour *t+h*. The sun's position is admissible because it
is known centuries in advance; a cloud is not.

The one mistake this file exists to prevent
-------------------------------------------
Shifting a lag the wrong way. `df["cf"].shift(-1)` reads the future, and it
looks exactly like `df["cf"].shift(1)`, which does not. A model trained on the
first will report a near-perfect forecast and be useless. `test_leakage.py`
builds that version deliberately and asserts the check catches it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.DatetimeIndex(df["time"])
    df = df.copy()
    df["hour"] = t.hour
    df["doy"] = t.dayofyear
    df["year"] = t.year
    # day of year as a circle: 31 December is next to 1 January, and an
    # integer feature says they are 364 apart
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    return df


def build(df: pd.DataFrame, horizon: int, cf_lags: list[int],
          use_clear_sky_index: bool = True,
          use_target_geometry: bool = True) -> pd.DataFrame:
    """One row per forecast: features known at t, target at t + horizon.

    The frame is shifted so that row i holds the target for hour t+h and the
    features observable at hour t. Rows where any lag or the target falls
    outside the series are dropped rather than filled - imputing them would
    invent observations the forecaster never had.
    """
    d = df.sort_values("time").reset_index(drop=True).copy()

    out = pd.DataFrame({"time": d["time"], "target_time": d["time"].shift(-horizon)})
    out["y"] = d["cf"].shift(-horizon)                     # the future, as the label
    out["target_elevation"] = d["elevation"].shift(-horizon)
    out["target_cs"] = d["cs"].shift(-horizon)
    out["target_is_day"] = d["is_day"].shift(-horizon)

    # everything below is observable at t: positive shifts only
    for lag in cf_lags:
        out[f"cf_lag{lag}"] = d["cf"].shift(lag - 1)
        if use_clear_sky_index:
            out[f"k_lag{lag}"] = d["k"].shift(lag - 1)
    out["cf_now"] = d["cf"]
    out["k_now"] = d["k"]
    out["cs_now"] = d["cs"]
    out["elevation_now"] = d["elevation"]

    if use_target_geometry:
        # hour and season of the hour being predicted, not of now
        tt = pd.DatetimeIndex(out["target_time"])
        out["target_hour"] = tt.hour
        out["target_doy_sin"] = np.sin(2 * np.pi * tt.dayofyear / 365.25)
        out["target_doy_cos"] = np.cos(2 * np.pi * tt.dayofyear / 365.25)

    return out.dropna().reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Everything except the target, the timestamps and the evaluation masks."""
    drop = {"time", "target_time", "y", "target_is_day"}
    return [c for c in frame.columns if c not in drop]


def max_observed_time(frame: pd.DataFrame, df: pd.DataFrame, horizon: int) -> pd.Series:
    """The latest timestamp each row's features could legitimately come from.

    Used by the leakage check: for row i the answer is the issue time, and no
    feature may encode anything after it.
    """
    return frame["time"]
