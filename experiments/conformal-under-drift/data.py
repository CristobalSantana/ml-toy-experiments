"""
data.py -- Taxi months as a regression task: predict the metered fare.

Reuses the parquet files cached by datasets/us/nyc_taxi and the cleaning rules
frozen in drift-detector-overlap/CRITERIA.md, so all three experiments that
touch this data describe the same population of trips.

Excluded on purpose
-------------------
`total_amount` contains `fare_amount`, so predicting the fare from it is
arithmetic rather than modelling. `tip_amount` is contaminated by payment
method - the TLC records card tips and cannot see cash ones - which is
measured in hdc-vs-boosting/test_leakage.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1] / "datasets" / "us" / "nyc_taxi" / "raw"

FEATURES = ["trip_distance", "trip_duration_min", "passenger_count",
            "pickup_hour", "pu_location", "do_location"]
TARGET = "fare_amount"

RAW_COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance",
               "fare_amount", "passenger_count", "PULocationID", "DOLocationID"]


def load_month(month: str, verbose: bool = False) -> pd.DataFrame:
    path = RAW / f"yellow_tripdata_{month}.parquet"
    if not path.exists():
        raise SystemExit(
            f"{path.name} is not cached. Run:\n"
            f"  python ../../datasets/us/nyc_taxi/load.py --months {month}")

    d = pd.read_parquet(path, columns=RAW_COLUMNS)
    d.columns = [c.lower() for c in d.columns]
    d["trip_duration_min"] = ((d["tpep_dropoff_datetime"] - d["tpep_pickup_datetime"])
                              .dt.total_seconds() / 60.0)
    d["pickup_hour"] = d["tpep_pickup_datetime"].dt.hour.astype(float)
    d["pu_location"] = d["pulocationid"].astype(float)
    d["do_location"] = d["dolocationid"].astype(float)

    keep = (d["trip_distance"].between(0, 50, inclusive="right")
            & d["fare_amount"].between(0, 300, inclusive="right")
            & d["trip_duration_min"].between(0, 180, inclusive="right")
            & d["passenger_count"].between(1, 6, inclusive="both")
            & (d["tpep_pickup_datetime"].dt.to_period("M").astype(str) == month))
    out = d.loc[keep, FEATURES + [TARGET]].dropna().reset_index(drop=True)
    if verbose:
        print(f"  {month}: {len(out):,} trips, mean fare "
              f"${out[TARGET].mean():.2f}")
    return out


def three_way_split(df: pd.DataFrame, n_fit: int, n_cal: int, n_eval: int,
                    seed: int):
    """Disjoint fit / calibrate / eval samples, drawn once.

    Disjoint matters more than usual here. Calibrating on rows the model was
    fitted on gives residuals that are too small, the interval comes out too
    narrow, and the coverage guarantee is quietly void before any drift is
    involved.
    """
    need = n_fit + n_cal + n_eval
    if len(df) < need:
        raise SystemExit(f"month has {len(df):,} trips, need {need:,}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    a, b = n_fit, n_fit + n_cal
    return (df.iloc[idx[:a]].reset_index(drop=True),
            df.iloc[idx[a:b]].reset_index(drop=True),
            df.iloc[idx[b:need]].reset_index(drop=True))


def sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    return df.iloc[idx].reset_index(drop=True)
