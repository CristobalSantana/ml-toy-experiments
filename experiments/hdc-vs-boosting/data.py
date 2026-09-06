"""
data.py -- The task: was this taxi ride paid in cash?

Built from the NYC taxi months already cached by datasets/us/nyc_taxi. Train
on May 2024, test on June 2024 - a forward split, so nothing from the test
month can inform the fit.

The feature that had to be thrown away
--------------------------------------
`tip_amount` is exactly zero for **100%** of cash trips and non-zero for 94%
of card trips, because the TLC records card tips and has no way to see cash
ones. The label is written into the feature by the collection process, not by
any temporal mistake. Including it turns a hard problem into a lookup, and
`test_leakage.py` demonstrates exactly how much.

`total_amount` carries the same contamination second-hand, since it is fare
plus extras plus tip. Both are excluded, and the exclusion is the reason the
task is hard enough to be worth running.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1] / "datasets" / "us" / "nyc_taxi" / "raw"

FEATURES = ["trip_distance", "fare_amount", "passenger_count",
            "trip_duration_min", "pickup_hour", "pu_location", "do_location"]

# Kept out on purpose. See the module docstring; test_leakage.py measures what
# including tip_amount would have bought.
LEAKY = ["tip_amount", "total_amount"]

RAW_COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance",
               "fare_amount", "tip_amount", "total_amount", "passenger_count",
               "payment_type", "PULocationID", "DOLocationID"]


def load_month(month: str, with_leaky: bool = False) -> pd.DataFrame:
    """One month as a binary task. `is_cash` is the label.

    Payment type 1 is card and 2 is cash; the other codes are disputes, no
    charge and voided trips, which are not a payment method and are dropped
    rather than folded into one of the two classes.
    """
    path = RAW / f"yellow_tripdata_{month}.parquet"
    if not path.exists():
        raise SystemExit(
            f"{path.name} is not cached. Run:\n"
            f"  python ../../datasets/us/nyc_taxi/load.py --months {month}")

    d = pd.read_parquet(path, columns=RAW_COLUMNS)
    d.columns = [c.lower() for c in d.columns]
    d = d[d["payment_type"].isin([1, 2])].copy()

    d["trip_duration_min"] = ((d["tpep_dropoff_datetime"] - d["tpep_pickup_datetime"])
                              .dt.total_seconds() / 60.0)
    d["pickup_hour"] = d["tpep_pickup_datetime"].dt.hour.astype(float)
    d["pu_location"] = d["pulocationid"].astype(float)
    d["do_location"] = d["dolocationid"].astype(float)
    d["is_cash"] = (d["payment_type"] == 2).astype(int)

    # the same cleaning rules the drift experiment froze, so the two
    # experiments are talking about the same population of trips
    keep = (d["trip_distance"].between(0, 50, inclusive="right")
            & d["fare_amount"].between(0, 300, inclusive="right")
            & d["trip_duration_min"].between(0, 180, inclusive="right")
            & d["passenger_count"].between(1, 6, inclusive="both")
            & (d["tpep_pickup_datetime"].dt.to_period("M").astype(str) == month))
    d = d[keep]

    cols = FEATURES + (LEAKY if with_leaky else []) + ["is_cash"]
    return d[cols].dropna().reset_index(drop=True)


def build(train_month: str, test_month: str, test_size: int, seed: int,
          with_leaky: bool = False, verbose: bool = True):
    """Training pool and a fixed test sample, drawn once."""
    tr = load_month(train_month, with_leaky)
    te = load_month(test_month, with_leaky)

    rng = np.random.default_rng(seed)
    if len(te) > test_size:
        te = te.iloc[rng.choice(len(te), test_size, replace=False)]

    if verbose:
        print(f"  train pool {train_month}: {len(tr):,} trips, "
              f"{tr.is_cash.mean():.1%} cash")
        print(f"  test       {test_month}: {len(te):,} trips, "
              f"{te.is_cash.mean():.1%} cash")
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def subsample(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """A training set of exactly n rows, drawn without stratifying.

    Not stratified on purpose: a real deployment does not get to guarantee
    both classes appear in a small sample, and how a method copes when the
    minority class is thin is part of what the learning curve is measuring.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), min(n, len(pool)), replace=False)
    return pool.iloc[idx].reset_index(drop=True)
