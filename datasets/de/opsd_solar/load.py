"""
datasets/de/opsd_solar/load.py -- German solar generation, measured hourly.

    python load.py --inspect     # what the raw file contains
    python load.py               # fetch, build the tidy table, write the manifest
    python load.py --validate    # re-check the cached file against the manifest
    python load.py --check-sun   # verify the solar geometry against PVGIS

One row per hour: how much solar power Germany actually produced, how much
capacity was installed at the time, and where the sun was.

This is **measured** generation reported by the four German transmission
system operators, not a simulation. That matters for a forecasting experiment:
predicting a simulator is a different and much easier problem than predicting
weather, and a result obtained on simulated output would not transfer.

Solar geometry
--------------
The sun's elevation is computed here rather than downloaded, because it is the
one part of solar output that is known exactly in advance and is therefore the
benchmark any forecast has to beat. The implementation follows the NOAA solar
position algorithm and is checked against PVGIS's own `H_sun` by
`--check-sun`; agreement is within a fraction of a degree.

A national aggregate has no single location. Panels are spread from 47.3 to
55.1 degrees north, so elevation is computed at the country centroid and is an
approximation - stated here rather than left for the reader to discover.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
PROCESSED_DIR = HERE / "processed"
MANIFEST = HERE / "MANIFEST.sha256"

# Open Power System Data, time series package. Versioned by date in the URL,
# so the 2020-10-06 release is a fixed artefact rather than a moving target.
RELEASE = "2020-10-06"
URL = (f"https://data.open-power-system-data.org/time_series/{RELEASE}/"
       f"time_series_60min_singleindex.csv")

COLUMNS = ["utc_timestamp", "DE_solar_generation_actual", "DE_solar_capacity"]

# Germany's centroid. Panels run from 47.3N to 55.1N, so this is an
# approximation for a national aggregate - see the module docstring.
LAT, LON = 51.16, 10.45


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# solar geometry
# --------------------------------------------------------------------------

def solar_elevation(times: pd.DatetimeIndex, lat: float = LAT,
                    lon: float = LON) -> np.ndarray:
    """Solar elevation in degrees, NOAA algorithm.

    Everything a forecast is *not* allowed to take credit for. The sun's
    position at a given place and instant has been computable for centuries;
    a model that predicts solar output well at night has predicted nothing.
    """
    t = pd.DatetimeIndex(times)
    if t.tz is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")

    # fractional year, radians
    doy = t.dayofyear.to_numpy(dtype=float)
    hour = (t.hour.to_numpy(dtype=float) + t.minute.to_numpy(dtype=float) / 60.0)
    gamma = 2.0 * np.pi / 365.0 * (doy - 1.0 + (hour - 12.0) / 24.0)

    # equation of time (minutes) and declination (radians)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma)
                       - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma)
                       - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))

    # true solar time -> hour angle
    time_offset = eqtime + 4.0 * lon           # minutes; UTC, so no timezone term
    tst = hour * 60.0 + time_offset
    ha = np.deg2rad(tst / 4.0 - 180.0)

    latr = np.deg2rad(lat)
    cos_zenith = (np.sin(latr) * np.sin(decl)
                  + np.cos(latr) * np.cos(decl) * np.cos(ha))
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    return np.rad2deg(np.pi / 2.0 - np.arccos(cos_zenith))


def check_sun(lat: float = LAT, lon: float = LON, year: int = 2020) -> None:
    """Cross-check the geometry against PVGIS, which reports its own H_sun.

    Two independent implementations agreeing is worth more than one
    implementation looking plausible.
    """
    url = (f"https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat={lat}&lon={lon}"
           f"&startyear={year}&endyear={year}&outputformat=json&pvcalculation=0")
    req = urllib.request.Request(url, headers={"User-Agent": "ml-toy-experiments/0.1"})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.loads(r.read())

    rows = data["outputs"]["hourly"]
    # PVGIS timestamps are "YYYYMMDD:HHMM", centred on the half hour
    ts = pd.to_datetime([r_["time"] for r_ in rows], format="%Y%m%d:%H%M", utc=True)
    theirs = np.array([r_["H_sun"] for r_ in rows], dtype=float)
    ours = solar_elevation(ts, lat, lon)

    day = theirs > 0
    err = np.abs(ours[day] - theirs[day])
    print(f"solar elevation vs PVGIS, {len(rows):,} hours at ({lat}, {lon})")
    print(f"  daylight hours compared : {int(day.sum()):,}")
    print(f"  mean absolute error     : {err.mean():.3f} deg")
    print(f"  90th percentile         : {np.percentile(err, 90):.3f} deg")
    print(f"  worst                   : {err.max():.3f} deg")
    ok = err.mean() < 1.0
    print(f"  {'OK' if ok else 'FAIL'} - two independent implementations agree"
          if ok else "  FAIL - the geometry does not match PVGIS")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def download(refresh: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"opsd_time_series_60min_{RELEASE}.csv"
    if dest.exists() and not refresh:
        return dest
    print(f"  downloading OPSD {RELEASE} (about 130 MB) ...", flush=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "ml-toy-experiments/0.1"})
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    print(f"    {dest.name}  {dest.stat().st_size / 1e6:.1f} MB")
    return dest


def load(refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Hourly German solar output as a capacity factor, plus solar geometry.

    Generation is divided by installed capacity because capacity grew 36% over
    the period (37.2 to 50.5 GW). A model fitted to raw megawatts would learn
    that trend and be rewarded for it, which is not forecasting.
    """
    path = download(refresh)
    raw = pd.read_csv(path, usecols=COLUMNS, parse_dates=["utc_timestamp"])
    n0 = len(raw)

    df = raw.dropna(subset=["DE_solar_generation_actual", "DE_solar_capacity"]).copy()
    df = df[df["DE_solar_capacity"] > 0]
    df = df.rename(columns={"utc_timestamp": "time",
                            "DE_solar_generation_actual": "generation_mw",
                            "DE_solar_capacity": "capacity_mw"})
    df["cf"] = df["generation_mw"] / df["capacity_mw"]
    df["elevation"] = solar_elevation(pd.DatetimeIndex(df["time"]))
    df["is_day"] = df["elevation"] > 5.0
    df = df.sort_values("time").reset_index(drop=True)

    # a capacity factor above 1 would mean generating more than the installed
    # peak; a handful of hours do, from capacity being reported with a lag
    over = int((df["cf"] > 1.0).sum())
    df["cf"] = df["cf"].clip(0.0, 1.0)

    if verbose:
        print(f"  {n0:,} rows -> {len(df):,} with both generation and capacity")
        print(f"  {df['time'].min()} -> {df['time'].max()}")
        print(f"  capacity {df.capacity_mw.min():,.0f} -> {df.capacity_mw.max():,.0f} MW"
              f"  ({df.capacity_mw.max()/df.capacity_mw.min()-1:+.0%})")
        print(f"  capacity factor: mean {df.cf.mean():.3f}, max {df.cf.max():.3f}")
        night = (~df["is_day"]).mean()
        zero = (df["cf"] == 0).mean()
        print(f"  {night:.1%} of hours are night (elevation <= 5 deg); "
              f"{zero:.1%} have exactly zero output")
        if over:
            print(f"  note: {over} hour(s) had cf > 1 and were clipped "
                  f"(capacity is reported with a lag)")
    return df


def write_manifest() -> None:
    path = RAW_DIR / f"opsd_time_series_60min_{RELEASE}.csv"
    if not path.exists():
        raise SystemExit("nothing cached yet - run `python load.py` first.")
    df = load(verbose=False)
    MANIFEST.write_text(json.dumps({
        "dataset": "opsd_de_solar",
        "source": "Open Power System Data, time series package",
        "release": RELEASE,
        "url": URL,
        "license": "see README - OPSD publishes the package under an open licence, "
                   "with the underlying TSO data attributed to its sources",
        "retrieval_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "hours_after_cleaning": int(len(df)),
        "first_hour": str(df["time"].min()),
        "last_hour": str(df["time"].max()),
        "centroid": {"lat": LAT, "lon": LON},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {MANIFEST.name}")


def validate() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST.name} not found - run `python load.py` first.")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    path = RAW_DIR / man["file"]
    if not path.exists():
        raise SystemExit(f"{man['file']} is not cached - run `python load.py`.")
    actual = _sha256(path)
    ok = actual == man["sha256"]
    print(f"  {'OK      ' if ok else 'MISMATCH'} {man['file']}  "
          f"{man['hours_after_cleaning']:,} hours")
    if not ok:
        raise SystemExit(
            "\nThe cached file differs from the manifest. OPSD versions its "
            "releases by date, so this should not happen unless the file was "
            "edited or a different release was downloaded.")


def inspect() -> None:
    path = download()
    head = pd.read_csv(path, nrows=5)
    solar = [c for c in head.columns if "solar" in c.lower()]
    print(f"\n{len(head.columns)} columns in the release, {len(solar)} mentioning solar:")
    for c in solar:
        print(f"  {c}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--check-sun", action="store_true")
    a = ap.parse_args()

    if a.inspect:
        inspect()
    elif a.validate:
        validate()
    elif a.check_sun:
        check_sun()
    else:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        load(refresh=a.refresh)
        write_manifest()


if __name__ == "__main__":
    main()
