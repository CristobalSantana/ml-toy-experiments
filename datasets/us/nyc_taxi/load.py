"""
datasets/us/nyc_taxi/load.py -- NYC yellow taxi trip records.

    python load.py --inspect            # what a month contains, uncleaned
    python load.py                      # fetch + clean the pre-registered months
    python load.py --months 2024-06     # just one
    python load.py --validate           # re-check cached files against the manifest

One row per taxi trip, published monthly by the NYC Taxi & Limousine
Commission. Public, unauthenticated, no key.

The cleaning rules are not this loader's to choose: they are frozen in
experiments/drift-detector-overlap/CRITERIA.md, and are duplicated here as
constants so that a change in either place shows up as a diff rather than as a
silently different dataset.

Two things this loader exists to prevent
----------------------------------------
**Schema drift.** The airport fee column is `airport_fee` in the 2019 files
and `Airport_fee` in the 2024 ones - the same field, differing only in
capitalisation. Concatenating the two without normalising produces two
columns, each about half null, and nothing raises. Column names are lowercased
on read.

**Stray rows.** The files are named for a month but are not confined to it.
The 2019-06 file contains trips dated 2001; the 2024-06 file contains one
dated 2026-06-26, which is in the future relative to the file itself. Rows
outside the nominal month are dropped, and the count is reported rather than
hidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
PROCESSED_DIR = HERE / "processed"
MANIFEST = HERE / "MANIFEST.sha256"

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"

# The six months the experiment is pre-registered on. The loader accepts any
# month; these are the ones whose checksums are committed.
MONTHS = ["2019-06", "2020-03", "2020-04", "2023-06", "2024-05", "2024-06"]

# The seven features, chosen because they exist in every month in the range.
# congestion_surcharge and airport_fee do not, which is itself recorded in
# CRITERIA.md rather than patched over.
FEATURES = ["trip_distance", "fare_amount", "tip_amount", "total_amount",
            "trip_duration_min", "passenger_count", "pickup_hour"]

# Frozen in CRITERIA.md on 2026-09-05, from counts measured on 2024-06.
#
# The third element is which ends are included, and it is not the same for
# every rule: CRITERIA writes four of them as `0 < x <= hi` but passenger
# count as `1 <= n <= 6`. Applying the half-open convention to all five drops
# every single-passenger trip - 78% of the file - and the run still completes,
# reporting a clean-looking dataset. The pre-registered row counts are what
# caught it.
LIMITS = {
    "trip_distance": (0.0, 50.0, "right"),      # miles; 52,954 zero-distance trips in 2024-06
    "fare_amount": (0.0, 300.0, "right"),       # USD;   63,043 non-positive fares
    "trip_duration_min": (0.0, 180.0, "right"),  # min;  negative durations exist
    "passenger_count": (1.0, 6.0, "both"),      # 35,839 zeros and 410,781 nulls
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(month: str, refresh: bool = False) -> Path:
    """Fetch one month into raw/. Cached: reruns never touch the network."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"yellow_tripdata_{month}.parquet"
    if dest.exists() and not refresh:
        return dest

    url = BASE_URL.format(month=month)
    print(f"  downloading {month} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ml-toy-experiments/0.1"})
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    print(f"    {dest.name}  {dest.stat().st_size / 1e6:.1f} MB")
    return dest


def load_month(month: str, clean: bool = True, refresh: bool = False,
               verbose: bool = True) -> pd.DataFrame:
    """One month, with the frozen cleaning rules applied.

    Returns the seven pre-registered features. Every rule reports what it
    removed: a cleaning step that quietly drops a third of the data is the
    kind of thing that turns up later as a "finding".
    """
    path = download(month, refresh=refresh)
    raw = pd.read_parquet(path)
    # the schema-drift fix, applied before anything reads a column by name
    raw.columns = [c.lower() for c in raw.columns]

    n0 = len(raw)
    df = pd.DataFrame({
        "pickup": raw["tpep_pickup_datetime"],
        "dropoff": raw["tpep_dropoff_datetime"],
        "trip_distance": raw["trip_distance"].astype("float64"),
        "fare_amount": raw["fare_amount"].astype("float64"),
        "tip_amount": raw["tip_amount"].astype("float64"),
        "total_amount": raw["total_amount"].astype("float64"),
        "passenger_count": raw["passenger_count"].astype("float64"),
    })
    df["trip_duration_min"] = (df["dropoff"] - df["pickup"]).dt.total_seconds() / 60.0
    df["pickup_hour"] = df["pickup"].dt.hour.astype("float64")

    if not clean:
        return df

    steps: list[tuple[str, int]] = []

    in_month = df["pickup"].dt.to_period("M").astype(str) == month
    steps.append(("outside the nominal month", int((~in_month).sum())))
    df = df[in_month]

    for col, (lo, hi, how) in LIMITS.items():
        keep = df[col].between(lo, hi, inclusive=how)
        span = f"[{lo:g}, {hi:g}]" if how == "both" else f"({lo:g}, {hi:g}]"
        steps.append((f"{col} outside {span}", int((~keep).sum())))
        df = df[keep]

    df = df[FEATURES].reset_index(drop=True)

    if verbose:
        kept = len(df)
        print(f"  {month}: {n0:,} rows -> {kept:,} kept ({kept / n0 * 100:.1f}%)")
        for label, n in steps:
            if n:
                print(f"      -{n:>9,}  {label}")
    return df


def write_manifest(months: list[str] | None = None) -> None:
    """Checksums, not bytes. The parquet files are hundreds of megabytes and
    are gitignored; this is what makes the provenance versioned.

    Always covers every month cached in raw/, not just the ones this
    invocation touched. Keying off the argument meant that `--months 2024-06`
    rewrote the manifest down to a single entry and silently dropped the
    provenance of the other five.
    """
    cached = sorted(p.stem.replace("yellow_tripdata_", "")
                    for p in RAW_DIR.glob("yellow_tripdata_*.parquet"))
    files = {}
    for m in cached:
        p = RAW_DIR / f"yellow_tripdata_{m}.parquet"
        if not p.exists():
            continue
        md = pd.read_parquet(p, columns=["trip_distance"])
        files[m] = {"file": p.name, "bytes": p.stat().st_size,
                    "rows": len(md), "sha256": _sha256(p)}

    MANIFEST.write_text(json.dumps({
        "dataset": "nyc_taxi_yellow",
        "source": "NYC Taxi & Limousine Commission, trip record data",
        "url_pattern": BASE_URL,
        "license": "public domain (NYC Open Data / TLC), see README",
        "retrieval_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "months": files,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {MANIFEST.name} covering {len(files)} month(s)")


def validate() -> None:
    """Re-check the cached files against the committed checksums."""
    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST.name} not found - run `python load.py` first.")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = []
    for m, rec in man["months"].items():
        p = RAW_DIR / rec["file"]
        if not p.exists():
            print(f"  MISSING  {m}")
            continue
        actual = _sha256(p)
        ok = actual == rec["sha256"]
        print(f"  {'OK      ' if ok else 'MISMATCH'} {m}  {rec['rows']:>9,} rows")
        if not ok:
            bad.append(m)
    if bad:
        raise SystemExit(
            f"\n{len(bad)} file(s) differ from the manifest: {', '.join(bad)}.\n"
            f"The published results were produced from the recorded files. TLC "
            f"reissues months on correction, so a mismatch is information, not "
            f"necessarily a fault - but it must be acknowledged, not ignored.")


def inspect(month: str = "2024-06") -> None:
    """What a month looks like before any rule is applied."""
    path = download(month)
    raw = pd.read_parquet(path)
    raw.columns = [c.lower() for c in raw.columns]
    print(f"\n{month}: {len(raw):,} rows, {len(raw.columns)} columns")
    for c in raw.columns:
        print(f"  {c:<24} {str(raw[c].dtype):<16} nulls {raw[c].isna().sum():>9,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--months", nargs="*", default=MONTHS)
    ap.add_argument("--refresh", action="store_true", help="re-download, ignoring the cache")
    ap.add_argument("--inspect", action="store_true", help="dump one month's raw schema")
    ap.add_argument("--validate", action="store_true", help="check cached files against the manifest")
    a = ap.parse_args()

    if a.inspect:
        inspect(a.months[0])
        return
    if a.validate:
        validate()
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {len(a.months)} month(s) into {RAW_DIR}\n")
    for m in a.months:
        load_month(m, refresh=a.refresh)
    write_manifest(a.months)


if __name__ == "__main__":
    main()
