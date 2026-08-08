"""
load.py -- Parse the Banco Central IPV Excel export (manual download from the
BDE) into one tidy quarterly table for Experiment 02 (temporal / drift arm).

    processed/ipv_quarterly.csv   one row per (quarter, series) with the index level
    MANIFEST.sha256               checksums + provenance (committed)

The BDE Excel layout varies with how the cuadro is exported, so:
  * `--inspect` prints every sheet's shape and head, to confirm the structure;
  * `build()` uses a defensive wide->long heuristic (find the quarter axis, melt
    the numeric series) and validates loudly. Finalize the reshape against the
    real file once it is in raw/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
PROCESSED_DIR = HERE / "processed"
MANIFEST = HERE / "MANIFEST.sha256"

# "2014.I", "2014 T1", "2014-Q1", "I 2014", "2014.1" ... -> (year, quarter)
_QUARTER_RE = re.compile(r"(?:(\d{4}).*?(?:T|Q|\.|\s)\s*([1-4IV]+))|(?:([1-4IV]+)\s*.*?(\d{4}))")
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "1": 1, "2": 2, "3": 3, "4": 4}


def _raw_files() -> list[Path]:
    return sorted(p for p in RAW_DIR.glob("*.xls*")) if RAW_DIR.is_dir() else []


def _read_all_sheets(path: Path) -> dict[str, pd.DataFrame]:
    try:
        return pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    except ImportError:
        sys.exit("ABORT: reading .xlsx needs openpyxl (`pip install openpyxl`). "
                 "For legacy .xls, `pip install xlrd`.")


def inspect() -> None:
    files = _raw_files()
    if not files:
        sys.exit(f"No Excel files in {RAW_DIR} - download the IPV cuadro first (see README).")
    for path in files:
        print(f"\n=== {path.name} ===")
        for sheet, df in _read_all_sheets(path).items():
            print(f"  sheet '{sheet}': {df.shape[0]} rows x {df.shape[1]} cols")
            print(df.head(8).to_string(max_cols=12))


def _parse_quarter(text: str):
    m = _QUARTER_RE.search(str(text).strip().upper())
    if not m:
        return None
    year, q = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
    q = _ROMAN.get(str(q).strip())
    if not q or not year:
        return None
    return pd.Timestamp(int(year), (q - 1) * 3 + 1, 1)  # first month of the quarter


def build() -> None:
    files = _raw_files()
    if not files:
        sys.exit(f"No Excel files in {RAW_DIR} - download the IPV cuadro first (see README).")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    tidy_frames = []
    for path in files:
        for sheet, df in _read_all_sheets(path).items():
            # Locate the quarter axis: the row or column with the most parseable quarters.
            col_scores = [(c, df[c].map(lambda v: _parse_quarter(v) is not None).sum()) for c in df.columns]
            best_col, best_col_n = max(col_scores, key=lambda t: t[1])
            row_scores = [(i, df.loc[i].map(lambda v: _parse_quarter(v) is not None).sum()) for i in df.index]
            best_row, best_row_n = max(row_scores, key=lambda t: t[1])

            if max(best_col_n, best_row_n) < 8:
                continue  # not the data sheet

            if best_col_n >= best_row_n:
                # quarters run down a column; series are the other columns
                quarters = df[best_col].map(_parse_quarter)
                keep = quarters.notna()
                long = df.loc[keep].drop(columns=[best_col]).apply(pd.to_numeric, errors="coerce")
                long.insert(0, "quarter", quarters[keep].values)
                melted = long.melt(id_vars="quarter", var_name="series_col", value_name="index_value")
            else:
                # quarters run across a row; series are the other rows
                quarters = df.loc[best_row].map(_parse_quarter)
                qcols = [c for c in df.columns if pd.notna(quarters[c])]
                block = df[qcols].apply(pd.to_numeric, errors="coerce")
                block.columns = [quarters[c] for c in qcols]
                melted = block.reset_index().melt(id_vars="index", var_name="quarter", value_name="index_value")
                melted = melted.rename(columns={"index": "series_col"})

            melted = melted.dropna(subset=["index_value"])
            melted["source_sheet"] = sheet
            tidy_frames.append(melted)

    if not tidy_frames:
        sys.exit("ABORT: no sheet had a recognizable quarterly axis. Run --inspect and "
                 "adapt the reshape to the actual BDE layout.")

    tidy = pd.concat(tidy_frames, ignore_index=True)
    tidy = tidy[["quarter", "series_col", "index_value", "source_sheet"]].sort_values(["series_col", "quarter"])

    # --- loud validation ---
    span = (tidy["quarter"].min(), tidy["quarter"].max())
    n_series = tidy["series_col"].nunique()
    print(f"IPV: {len(tidy)} obs, {n_series} series, {span[0].date()} .. {span[1].date()}")
    if span[0] > pd.Timestamp(2015, 1, 1):
        print(f"WARNING: series starts {span[0].date()} (brief expects coverage from 2014).")

    out = PROCESSED_DIR / "ipv_quarterly.csv"
    tidy.to_csv(out, index=False, encoding="utf-8")
    _write_manifest(files, tidy)
    print(f"Wrote {out} and {MANIFEST.name}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(files: list[Path], tidy: pd.DataFrame) -> None:
    manifest = {
        "dataset": "bcch_ipv",
        "retrieval_date": date.today().isoformat(),
        "source": "Banco Central de Chile - Base de Datos Estadísticos (BDE), IPV cuadro (manual export)",
        "publication_lag_days": 120,
        "n_obs": int(len(tidy)),
        "n_series": int(tidy["series_col"].nunique()),
        "quarter_min": str(tidy["quarter"].min().date()),
        "quarter_max": str(tidy["quarter"].max().date()),
        "raw_files": {p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in files},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def validate() -> None:
    if not MANIFEST.exists():
        sys.exit("No MANIFEST.sha256 - run `python load.py` first.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mismatch = [name for name, meta in manifest["raw_files"].items()
                if not (RAW_DIR / name).exists() or _sha256(RAW_DIR / name) != meta["sha256"]]
    if mismatch:
        sys.exit(f"CHECKSUM MISMATCH on: {mismatch}")
    print(f"OK: {len(manifest['raw_files'])} raw file(s) match the committed manifest.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--inspect", action="store_true", help="print sheet shapes/heads and exit")
    ap.add_argument("--validate", action="store_true", help="check raw files against the manifest and exit")
    args = ap.parse_args()
    inspect() if args.inspect else (validate() if args.validate else build())
