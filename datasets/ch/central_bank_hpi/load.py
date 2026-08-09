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


def _parse_quarter(text):
    """Return the Timestamp of the quarter's first month, or None.

    Handles both BDE export flavours: real datetime cells (the IPV cuadro
    exports '2002-03-01 00:00:00' per quarter) and text labels ('2002.I',
    '2014 T1', 'IV 2015').
    """
    if isinstance(text, (pd.Timestamp, datetime)) or hasattr(text, "year") and hasattr(text, "month"):
        try:
            ts = pd.Timestamp(text)
        except (ValueError, TypeError):
            return None
        return pd.Timestamp(ts.year, ((ts.month - 1) // 3) * 3 + 1, 1)
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    m = _QUARTER_RE.search(str(text).strip().upper())
    if not m:
        return None
    year, q = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
    q = _ROMAN.get(str(q).strip())
    if not q or not year:
        return None
    return pd.Timestamp(int(year), (q - 1) * 3 + 1, 1)  # first month of the quarter


def _assert_is_price_index(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    """Guard against downloading the wrong BDE cuadro.

    The BDE's Vivienda section also carries "Indicadores de ventas efectivas de
    viviendas" (series F034.CVV.* - a *transaction-volume* index) which is easy
    to grab by mistake: it is quarterly, dwelling-type split, and looks like the
    IPV at a glance. It measures how many homes sold, not at what price, so it
    cannot serve as this experiment's temporal target. Fail loudly rather than
    silently modelling the wrong quantity.
    """
    blob = " ".join(
        str(v) for df in sheets.values()
        for v in df.head(30).astype(str).to_numpy().ravel()
    ).lower()
    # normalize the latin-1 mojibake the BDE export produces (índice -> �ndice)
    blob = blob.replace("�", "").replace("í", "i").replace("é", "e").replace("á", "a")

    wrong_markers = ["ventas efectivas", ".cvv.", "operaciones de compraventa"]
    hit = next((m for m in wrong_markers if m in blob), None)
    if hit:
        sys.exit(
            f"ABORT: {path.name} is NOT the IPV.\n"
            f"  Found marker: '{hit}'\n"
            f"  This looks like 'Indicadores de ventas efectivas de viviendas' (series\n"
            f"  F034.CVV.*), which is a TRANSACTION-VOLUME index (how many homes were\n"
            f"  sold), not a PRICE index.\n"
            f"  Re-download the cuadro 'Indice de precios de vivienda (IPV)' from the BDE\n"
            f"  (Estadisticas experimentales chapter). See README.md."
        )

    price_markers = ["precios de vivienda", "precio de vivienda", "ipv"]
    if not any(m in blob for m in price_markers):
        sys.exit(
            f"ABORT: {path.name} does not look like a housing PRICE index - none of\n"
            f"  {price_markers} appear in its header/metadata sheets.\n"
            f"  Run `python load.py --inspect` and confirm you exported the IPV cuadro."
        )


def _tidy_sheet(df: pd.DataFrame, min_quarters: int = 8) -> pd.DataFrame | None:
    """Reshape one BDE sheet to long form: (quarter, series_col, index_value).

    The BDE exports the IPV cuadro wide - one header row of quarter dates, then
    one row per series whose leading cell is the series description. We also
    handle the transposed layout (quarters down a column) since other BDE
    cuadros export that way. Returns None if the sheet has no quarter axis
    (metadata sheets).
    """
    # --- wide: quarters across a header row ---
    row_hits = {i: df.loc[i].map(lambda v: _parse_quarter(v) is not None).sum() for i in df.index}
    header_row = max(row_hits, key=row_hits.get)
    if row_hits[header_row] >= min_quarters:
        quarters = df.loc[header_row].map(_parse_quarter)
        qcols = [c for c in df.columns if quarters[c] is not None and pd.notna(quarters[c])]
        labcols = [c for c in df.columns if c not in qcols]
        body = df.loc[[i for i in df.index if i > header_row]]
        if body.empty:
            return None
        # Series name from the label cells. BDE cuadros often lead with a "Reg"
        # hierarchy-numbering column ("1.", "1.1.", "1.1.2.") before the real
        # "Descripción series" column, so skip pure-numbering tokens and keep
        # the descriptive text; fall back to the numbering only if that is all
        # there is.
        def _name(row) -> str:
            cells = [str(v).strip() for v in row if str(v).strip() and str(v).strip().lower() != "nan"]
            descriptive = [c for c in cells if not re.fullmatch(r"[\d.]+", c)]
            return " ".join(descriptive) if descriptive else (" ".join(cells) if cells else "")

        names = (body[labcols].apply(_name, axis=1)
                 if labcols else pd.Series("", index=body.index))
        names = names.mask(names.eq(""), pd.Series(body.index, index=body.index).map(lambda i: f"row{i}"))
        out = pd.DataFrame(
            body[qcols].apply(pd.to_numeric, errors="coerce").to_numpy(),
            index=pd.Index(names, name="series_col"),
            columns=pd.Index([quarters[c] for c in qcols], name="quarter"),
        )
        return (out.stack().rename("index_value").reset_index()
                   .dropna(subset=["index_value"]))

    # --- transposed: quarters down a column ---
    col_hits = {c: df[c].map(lambda v: _parse_quarter(v) is not None).sum() for c in df.columns}
    qcol = max(col_hits, key=col_hits.get)
    if col_hits[qcol] < min_quarters:
        return None
    quarters = df[qcol].map(_parse_quarter)
    keep = quarters.notna()
    vals = df.loc[keep, [c for c in df.columns if c != qcol]].apply(pd.to_numeric, errors="coerce")
    vals.index = pd.Index(quarters[keep], name="quarter")
    vals.columns = pd.Index([f"col{c}" for c in vals.columns], name="series_col")
    return (vals.stack().rename("index_value").reset_index()
                .dropna(subset=["index_value"]))


def build() -> None:
    files = _raw_files()
    if not files:
        sys.exit(f"No Excel files in {RAW_DIR} - download the IPV cuadro first (see README).")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for path in files:
        _assert_is_price_index(path, _read_all_sheets(path))

    tidy_frames = []
    for path in files:
        for sheet, df in _read_all_sheets(path).items():
            tidy = _tidy_sheet(df)
            if tidy is None:
                continue
            tidy["source_sheet"] = sheet
            tidy_frames.append(tidy)

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

    # CRITERIA.md needs the IPV "per zone and dwelling type"; the BDE lets you
    # export just the headline series, which parses fine but cannot support the
    # per-zone drift analysis. Say so loudly rather than discovering it later.
    names = " ".join(tidy["series_col"].unique()).lower()
    has_dwelling = any(k in names for k in ("casa", "departamento", "depto"))
    has_zone = any(k in names for k in ("zona", "region", "regi", "metropolitana", "rm ", "norte", "sur", "centro"))
    if not (has_dwelling and has_zone):
        print(
            "\nWARNING: this export has no zone/dwelling-type disaggregation "
            f"({n_series} series: {list(tidy['series_col'].unique())[:4]}).\n"
            "  CRITERIA.md specifies the temporal target 'per zone and dwelling type'.\n"
            "  The headline series alone supports only an aggregate drift analysis.\n"
            "  To get the full breakdown, re-export the IPV cuadro from the BDE with the\n"
            "  casas/departamentos and geographic-zone series selected.\n"
        )

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
        "dataset": "central_bank_hpi",
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
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


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
