"""
load.py -- Parse the SII "Detalle Catastral" per-comuna files into one clean
rol-level table for Experiment 02 (cross-sectional arm).

Reads the two non-agrícola (urban) files per comuna from raw/<comuna_code>/,
aggregates construction lines per rol, joins to the rol record, computes the
target (avalúo fiscal per m² built, deflated to UF), and writes:

    processed/sii_rol_level.csv     one row per residential rol
    MANIFEST.sha256                 checksums + provenance (committed)

The file format (no header, "|"-separated, latin-1) is documented in this
folder's README and in the SII's estructura_detalle_catastral.pdf. Column
indices below follow that spec. The loader validates loudly: it aborts if a
comuna is missing, if a file's column count is implausible, or if the target
comes out non-finite.

Nothing here automates the SII login; the raw files are downloaded manually
(they sit behind personal RUT+Clave credentials). See README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
PROCESSED_DIR = HERE / "processed"
MANIFEST = HERE / "MANIFEST.sha256"

# 8 Región Metropolitana comunas (SII codes from estructura_detalle_catastral.pdf)
COMUNAS = {
    "15108": "Las Condes", "15103": "Providencia", "15105": "Nunoa",
    "15151": "Macul", "15128": "La Florida", "14109": "Maipu",
    "16301": "Puente Alto", "16165": "El Bosque",
}

# 0-based column indices into the pipe-separated files (spec is 1-based).
ROLES_COLS = {          # "Roles no agrícolas": one record per rol
    "comuna": 0, "manzana": 1, "predial": 2,
    "avaluo_total": 4, "destino": 6, "avaluo_exento": 7, "sup_terreno": 14,
}
CONSTR_COLS = {         # "Terrenos y construcciones no agrícolas": many per rol
    "comuna": 0, "manzana": 1, "predial": 2,
    "material": 4, "calidad": 5, "anio": 6, "sup_construida": 7, "pisos": 10,
}
QUALITY_LABELS = {1: "Superior", 2: "Media Superior", 3: "Media",
                  4: "Media Inferior", 5: "Inferior"}


def _to_num(series: pd.Series) -> pd.Series:
    """Coerce a raw text column to numeric, tolerating stray spaces and the
    Chilean thousands/decimal punctuation that sometimes leaks into exports."""
    s = series.astype(str).str.strip().str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _classify(name: str) -> str | None:
    """Classify a member filename: 'CATASNL' = non-agrícola construction lines,
    'CATASN' (not NL) = non-agrícola roles, everything else (agrícola CATASA/
    CATASAL, or the .zip itself) -> None (skip). Order matters: NL first."""
    n = name.upper()
    if "CATASNL" in n:
        return "constr"
    if "CATASN" in n:
        return "roles"
    return None


def _iter_member_files(folder: Path):
    """Yield (name, bytes) for candidate data files in a comuna folder,
    transparently reading from a per-comuna .zip if present (the SII download
    ships one ZIP with the 4 detalle-catastral files), else loose files."""
    zips = [p for p in folder.iterdir() if p.suffix.lower() == ".zip"]
    if zips:
        for zp in zips:
            with zipfile.ZipFile(zp) as z:
                for info in z.infolist():
                    if not info.is_dir() and info.file_size > 0:
                        yield info.filename, z.read(info.filename)
    else:
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() != ".zip":
                yield p.name, p.read_bytes()


def _read_pipe(data: bytes, name: str, ncols_min: int) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data), sep="|", header=None, dtype=str,
                     encoding="latin-1", engine="python", on_bad_lines="skip")
    if df.shape[1] < ncols_min:
        sys.exit(f"ABORT: {name} has {df.shape[1]} columns, expected >= {ncols_min}. "
                 f"File format may differ from the documented spec - inspect it.")
    return df


def _load_comuna(code: str) -> pd.DataFrame:
    folder = RAW_DIR / code
    if not folder.is_dir():
        sys.exit(f"ABORT: missing raw folder for comuna {code} ({COMUNAS[code]}): {folder}\n"
                 f"Download its Detalle Catastral and place the files there (see README).")
    roles_parts, constr_parts = [], []
    for name, data in _iter_member_files(folder):
        kind = _classify(name)
        if kind == "roles":
            roles_parts.append(_read_pipe(data, name, 15))
        elif kind == "constr":
            constr_parts.append(_read_pipe(data, name, 11))
    if not roles_parts or not constr_parts:
        sys.exit(f"ABORT: comuna {code}: could not find both non-agrícola files "
                 f"(roles={len(roles_parts)}, constr={len(constr_parts)}). "
                 f"Expected BRTMPCATASN_* and BRTMPCATASNL_* inside the download.")

    roles = pd.concat(roles_parts, ignore_index=True)
    constr = pd.concat(constr_parts, ignore_index=True)

    # --- rol-level record ---
    r = pd.DataFrame({
        "comuna": roles[ROLES_COLS["comuna"]].str.strip(),
        "manzana": roles[ROLES_COLS["manzana"]].str.strip(),
        "predial": roles[ROLES_COLS["predial"]].str.strip(),
        "avaluo_fiscal_total": _to_num(roles[ROLES_COLS["avaluo_total"]]),
        "destino": roles[ROLES_COLS["destino"]].str.strip(),
        "avaluo_exento": _to_num(roles[ROLES_COLS["avaluo_exento"]]),
        "sup_terreno_m2": _to_num(roles[ROLES_COLS["sup_terreno"]]),
    })

    # --- aggregate construction lines per rol ---
    c = pd.DataFrame({
        "comuna": constr[CONSTR_COLS["comuna"]].str.strip(),
        "manzana": constr[CONSTR_COLS["manzana"]].str.strip(),
        "predial": constr[CONSTR_COLS["predial"]].str.strip(),
        "material": constr[CONSTR_COLS["material"]].str.strip(),
        "calidad": _to_num(constr[CONSTR_COLS["calidad"]]),
        "anio": _to_num(constr[CONSTR_COLS["anio"]]),
        "sup_construida_m2": _to_num(constr[CONSTR_COLS["sup_construida"]]),
        "pisos": _to_num(constr[CONSTR_COLS["pisos"]]),
    })
    keys = ["comuna", "manzana", "predial"]
    # area-weighted quality: sum(calidad*area)/sum(area) over lines with a quality
    c["_cal_x_area"] = c["calidad"] * c["sup_construida_m2"]
    c["_area_if_cal"] = c["sup_construida_m2"].where(c["calidad"].notna())
    g = c.groupby(keys, dropna=False)
    agg = g.agg(
        sup_construida_m2=("sup_construida_m2", "sum"),
        anio_construccion=("anio", "max"),          # newest line as the rol's build year
        n_pisos=("pisos", "max"),
        n_lineas_construccion=("sup_construida_m2", "size"),
        _cal_area=("_cal_x_area", "sum"),
        _area_cal=("_area_if_cal", "sum"),
    ).reset_index()
    agg["calidad_ponderada"] = agg["_cal_area"] / agg["_area_cal"].replace(0, np.nan)
    mat = g["material"].agg(lambda s: s.mode().iloc[0] if not s.mode(dropna=True).empty else None)
    agg = agg.merge(mat.rename("material_predom").reset_index(), on=keys)
    agg = agg.drop(columns=["_cal_area", "_area_cal"])

    df = r.merge(agg, on=keys, how="left")
    df["comuna_code"] = code
    df["comuna_nombre"] = COMUNAS[code]
    return df


def _fetch_uf(retrieval: date) -> tuple[float, str]:
    """UF value at the retrieval date, to deflate nominal-peso avalúos into UF.
    For a single cross-sectional snapshot this is one constant; it only sets the
    unit scale. Source: mindicador.cl (free, no auth). Falls back to --uf."""
    url = f"https://mindicador.cl/api/uf/{retrieval.strftime('%d-%m-%Y')}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        val = float(data["serie"][0]["valor"])
        return val, f"mindicador.cl ({url})"
    except Exception as e:  # noqa: BLE001
        sys.exit(f"ABORT: could not fetch UF from mindicador.cl ({e}). "
                 f"Re-run with --uf <value> to supply it manually.")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_cycle(raw_files: list[Path]) -> str:
    """Read the reavalúo cycle straight off the SII filenames, which encode it
    as BRTMPCATAS_<year>_<period>_<comuna>.zip - so provenance records the
    actual cycle of the files on disk rather than a hand-typed guess."""
    cycles = set()
    for p in raw_files:
        m = re.search(r"BRTMPCATAS_(\d{4})_(\d+)_", p.name.upper())
        if m:
            cycles.add(f"{m.group(1)}-{m.group(2)}")
    if not cycles:
        return "unknown (filenames did not encode a cycle)"
    label = ", ".join(sorted(cycles))
    warn = "" if len(cycles) == 1 else "  WARNING: comunas span different cycles!"
    return f"non-agrícola, SII cycle {label} (from filenames){warn}"


def build(uf_override: float | None) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    retrieval = date.today()

    frames = [_load_comuna(code) for code in COMUNAS]
    df = pd.concat(frames, ignore_index=True)

    uf_value, uf_source = (uf_override, "manual --uf") if uf_override else _fetch_uf(retrieval)

    # Primary residential target: avalúo per m² built, in UF.
    res = df[(df["destino"] == "H") & (df["sup_construida_m2"] > 0)].copy()
    res["avaluo_uf_per_m2"] = (res["avaluo_fiscal_total"] / res["sup_construida_m2"]) / uf_value
    res["calidad_label"] = res["calidad_ponderada"].round().map(QUALITY_LABELS)

    # --- loud validation ---
    bad = ~np.isfinite(res["avaluo_uf_per_m2"])
    if bad.any():
        res = res[~bad]
    if len(res) == 0:
        sys.exit("ABORT: no residential rols with positive built area and finite target were produced.")
    # sanity band: UF/m² outside [0.5, 500] is almost certainly a parse error
    band = res["avaluo_uf_per_m2"].between(0.5, 500)
    print(f"target UF/m2: median={res['avaluo_uf_per_m2'].median():.2f}, "
          f"{100*band.mean():.1f}% within [0.5, 500] plausibility band")

    out = PROCESSED_DIR / "sii_rol_level.csv"
    res.to_csv(out, index=False, encoding="utf-8")

    # --- manifest (committed): provenance without the bulk bytes ---
    raw_files = sorted(p for p in RAW_DIR.rglob("*") if p.is_file())
    manifest = {
        "dataset": "sii_cadastre",
        "retrieval_date": retrieval.isoformat(),
        "reavaluo_cycle": _detect_cycle(raw_files),
        "source": "SII Sitio de Transparencia -> Detalle Catastral y ROL de cobro (manual, login-gated)",
        "uf_value_used": uf_value, "uf_source": uf_source,
        "comunas": COMUNAS,
        "n_rows_residential": int(len(res)),
        "n_rows_all_roles": int(len(df)),
        "raw_files": {str(p.relative_to(HERE)): {"sha256": _sha256(p), "bytes": p.stat().st_size}
                      for p in raw_files},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out} ({len(res)} residential rols) and {MANIFEST.name}")


def validate() -> None:
    if not MANIFEST.exists():
        sys.exit("No MANIFEST.sha256 - run `python load.py` first.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mismatch = []
    for rel, meta in manifest["raw_files"].items():
        p = HERE / rel
        if not p.exists() or _sha256(p) != meta["sha256"]:
            mismatch.append(rel)
    if mismatch:
        sys.exit(f"CHECKSUM MISMATCH on {len(mismatch)} raw file(s): {mismatch}")
    print(f"OK: {len(manifest['raw_files'])} raw files match the committed manifest.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--validate", action="store_true", help="check raw files against the manifest and exit")
    ap.add_argument("--uf", type=float, default=None, help="UF value to use instead of fetching it")
    args = ap.parse_args()
    validate() if args.validate else build(args.uf)
