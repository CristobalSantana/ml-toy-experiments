"""
features.py -- Build the modelling table for Experiment 02's cross-sectional arm.

Takes the rol-level SII table produced by datasets/ch/sii_cadastre/load.py and
returns a clean feature matrix, the target, and the manzana grouping key, with
every cleaning decision driven by config.yaml and reported out loud.

The target is log10(avalúo fiscal per m² built, in UF). Deflation to UF happens
in the dataset loader, not here, so it is applied exactly once.

Assessed value is not market price. Nothing produced here estimates what a
property would sell for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

import leakage  # noqa: E402

REFERENCE_YEAR = 2026  # the SII cycle these avalúos belong to (see MANIFEST)


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or EXPERIMENT_ROOT / "config.yaml").read_text(encoding="utf-8"))


def load_raw(cfg: dict) -> pd.DataFrame:
    path = REPO_ROOT / cfg["data"]["sii_processed"]
    if not path.exists():
        sys.exit(f"Missing {path}\nRun: python datasets/ch/sii_cadastre/load.py")
    return pd.read_csv(path)


def clean(df: pd.DataFrame, cfg: dict, verbose: bool = True) -> pd.DataFrame:
    """Drop or repair the quirks measured in the raw cadastre.

    These counts are real properties of the SII export, not hypotheticals: 34
    rows carry impossible construction years (e.g. 1514), 405 rows fall outside
    the documented 1..5 quality scale, and n_pisos runs up to 224 (no building
    in Chile exceeds ~64 floors).
    """
    c = cfg["cleaning"]
    n0 = len(df)
    out = df.copy()
    dropped = {}

    # implausible / unusable rows -> dropped (they cannot be repaired honestly)
    small = out["sup_construida_m2"] < c["min_built_area_m2"]
    dropped["built area below minimum"] = int(small.sum())
    out = out[~small]

    lo, hi = c["target_band"]
    off_band = ~out["avaluo_uf_per_m2"].between(lo, hi)
    dropped["target outside plausibility band"] = int(off_band.sum())
    out = out[~off_band]

    # repairable fields -> clipped to their documented range, with the
    # out-of-range count reported rather than silently absorbed
    bad_year = ~out["anio_construccion"].between(c["year_min"], c["year_max"])
    out.loc[bad_year, "anio_construccion"] = np.nan

    bad_q = ~out["calidad_ponderada"].between(c["quality_min"], c["quality_max"])
    out["calidad_ponderada"] = out["calidad_ponderada"].clip(c["quality_min"], c["quality_max"])

    bad_floors = out["n_pisos"] > c["floors_max"]
    out["n_pisos"] = out["n_pisos"].clip(upper=c["floors_max"])

    if verbose:
        print(f"  cleaning: {n0} -> {len(out)} rows")
        for k, v in dropped.items():
            if v:
                print(f"    dropped {v} ({100*v/n0:.3f}%): {k}")
        print(f"    year set to NaN (outside {c['year_min']}-{c['year_max']}): {int(bad_year.sum())}")
        print(f"    quality clipped to [{c['quality_min']},{c['quality_max']}]: {int(bad_q.sum())}")
        print(f"    floors clipped to <= {c['floors_max']}: {int(bad_floors.sum())}")
    return out


def build_features(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Return (X, y, groups, built_area).

    Property type is derived, not assumed: a rol with zero land area is an
    apartment, because a departamento's land sits in the shared 'bien común'
    rol rather than its own. Verified in the data - those rows have a median of
    8 floors versus 1, and a higher UF/m² (32.1 vs 20.5).
    """
    f = pd.DataFrame(index=df.index)

    f["log10_sup_construida_m2"] = np.log10(df["sup_construida_m2"])
    # log10 of land area, with apartments (0 m²) mapped to NaN and flagged;
    # imputing 0 -> -inf would be nonsense, and 0 is meaningful, not missing.
    land = df["sup_terreno_m2"].replace(0, np.nan)
    f["log10_sup_terreno_m2"] = np.log10(land)
    # float, not int: sklearn's partial_dependence refuses integer columns
    # (implicit rounding), and the 0/1 values are identical either way.
    f["es_departamento"] = (df["sup_terreno_m2"] == 0).astype(float)

    f["calidad_ponderada"] = df["calidad_ponderada"]
    # Construction year only. An `edad_anios = REFERENCE_YEAR - year` column was
    # dropped after the collinearity diagnostic flagged it: age is an exact
    # linear function of year (Spearman -1.000, VIF ~1e15), so it is the same
    # information twice - it destabilises linear coefficients and splits tree
    # importance arbitrarily between two identical columns.
    f["anio_construccion"] = df["anio_construccion"]
    f["n_pisos"] = df["n_pisos"]
    f["n_lineas_construccion"] = df["n_lineas_construccion"]

    f["comuna_nombre"] = df["comuna_nombre"].astype("category")
    f["material_predom"] = df["material_predom"].astype("category")

    y = np.log10(df[cfg["target"]["source_column"]]).rename(cfg["target"]["name"])
    groups = (df["comuna_code"].astype(str) + "-" + df["manzana"].astype(str)).rename(
        cfg["group_key"]["name"])
    return f, y, groups, df["sup_construida_m2"]


def freeze_holdout(groups: pd.Series, cfg: dict, verbose: bool = True) -> pd.Series:
    """Boolean mask marking the frozen held-out manzanas.

    CRITERIA.md: a fixed 10% of manzanas, chosen by a fixed seed at the start,
    untouched until the final run. Selection is over *sorted unique* manzanas
    with a seeded RNG, so it depends only on the seed and the set of manzanas -
    never on row order or on any value of the target.
    """
    uniq = np.sort(groups.unique())
    rng = np.random.default_rng(cfg["seed"])
    n_hold = int(round(cfg["split"]["holdout_manzana_frac"] * len(uniq)))
    held = set(rng.choice(uniq, size=n_hold, replace=False))
    mask = groups.isin(held)
    if verbose:
        print(f"  holdout: {n_hold}/{len(uniq)} manzanas frozen "
              f"({100*mask.mean():.1f}% of rows) with seed {cfg['seed']}")
    return mask.rename("is_holdout")


def build(cfg: dict | None = None, verbose: bool = True):
    """Full Phase-3 pipeline: load -> clean -> features -> leakage -> holdout."""
    cfg = cfg or load_config()
    if verbose:
        print("Experiment 02 - building cross-sectional modelling table")

    df = clean(load_raw(cfg), cfg, verbose)
    X, y, groups, built_area = build_features(df, cfg)
    reports = leakage.run_all_cross_sectional(X, y, built_area, groups, cfg, verbose)
    holdout = freeze_holdout(groups, cfg, verbose)

    if verbose:
        dev = ~holdout
        print(f"  final: {len(X)} rows, {X.shape[1]} features "
              f"({int(dev.sum())} development / {int(holdout.sum())} held out)")
        print(f"  target {y.name}: median {y.median():.3f} "
              f"(= {10**y.median():.1f} UF/m2), sd {y.std():.3f}")
    return X, y, groups, holdout, reports


if __name__ == "__main__":
    build()
