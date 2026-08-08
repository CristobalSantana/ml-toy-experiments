# datasets/sii_cadastre - SII "Detalle Catastral" (assessed fiscal value)

Per-`rol` cadastral records from Chile's Servicio de Impuestos Internos (SII):
the **avalúo fiscal** (assessed fiscal value) of each property, plus its
built area, land area, construction quality, year, material and destino. This
is the source for Experiment 02's **cross-sectional arm**.

> **Assessed value is not market price.** The avalúo fiscal is an
> administrative tax base set at reavalúo and adjusted by IPC. We model it as
> itself; nothing here is an estimate of transaction/market price.

## Provenance & access reality (deviation from the brief)

The brief assumed the SII cadastre is "public, no authentication". That holds
for the **map viewer** and single-property certificates, but the **bulk
per-comuna download requires an SII login (RUT + Clave Tributaria)**:

- Path: SII → *Sitio de Transparencia* → **"Detalle Catastral y ROL de cobro"**
  → `zeusr.sii.cl/AUT2000/InicioAutenticacion/IngresoRutClave.html?...caller=DETALLE_CAT_Y_ROL_COBRO`

Because those are personal tax credentials, **the download is manual** (done by
the repo owner, with their own login); this loader never sees or automates the
login. This is recorded as a deviation in the experiment README, not in
`CRITERIA.md`.

**2026 reavalúo note:** a non-agrícola reavalúo is in process during 2026, so
the vigente values reflect the current cycle. Record the exact retrieval date
and the reavalúo cycle in `MANIFEST.sha256` (the loader stamps this).

## What to download (8 Región Metropolitana comunas)

Log in, choose **Detalle Catastral y ROL de cobro**, and download the cadastral
detail for each comuna below (SII comuna codes confirmed from the official
`estructura_detalle_catastral.pdf`):

| Comuna | SII code |
|---|---|
| Las Condes | 15108 |
| Providencia | 15103 |
| Ñuñoa | 15105 |
| Macul | 15151 |
| La Florida | 15128 |
| Maipú | 14109 |
| Puente Alto | 16301 |
| El Bosque | 16165 |

Place the downloaded files under `raw/`, one folder per comuna code:

```
datasets/sii_cadastre/raw/
    15108/   (Las Condes: the pipe-delimited detalle catastral files)
    15103/   (Providencia)
    ...
    16165/   (El Bosque)
```

`raw/` is gitignored. Keep the files exactly as downloaded; the loader parses
them and writes a checksum manifest so provenance is versioned.

## File format (from estructura_detalle_catastral.pdf)

The download contains 4 files; **we use the two non-agrícola (urban) ones**.
No header row, fields separated by `|`, one comuna per download.

**Roles no agrícolas** (one record per rol) - fields we read:
1 comuna · 2 manzana · 3 predial · 5 **avalúo fiscal total** · 7 destino
(H=Habitacional, …) · 8 avalúo exento · 15 **superficie de terreno (m²)**

**Terrenos y construcciones no agrícolas** (several records per rol, one per
construction line):
1 comuna · 2 manzana · 3 predial · 5 material · 6 **calidad** (1=Superior…
5=Inferior) · 7 **año** · 8 **superficie construida (m²)** · 11 nº pisos

The loader keys both on `(comuna, manzana, predial)`, aggregates the
construction lines per rol (total built area, dominant quality/material,
year, floors), and joins to the rol table.

## Target & units

`load.py` produces one row per residential rol (destino H, built area > 0)
with:
- **`avaluo_uf_per_m2`** = avalúo fiscal total ÷ total built area, **deflated
  to UF** using the UF value at the retrieval date (constant across this
  cross-sectional snapshot; source and value stamped in the manifest).
- Structural features: built area, land area, quality, material, year, floors,
  destino.
- Locational keys: comuna, manzana (the grouping unit for leakage-safe CV).

`avaluo_fiscal_total` and `superficie_construida` are both kept in the
processed table, but the modelling code **must not** use both as features
alongside the per-m² target (that would reconstruct the target exactly) - the
leakage check in the experiment enforces this.

## Run

```bash
python load.py            # parse raw/ -> processed/sii_rol_level.parquet + MANIFEST.sha256
python load.py --validate # re-check that processed matches the committed manifest
```
