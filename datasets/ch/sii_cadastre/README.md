# datasets/ch/sii_cadastre - SII "Detalle Catastral" (assessed fiscal value)

Per-`rol` cadastral records from Chile's Servicio de Impuestos Internos (SII):
the **avalúo fiscal** (assessed fiscal value) of each property, plus its built
area, land area, construction quality, year and use. This is the source for the
[real-estate-valuation](../../../experiments/real-estate-valuation/) experiment.

> **Assessed value is not market price.** The avalúo fiscal is an
> administrative tax base set at *reavalúo* and adjusted between cycles. We
> model it as itself; nothing here estimates a transaction price.

> **Retrieved 2026-08-09, reavalúo cycle 2026-1.** The download link, the file
> layout and the column list documented below were all valid on that date. The
> SII reorganises its portal and revises the cadastral format between cycles,
> so **both the link and the columns may have changed since**. Verify against
> the current `estructura_detalle_catastral.pdf` before trusting this page.

## How to download

The bulk per-comuna download is **not anonymous**: it sits behind an SII login
(RUT + Clave Tributaria). Because those are personal tax credentials, the
download is manual and this loader never automates the login.

1. SII → *Sitio de Transparencia* → **"Detalle Catastral y ROL de cobro"**
   (`zeusr.sii.cl/AUT2000/InicioAutenticacion/IngresoRutClave.html?...caller=DETALLE_CAT_Y_ROL_COBRO`)
2. Download the cadastral detail for each comuna below.
3. Drop each ZIP into its own folder named after the comuna code:

```
datasets/ch/sii_cadastre/raw/
    15108/  BRTMPCATAS_2026_1_15108.zip      (Las Condes)
    15103/  BRTMPCATAS_2026_1_15103.zip      (Providencia)
    ...
```

No need to unzip or rename: the loader reads members straight from the ZIP and
recognises the non-agrícola files by name.

| Comuna | SII code | | Comuna | SII code |
|---|---|---|---|---|
| Las Condes | 15108 | | La Florida | 15128 |
| Providencia | 15103 | | Maipú | 14109 |
| Ñuñoa | 15105 | | Puente Alto | 16301 |
| Macul | 15151 | | El Bosque | 16165 |

`raw/` and `processed/` are gitignored; the committed `MANIFEST.sha256` records
the checksums, retrieval date, reavalúo cycle and UF value used.

## Columns in the downloaded files

Each ZIP contains **four** pipe-delimited (`|`) text files with **no header
row**, encoded latin-1. Two cover agrícola (rural) property and two non-agrícola
(urban); this experiment uses only the urban pair. Column numbers below are
1-based, as in the SII spec.

### `BRTMPCATASN_*` - urban roles (one row per property)

| # | Column | Meaning |
|---|---|---|
| 1 | Código SII de la Comuna | Commune code (e.g. 15103 = Providencia) |
| 2 | Número de Manzana | City block number. With 1 and 3 forms the `rol` |
| 3 | Número de Predial | Plot/unit number within the block |
| 4 | Dirección o nombre del predio | Street address or property name |
| 5 | **Avalúo fiscal total** | **Total assessed value, nominal CLP** |
| 6 | Contribución semestral | Semi-annual property tax (incl. refuse charge) |
| 7 | Código de destino principal | Primary use: `H` habitacional, `Z` parking, `L` storage, `O` office, `C` commerce, … |
| 8 | Avalúo exento | Tax-exempt portion of the assessed value |
| 9-11 | Rol Bien Común 1 | Commune / block / plot of the first shared-property rol |
| 12-14 | Rol Bien Común 2 | Same, second shared-property rol |
| 15 | **Superficie total del terreno** | **Land area, m², no decimals** |
| 16 | Código de Ubicación | Location code |
| 17-19 | Rol Padre | Commune / block / plot of the parent rol |

### `BRTMPCATASNL_*` - urban construction lines (several rows per property)

| # | Column | Meaning |
|---|---|---|
| 1-3 | Comuna / Manzana / Predial | Join key back to the roles file |
| 4 | Número correlativo de línea | Line number within the property |
| 5 | Código de material estructural | `B` reinforced concrete, `C` masonry, `E` timber, `F` adobe, `G` steel sections, `K` prefabricated, … |
| 6 | **Código de calidad** | **1 Superior, 2 Media Superior, 3 Media, 4 Media Inferior, 5 Inferior** |
| 7 | **Año de la línea de construcción** | **Construction year** |
| 8 | **Superficie de la línea** | **Built area, m² (or m³ for some structures), no decimals** |
| 9 | Código de destino de la línea | Use of this construction line |
| 10 | Código de condición especial | `SB` basement, `AL` attic, `MS` mansard, … |
| 11 | Número de Pisos | Number of floors |

### `BRTMPCATASA_*` / `BRTMPCATASAL_*` - agrícola (rural)

Same idea for rural property, with soil-class columns instead of construction
lines. **Empty for all eight comunas used here**, which are urban.

## What `load.py` produces

`processed/sii_rol_level.csv`, one row per **residential** rol (`destino = H`
with positive built area). Construction lines are aggregated per rol: built
areas summed, quality area-weighted, newest year kept, modal material.

| Column | Meaning |
|---|---|
| `comuna_code`, `comuna_nombre` | Commune code and name |
| `manzana`, `predial` | Block and plot; `comuna+manzana` is the grouping key |
| `avaluo_fiscal_total` | Total assessed value, nominal CLP |
| `avaluo_exento` | Exempt portion |
| `sup_terreno_m2` | Land area, m² (0 for apartments - their land sits in the shared rol) |
| `sup_construida_m2` | Total built area, m² |
| `calidad_ponderada` | Area-weighted construction quality, 1 (best) to 5 |
| `material_predom` | Modal structural material |
| `anio_construccion` | Most recent construction year |
| `n_pisos` | Floors |
| `n_lineas_construccion` | Number of construction lines |
| **`avaluo_uf_per_m2`** | **Assessed value per m² built, in UF** - the modelling target |

Deflation to UF happens here, once, using the UF value at the retrieval date
(recorded in the manifest). Property in Chile is quoted in UF; nominal pesos
would inject an inflationary trend that any model would learn as signal.

## Run

```bash
python load.py             # parse raw/ -> processed/ + MANIFEST.sha256
python load.py --validate  # re-check raw files against the committed manifest
python load.py --uf 40000  # supply the UF value instead of fetching it
```
