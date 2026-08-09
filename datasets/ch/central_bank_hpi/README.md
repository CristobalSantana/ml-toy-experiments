# datasets/ch/central_bank_hpi - Central Bank of Chile housing price index (IPV)

Chile's quarterly **Housing Price Index** (*Índice de Precios de Vivienda*, IPV)
from the Banco Central de Chile. It is built from SII administrative records of
actual residential transactions (form F2890) and published quarterly with a
~120-day lag. This is the source for the temporal / drift arm of the
[real-estate-valuation](../../../experiments/real-estate-valuation/) experiment.

Unlike the [SII cadastre](../sii_cadastre/), which records an administrative
tax base, the IPV tracks **real transaction prices** - the two are different
quantities and are never mixed.

> **Retrieved 2026-08-09.** The navigation path, cuadro codes and the series
> listed below were valid on that date. The Central Bank reorganises its
> statistical database and revises series composition periodically, so **both
> the links and the series may have changed since**. Check the cuadro contents
> before relying on this page.

## How to download

The programmatic route (the `SieteRestWS` API) requires free registered
credentials; we use the manual Excel export instead, so there are no
credentials to manage.

Two cuadros are needed, from the Base de Datos Estadísticos
(<https://si3.bcentral.cl/siete>):

| Cuadro | Direct link (valid 2026-08-09) |
|---|---|
| IPV general, houses and apartments | `.../Cuadro/CAP_TASA_INTERES/MN_IND_VIVIENDA/IS_GENERAL_PROPIEDAD_08/638289254701071624` |
| IPV by geographic zone | `.../Cuadro/CAP_IND_SEC/MN_IND_SEC20/IS_PRECIOS_GEO_2008/IS64_b` |

In each one, **tick every series before exporting** - the BDE exports only the
visible series by default, which is easy to miss and yields a single headline
column. Save the `.xlsx` files anywhere under `raw/`; the loader globs `*.xls*`
and merges them, so file names and file count do not matter.

```
datasets/ch/central_bank_hpi/raw/
    IS_GENERAL_PROPIEDAD_08.xlsx
    IS_PRECIOS_GEO_2008.xlsx
```

**Watch out for the wrong cuadro.** The Vivienda section also carries
*"Indicadores de ventas efectivas de viviendas"* (series `F034.CVV.*`), a
**transaction-volume** index that is quarterly and split by dwelling type and
so looks like the IPV at a glance. It counts how many homes sold, not at what
price. `load.py` aborts with an explanation if it sees one.

## Columns / sheets in the downloaded files

Each BDE export is an `.xlsx` with three sheets.

### Sheet `Cuadro` - the data, in wide form

| Position | Content |
|---|---|
| Row 1 | Cuadro title, e.g. *"IPV General, casas y departamentos"* |
| Row 3 | Header: a `Reg` hierarchy-numbering column, a `Descripción series` column, then **one column per quarter** as a real date (`2014-03-01`, `2014-06-01`, …) |
| Rows 4+ | One row per series: its numbering (`1.`, `1.1.`, `1.1.2.`), its description, then the index value for each quarter |

The `Reg` column is numbering, not a name - reading it as the series label
yields rows called `1.` and `1.1.2.` instead of `IPV Casas`.

### Sheet `Metadatos`

Source, unit (índice), frequency (trimestral), publication lag (4 meses), base
year (2008), last update, first and last observation, methodology link.

### Sheet `Series`

The BDE series code and full description for each series, e.g.
`F034.IPV.FLU.BCCH.2008.0.T` = *"Índice precios de vivienda, BCCh"*.

### The 17 series retrieved

| Group | Series |
|---|---|
| Headline | `Índice precios de vivienda, BCCh` (2002-2026) |
| By dwelling type | `IPV General`, `IPV Casas`, `IPV Casas Nuevas`, `IPV Casas Usadas`, `IPV Departamentos`, `IPV Departamentos Nuevos`, `IPV Departamentos Usados`, `IPV Nuevas`, `IPV Usadas` |
| By zone (Metropolitan Region) | `Región Metropolitana`, `RM Centro`, `RM Oriente`, `RM Oriente Casas`, `RM Oriente Dptos.`, `RM Poniente`, `RM Sur` |

The disaggregated series start in 2014; the headline goes back to 2002. The
Central Bank publishes 19 series in total - the three **non-RM national zones**
were not exported here, since all eight cadastre comunas are in the
Metropolitan Region. `IPV Casas Nuevas` ends in 2022: the Central Bank
discontinued it for too few transactions.

## What `load.py` produces

`processed/ipv_quarterly.csv`, in tidy long form:

| Column | Meaning |
|---|---|
| `quarter` | First month of the quarter, as a date |
| `series_col` | Series description (e.g. `IPV Casas Usadas`) |
| `index_value` | Index level, base 2008 = 100 |
| `source_sheet` | Sheet the row came from, for traceability |

## Run

```bash
python load.py --inspect   # print each sheet's shape and head
python load.py             # parse raw/*.xls* -> processed/ + MANIFEST.sha256
python load.py --validate  # re-check raw files against the committed manifest
```
