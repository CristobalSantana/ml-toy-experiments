# datasets/bcch_ipv - Banco Central IPV (Índice de Precios de Vivienda)

Chile's quarterly Housing Price Index from the Banco Central de Chile (BCCh),
the source for Experiment 02's **temporal / drift arm**. The IPV is built from
SII administrative records of actual residential transactions (form F2890),
published quarterly with a ~120-day lag.

## What to download

The clean programmatic path (SieteRestWS API) needs free BCCh credentials; we
chose the **manual download** from the Base de Datos Estadísticos (BDE)
instead - no credentials to manage.

1. Go to the BDE: <https://si3.bcentral.cl/siete> (or the IPV content page
   <https://www.bcentral.cl/areas/estadisticas/estadisticas-experimentales/ipv>).
2. Navigate: **Indicadores sectoriales → Construcción → Vivienda → IPV**.
3. Select, from **2014 to the latest available quarter**:
   - IPV **general**, and the **casa** (house) and **departamento** (apartment)
     breakdowns;
   - the **geographic disaggregation** (7 zones, with the Región
     Metropolitana split into 4) if available in the cuadro.
4. Export to **Excel** and save it under `raw/`:

```
datasets/bcch_ipv/raw/
    ipv_bcch.xlsx        (or whatever the BDE names it - the loader globs *.xls*)
```

`raw/` is gitignored; the loader writes a committed checksum manifest.

## Provenance to record

The loader stamps the retrieval date and the source into `MANIFEST.sha256`.
Note the **120-day publication lag**: the most recent calendar quarter is
usually not yet published - the held-out "most recent 4 quarters" in
`CRITERIA.md` means the last 4 *published* quarters at retrieval time.

## Run

```bash
python load.py --inspect   # print sheet names + head, to confirm the layout
python load.py             # parse raw/*.xls* -> processed/ipv_quarterly.csv + MANIFEST.sha256
python load.py --validate  # re-check raw files against the committed manifest
```

The processed table is tidy/long: one row per `(quarter, zone, dwelling_type)`
with the index level. The exact sheet/column reshape is finalized against the
actual BDE export (its layout varies with how the cuadro is exported), which is
why `--inspect` exists.
