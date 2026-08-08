# datasets

Public, real-world datasets used by experiments - as opposed to the
**synthetic** data in [`generators/`](../generators/), which is produced from a
known equation.

Datasets are grouped **by country** (ISO 3166-1 alpha-2, lowercase), because
public data is almost always national: its source, licence, units and
administrative concepts only make sense within one country's system.

```
datasets/
    ch/          Chile
        <name>/
            load.py            downloads and/or parses the data into a clean table
            README.md          what it is, its source, licence, and provenance
            MANIFEST.sha256    checksums + retrieval date (committed)
            raw/               source files - gitignored, not redistributed
            processed/         derived tables - gitignored, rebuilt by load.py
    <cc>/        (next country, same layout)
```

Guidelines:

- **Document the source and licence** of every dataset in its `README.md`.
  Only use openly available data, and respect each source's terms of use.
- **Don't commit bulk data.** `raw/` and `processed/` are gitignored; the
  committed `MANIFEST.sha256` records exactly which files were retrieved, when,
  and their checksums - so provenance is versioned without the bytes.
- **Keep loading deterministic** and cached: a rerun must never re-hit a remote
  API, so an experiment that depends on a dataset stays reproducible.
- **Fail loudly** on the wrong input. Portals change and it is easy to download
  a similarly-named series; a loader should abort with an explanation rather
  than quietly parse the wrong quantity.

## Available datasets

| Country | Dataset | What it is |
|---|---|---|
| 🇨🇱 `ch` | [`sii_cadastre`](ch/sii_cadastre/) | SII "Detalle Catastral" - assessed fiscal value (avalúo fiscal) per property |
| 🇨🇱 `ch` | [`bcch_ipv`](ch/bcch_ipv/) | Banco Central IPV - quarterly housing price index, by zone and dwelling type |
