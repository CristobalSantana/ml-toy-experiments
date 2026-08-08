# datasets/ch - Chile

Public Chilean datasets. See [`../README.md`](../README.md) for the folder
convention shared by all countries.

| Dataset | Source | What it is |
|---|---|---|
| [`sii_cadastre`](sii_cadastre/) | Servicio de Impuestos Internos (SII) | "Detalle Catastral": assessed fiscal value (avalúo fiscal), built/land area, construction quality, year and destino, per `rol` (comuna-manzana-predio) |
| [`bcch_ipv`](bcch_ipv/) | Banco Central de Chile (BDE) | IPV: quarterly housing **price** index, broken down by dwelling type (casas/departamentos, nuevas/usadas) and geographic zone |

## Notes on Chilean data

- **UF, not pesos.** Property is quoted in *Unidades de Fomento*. Nominal pesos
  carry an inflationary trend that a model will happily learn as signal, so
  anything monetary is deflated to UF (or constant-date currency) before use.
- **Assessed value ≠ market price.** The SII's avalúo fiscal is an
  administrative tax base fixed at *reavalúo* and indexed between cycles. It is
  not a transaction price, and nothing here should be presented as an estimate
  of one. The IPV, by contrast, *is* built from real transactions (SII form
  F2890) - the two are different quantities and are kept separate.
- **Reavalúo cycles matter.** Assessed values step at each reavalúo, so any
  panel spanning a cycle boundary has a structural break by construction - the
  cycle is recorded in each `MANIFEST.sha256`.
- **Both sources are credential-gated for bulk access**, so the raw files are
  downloaded manually and parsed locally; the loaders never automate a login.
