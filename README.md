# ml-toy-experiments

Short, self-contained **toy experiments** in machine learning (or "AI", as
it tends to be called these days). Each one pairs a model or architecture
with data - either **synthetic** (generated from a known equation, so there
is a ground truth to check against) or **public** (a real, openly available
dataset) - and runs a small, fully reproducible study with an honest
write-up of what happened, including where the interesting idea *loses*, not
just where it wins.

Nothing here is meant to be state-of-the-art or production-scale. They are
"toy" on purpose: small enough to read end to end, fast enough to rerun, and
complete enough to trust.

## Structure

```
generators/   synthetic data generators (from a known equation)
datasets/     public / real-world datasets, grouped by country
models/       model & architecture implementations, one per folder
experiments/  each experiment combines a data source + one or more models
```

- **`generators/<name>/`** - `generate.py`, a `README.md` explaining the
  equation and its meaning, and an `outputs/` folder with the reference data.
- **`datasets/<country>/<name>/`** - a loader plus a `README.md` documenting
  the source, license, and provenance of the public data. Grouped by ISO
  country code (`ch`, ...), since public data is national: its source, units
  and administrative concepts only make sense within one country's system.
- **`models/<name>/`** - the implementation and a short `README.md`.
- **`experiments/<name>/`** - `run.py` (single entry point, runnable end to
  end), `config.yaml` (every reproducible parameter - constants, seeds,
  sizes), a `README.md` with the finding and how to reproduce it, and
  `outputs/`, `checkpoints/`, `media/` for the results.

## Experiments

| Experiment | Question | Finding |
|---|---|---|
| [`kan_vs_mlp_battery_diffusion`](experiments/kan_vs_mlp_battery_diffusion/) | Can a KAN solve a PDE as a physics-informed network as well as an MLP with the same parameter budget? | KAN is ~1.6x more accurate, consistently across 5 seeds - but ~16x slower to train. |
| [`real-estate-valuation`](experiments/real-estate-valuation/) | On real tabular data, does a tabular foundation model (TabPFN v2) beat gradient-boosted trees - and at what cost? | Classical methods win the pre-registered rule: accuracy ties inside TabPFN's envelope, but CatBoost gets there with 7x less time and 224x less memory. TabPFN's real limit is that it cannot ingest the data at all, which costs ~2x more error than any architecture difference. |
| [`drift-detector-overlap`](experiments/drift-detector-overlap/) | A drift detector is a classifier that tries to tell last month's data from this month's. Where does it stop working, and can that boundary be found on synthetic data first? | It fails as a cliff, not a slope: flawless until the distributions overlap ~80%, then it collapses - and where the cliff sits is set by the window size, not the drift. On real NYC taxi months the synthetic curve gets the *ordering* right and the *level* badly wrong, detecting 0.99 where it predicted 0.56. |
| [`solar-forecast-skill`](experiments/solar-forecast-skill/) | Solar forecasting papers report R² above 0.90. How much of that is forecasting, and how much is the sun rising and setting on schedule? | Climatology - the average output for that day and hour, which cannot see a cloud - scores R² 0.878 on a held-out year. The same forecasts score −5.75 to +0.89 once measured against the benchmark they have to beat. An echo state network with untrained recurrent weights matched a tuned gradient booster; at 24 hours ahead plain ridge regression beat both. |
| [`btc-01-rsi-divergence`](experiments/btc-01-rsi-divergence/) | Does RSI divergence, one of the most widely taught chart patterns, beat simply holding BTC after costs? | No, on both timeframes tested. Held out, it *lost* 33% (daily) and 24% (hourly) over a period when BTC rose 49%. The useful part is why it looked like it worked: the best of 18 parameter sets beat buy-and-hold in development and collapsed out of sample. |

> **On the BTC experiments.** These are numbered as a series and are studies of
> method, not trading advice. None of them is a recommendation, and a positive
> result would not make one. See each experiment's README for what it does and
> does not account for.

## Data

Two kinds of data feed the experiments:

**Synthetic** ([`generators/`](generators/)) - produced from a known
equation, so every experiment has an exact ground truth to measure against.

| Generator | Equation |
|---|---|
| [`diffusion_1d`](generators/diffusion_1d/) | 1D diffusion / heat equation, framed as lithium-ion battery charging |

**Public** ([`datasets/`](datasets/)) - real, openly available datasets,
grouped by country and each documented with its source and license.

| Country | Dataset | Source |
|---|---|---|
| 🇨🇱 `ch` | [`sii_cadastre`](datasets/ch/sii_cadastre/) | SII - assessed fiscal value (avalúo fiscal) per property |
| 🇨🇱 `ch` | [`central_bank_hpi`](datasets/ch/central_bank_hpi/) | Central Bank of Chile - quarterly housing price index (IPV) |
| 🇺🇸 `us` | [`nyc_taxi`](datasets/us/nyc_taxi/) | NYC Taxi & Limousine Commission - one row per taxi trip, monthly since 2009 |
| 🇩🇪 `de` | [`opsd_solar`](datasets/de/opsd_solar/) | Open Power System Data - hourly German solar generation as measured by the four TSOs |

## Models

| Model | What it is |
|---|---|
| [`kan`](models/kan/) | Kolmogorov-Arnold Network - learnable spline functions on every edge instead of fixed activations |
| [`mlp`](models/mlp/) | Standard feedforward baseline |

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate   # or source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Then `cd` into an experiment folder and run its `run.py` - see that
experiment's own `README.md` for expected runtime and what it produces.
