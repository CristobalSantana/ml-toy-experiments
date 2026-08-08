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
| 🇨🇱 `ch` | [`bcch_ipv`](datasets/ch/bcch_ipv/) | Banco Central de Chile - quarterly housing price index (IPV) |

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
