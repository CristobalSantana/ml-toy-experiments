# Experiment 02 - Plusvalía: pre-registered criteria

**Pre-registration date: 2026-08-08.** This file is frozen at the commit that
introduces it. **No result may change it.** If reality forces a deviation
(an endpoint dies, a model cannot be fit, a metric is ill-defined), the
deviation is recorded in `README.md` under a "Deviations from
pre-registration" heading, with its reason - it is *never* fixed by editing
this file. The whole point is that the success conditions below are written
before any number is seen.

## Thesis under test

On tabular data, classical methods (gradient-boosted trees above all) remain
competitive-to-superior, and the decisive comparison is **error against
cost**, not error alone. TabPFN v2 (Hollmann et al., *Nature* 2025) is the
modern counterexample we test against, not a foregone conclusion.
References: Grinsztajn et al. (2022); Shwartz-Ziv & Armon (2022).

This is a hypothesis to be **tested and reported in whichever direction it
falls**, including the outcomes that contradict the thesis.

## Two arms

- **Cross-sectional arm (primary).** Model assessed fiscal value per square
  metre at block level, from the SII cadastre, one reavalúo snapshot. This is
  where the six models are compared for the error-vs-cost thesis.
- **Temporal arm (secondary).** The Banco Central IPV quarterly series, used
  for the drift analysis and the walk-forward evaluation. Not a per-property
  model; a series/panel.

## Target variables and units

- **Primary target:** `log10(assessed_value_per_m2)` where
  `assessed_value_per_m2` is the SII **avalúo fiscal** divided by built area,
  **expressed in UF per m²** (deflated with the UF value at the avalúo's
  reference date). Property in Chile is quoted in UF; nominal pesos would
  inject a spurious inflationary trend that every model would learn as
  signal. Deflation to UF (or constant-date currency) is **mandatory**, not a
  modelling choice.
- **Secondary target:** IPV index level and its quarterly log-change, per zone
  and dwelling type.
- **Assessed value is not market price.** They are different quantities. The
  code and the README state this explicitly. We predict assessed value; we do
  **not** present it, or any model output, as an estimate of market price.

## Metrics

Reported for **every** model, always as **mean ± spread over 5 seeds**:

**Error (primary target, on the log10 scale unless noted):**
- **Primary:** MAE on `log10(value/m²)`.
- Secondary: R² on the log target; median absolute percentage error (MdAPE) on
  the back-transformed UF/m² value; error broken down by comuna.

**Cost (equally part of the deliverable):**
- Wall-clock training time (s).
- Peak resident memory during fit (MB).
- Inference wall-clock for the held-out set (s).
- Approximate compute: FLOPs estimate where derivable, otherwise measured
  energy (e.g. via a RAPL/`codecarbon`-style reading) if the machine exposes
  it; if neither is measurable, this is recorded as "not measurable on this
  machine" rather than faked.

The deliverable is the **accuracy-vs-cost frontier**. A table of error alone
does not satisfy this experiment.

## Decision rules (declared before any result)

Let `E_c` = 5-seed mean primary error (MAE on log target) of the **best
classical model**, where classical = {Ridge, Random Forest, XGBoost/LightGBM
at defaults, CatBoost, MLP}. Let `E_t` = 5-seed mean primary error of
**TabPFN v2**. Let `s` = the pooled spread (sum of the two 5-seed standard
deviations).

- **Classical methods win** if **either**:
  1. `E_c < E_t - s` (the best classical model is more accurate and the 5-seed
     distributions separate), **or**
  2. `E_c ≤ E_t + s` (accuracy statistically indistinguishable) **and** the
     best classical model reaches it at **≥ 5×** lower training wall-clock
     **or ≥ 5×** lower peak memory than TabPFN. Equal accuracy at a fraction
     of the cost is a win for the thesis.

- **TabPFN wins** if `E_t < E_c - s` (TabPFN more accurate, distributions
  separate) **within its documented regime**. If the dataset exceeds TabPFN's
  documented row/feature limits and we must subsample to fit it, the
  subsampling is disclosed and the win is labelled **"regime-limited"**: a win
  inside TabPFN's operating envelope, not a general one.

- **Inconclusive** if the distributions overlap (`|E_c - E_t| ≤ s`) and no
  ≥5× cost gap exists. Reported as inconclusive; not spun either way.

## Drift cutoff hypothesis (declared, to be tested - not assumed)

- **Hypothesis:** cutting the training window to **2023-01 → present** yields
  held-out error that is **equal to or better than** training on the full
  **2020-01 → present** window, because the 2020-2022 span is distributionally
  drifted (pandemic-era distortion plus a reavalúo-cycle change).
- **The a-priori cutoff is 2023-01.** It is chosen now, on domain reasoning,
  and does **not** move to whatever the detector happens to like.
- **Test procedure:** train every model twice - (a) full 2020-present, (b)
  2023-present only - and compare **both on the untouched held-out final
  period** (below). The winning window is the one with lower held-out primary
  error (mean ± spread).
- **Reported in both directions.** If discarding the 2020-2022 data makes
  held-out error *worse*, that is the more interesting result and it is
  reported as the headline, not buried.

**Drift detector thresholds (frozen reference window):**
- The reference window is the **first four quarters** of the analysis span and
  is **fixed and versioned - never rolling**.
- **Univariate:** per-feature Kolmogorov-Smirnov and Population Stability
  Index, with **Benjamini-Hochberg FDR control at 5%** across features (one
  test per feature at 5% would manufacture false alarms). A feature is
  "shifted" if BH-adjusted KS p < 0.05 **or** PSI ≥ 0.25.
- **Multivariate:** a classifier two-sample test (reference vs each later
  window). A window is **"drifted"** if the detector's ROC-AUC ≥ **0.70** with
  its 95% CI lower bound > 0.5. Report the AUC over time and the detector's
  feature importances.

## Held-out period - not touched until the very end

- **Temporal arm:** the **most recent 4 quarters** available at data-retrieval
  time are held out entirely. They are never used for training, tuning, model
  selection, drift-window choice, or feature engineering decisions - only for
  the single final evaluation.
- **Cross-sectional arm:** a frozen **10% of manzanas**, selected by a fixed
  seed at the start, is held out and untouched until the final run. The
  remaining 90% is used with grouped-by-manzana 5-fold × 5-seed CV for all
  development.

## Validation protocol

- **Cross-sectional:** grouped K-fold **by manzana**, **5 folds × 5 seeds**.
  Rows from the same manzana never span train and test.
- **Temporal:** **walk-forward** on the IPV series. **No random splits** on the
  time series, ever.
- Every reported number is a **5-seed mean and spread**. No single-run numbers
  appear as results.

## Leakage rules (checked before any model is fit; failures are loud)

1. **No feature may be a deterministic function of the target.** Because the
   target is value **per m²**, the features may **not** simultaneously contain
   total assessed value and area (from which the target is exactly
   recoverable). This is asserted in code and aborts the run if violated.
2. **Strict as-of cut** on every rolling/aggregate/lagged feature: no future
   period contributes to a row's features.
3. **No manzana split across train and test** (enforced by the grouped CV
   above).

## Reproducibility & data-handling commitments

- A single script reproduces everything from scratch under a **fixed seed**.
- Every dataset pull logs its **exact retrieval date and source URL**, and the
  reavalúo cycle for SII.
- SII / Banco Central responses are **cached to disk**; a rerun never re-hits
  the APIs. Rate limits and terms of use are respected (throttle + cache).
- Raw pulls and processed tables are gitignored; only a **committed checksum
  manifest** records what was retrieved.

## What this file does not fix

Implementation choices that do **not** touch the success conditions above are
free to evolve (exact hyperparameters within "reasonable defaults", plot
styling, which comunas are pulled, library versions). What is frozen is:
the target and units, the metrics, the win/loss decision rules, the drift
cutoff and detector thresholds, the held-out periods, the validation
protocol, and the leakage rules.
