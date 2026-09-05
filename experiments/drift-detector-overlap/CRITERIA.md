# Pre-registration: where the classifier two-sample test stops working

**Frozen 2026-09-05, before any detector was run.**

Everything below is decided in advance. Results go in `README.md`; if anything
here turns out to be wrong or unworkable, the deviation is recorded there and
this file is **not edited**.

## The question

A classifier two-sample test (Lopez-Paz and Oquab, 2017) detects drift by
training a model to tell a reference sample from a current one. If it does no
better than chance the two are indistinguishable; if it separates them
cleanly, they differ.

It is known to work. The question is **where it stops working, and whether
that boundary can be predicted before deploying it.**

Three things get measured:

1. **Synthetic.** How detector quality falls as the two distributions overlap
   more, with the overlap controlled exactly.
2. **Real.** Whether the synthetic curve predicts what happens on NYC taxi
   data, where the drift is real, dated and not of our choosing.
3. **Sample size.** Whether a hypothesis test's verdict is driven by the size
   of the window rather than by the size of the shift.

## Why this pairing

The synthetic arm alone is a demonstration: distributions built to have a
known overlap will behave the way the mathematics says. The real arm alone is
an anecdote: two months differ, and there is no ground truth for how much.

Run together, the synthetic curve becomes a **prediction** that the real data
can falsify. That is the only part of this worth calling a finding.

---

## Data

### Synthetic

Gaussian samples with the overlapping coefficient set analytically. For two
1-D Gaussians of equal variance separated by δ,

    OVL = 2 · Φ(−δ / 2σ)     so     δ = −2σ · Φ⁻¹(OVL / 2)

which lets the overlap be dialled in exactly rather than searched for.

- **Primary sweep**: 1 feature, OVL from 0.05 to 1.00 in steps of 0.05.
- **Dilution sweep**: 8 features, the shift applied to one of them, the other
  7 identical noise. The marginal OVL of the shifted feature is the same as
  the primary sweep, so the two curves are directly comparable and the gap
  between them is the cost of having to find the signal.

### Real: NYC yellow taxi trips

Source: NYC Taxi & Limousine Commission, monthly parquet, public and
unauthenticated. Documented in `datasets/us/nyc_taxi/`.

Five pairs, **fixed now**, chosen for documented reasons and not for the
results they give:

| # | Pair | Why this pair | Expected drift |
|---|---|---|---|
| 0 | 2024-06 vs itself, random split | a true null: same distribution by construction | none |
| 1 | 2024-05 vs 2024-06 | adjacent months, same season | smallest |
| 2 | 2023-06 vs 2024-06 | one year apart, same month, so season is held fixed | small |
| 3 | 2020-03 vs 2020-04 | the pandemic shutdown, the sharpest dated shift in the series | large |
| 4 | 2019-06 vs 2024-06 | pre-pandemic against now | largest |

**Pair 0 is the control that makes the rest interpretable.** A detector that
flags drift between two random halves of a single month is broken, and no
result from pairs 1-4 would mean anything.

The a priori ordering of expected drift is **1 < 2 < 3 ≈ 4**, and it is
recorded here so that a mismatch counts against the method rather than being
explained away afterwards.

### Features

Seven, present in every month in the range:

`trip_distance`, `fare_amount`, `tip_amount`, `total_amount`,
`trip_duration_min` (derived: dropoff − pickup), `passenger_count`,
`pickup_hour` (derived).

Deliberately excluded: `congestion_surcharge` and `airport_fee`, which do not
exist across the whole range. Their absence is a finding in its own right and
is recorded below, not silently patched.

### Cleaning, fixed in advance

Verified as necessary on 2024-06 before freezing this file - the counts are
from that month, out of 3,539,193 rows:

| Rule | Why | Rows affected |
|---|---|---|
| pickup month must equal the file's nominal month | the 2019-06 file contains trips dated 2001; the 2024-06 file contains one dated 2026-06-26 | 51 (0.001%) |
| `0 < trip_distance ≤ 50` miles | 52,954 trips of zero distance | 53,574 |
| `0 < fare_amount ≤ 300` | 63,043 non-positive fares | 63,424 |
| `0 < trip_duration_min ≤ 180` | negative durations exist | 3,368 |
| `1 ≤ passenger_count ≤ 6` | 35,839 zeros and 410,781 nulls | ~447,000 |

Cleaning is applied identically to both sides of every pair. **The rules are
frozen now** because cleaning thresholds chosen after seeing detector output
are a way to tune the answer.

### The coverage caveat, stated before it can be spun

Yellow-taxi volume fell from 6.97 M trips in 2019-06 to 3.54 M in 2024-06.
**A large part of that is not a change in how New Yorkers travel; it is a
change in what this file covers**, since high-volume for-hire trips (Uber,
Lyft) are published in separate files.

This experiment therefore says nothing about trip *volume*. It compares only
the distribution of features **within** the trips the file does contain. Any
finding phrased as "taxi demand shifted" would be this caveat being ignored.

---

## The detector

Classifier two-sample test, fixed configuration:

- **Classifier**: `sklearn.ensemble.HistGradientBoostingClassifier`,
  `max_iter=100`, `max_depth=6`, `learning_rate=0.1`, `random_state` from the
  trial seed. No tuning, on either arm.
- **Protocol**: pool the two samples, label 0/1, stratified 50/50 train/test
  split. Train on the training half, evaluate on the held-out half only.
- **Statistic**: held-out accuracy.
- **Decision**: drift is declared when accuracy is significantly above 0.5
  under a one-sided binomial test at **α = 0.05**.

Evaluating on the rows the classifier trained on is this experiment's version
of look-ahead: it would report near-perfect separation for two identical
samples. The held-out split is not optional and `test_detector.py` will assert
that a same-distribution pair is not flagged.

### Sample sizes

The detector's power depends on the window, so the window is a variable rather
than a detail. Every synthetic sweep is run at **n ∈ {500, 5000, 50000}** rows
per side. The real pairs use **n = 50,000** per side, subsampled with a fixed
seed so that all five pairs are compared at identical n.

### Headline metric

For each (OVL, n) cell, **100 trials**: 50 where the two samples genuinely
differ at that overlap, 50 where they are drawn from the same distribution.
The detector makes a binary call on each. The reported number is the
**Matthews correlation coefficient** of those 100 decisions against the truth.

MCC rather than accuracy, because the trials are balanced by construction but
a detector that always says "drift" would still score 0.5 accuracy, and MCC
sends that to 0.

### Overlap on real data

The analytic formula does not apply, so the empirical overlapping coefficient
is used: both samples histogrammed on a shared range with **100 bins**, and
OVL is the area under the pointwise minimum of the two densities. Reported per
feature, and as the mean across the seven features for the summary ordering.

---

## Pre-registered predictions

Written now so that "as expected" cannot be decided later.

- **P1.** Detector MCC falls as OVL rises. Spearman correlation between OVL
  and MCC ≤ −0.9 on the primary sweep.
- **P2.** At OVL = 1.00 the detector's false-alarm rate lies in [0.01, 0.10],
  consistent with α = 0.05. Outside that range the detector is miscalibrated
  and every other number here is suspect.
- **P3.** Larger n moves the curve right: at any fixed MCC threshold, the OVL
  at which it is reached is higher for n = 50,000 than for n = 500.
- **P4.** The 8-feature dilution sweep sits **below** the 1-feature sweep at
  the same marginal OVL. Nuisance dimensions make detection harder.
- **P5.** On real data, pair 0 (the null) is **not** flagged, and the ordering
  of pairs 1-4 by detector MCC matches their ordering by (1 − mean OVL).
- **P6.** Sweeping n from 1,000 to 1,000,000 on pair 1, the number of the
  seven features flagged by a per-feature KS test at p < 0.05 increases
  monotonically, and at n = 1,000,000 it reaches 7 of 7 - not because the
  drift grew, but because the window did.

## What counts as a failure

- **P2 failing** invalidates the whole experiment; a detector that cannot hold
  its own false-alarm rate is not measuring anything.
- **P5 failing** would be the interesting outcome: it would mean the synthetic
  curve does not transfer to real data, and the synthetic experiment the
  article promised is not evidence about production drift.
- P1, P3, P4 and P6 failing individually would each be reported as a negative
  result on that specific claim, not as a broken experiment.

## Known and recorded in advance

**Schema drift is already present in this dataset.** The airport fee column is
named `airport_fee` in the 2019 file and `Airport_fee` in the 2024 file - the
same field, differing only in capitalisation. A pipeline that concatenates the
two produces two columns, each roughly half null, with no error raised.

The loader normalises column names to lower case. That normalisation is the
fix a production pipeline needs, and the fact that it is needed at all is the
appendix category the article calls schema drift: an engineering failure
rather than a statistical one, and the most frequent cause of sudden
degradation.

## Reproducibility

- Monthly parquet files are cached under `data/` with a `sha256` per file in a
  manifest, so a rerun uses the same bytes or says loudly that it cannot.
- Every subsample, train/test split and synthetic draw takes its seed from a
  single `SEED = 20260905` in `config.yaml`.
- `run_all.py` is the entry point and runs the detector self-checks before
  anything else. If the null pair is flagged, the run stops.

## Deviations

Recorded in `README.md` as they happen. This file is not edited after this
point, including after the results are known.
