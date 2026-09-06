# Pre-registration: a distribution-free guarantee, in a world that drifts

**Frozen 2026-09-05, before any interval was built.**

Written in advance. Results go in `README.md`; deviations are recorded there
and this file is **not edited**.

## The question

Split conformal prediction turns any point predictor into an interval
predictor with a guarantee that is unusually strong for machine learning:

> for a chosen α, the interval covers the true value with probability at least
> 1 − α, **whatever the data distribution and whatever the model**.

No Gaussian assumption, no well-specified model, no asymptotics. It is exact
in finite samples. The guarantee is genuinely remarkable, and the method has
spread quickly because of it.

It rests on one condition: **exchangeability.** The calibration data and the
data being predicted have to be drawn from the same distribution, in a way
that makes their order irrelevant. Production data is not exchangeable with
last quarter's calibration set, and everyone knows it.

**So how fast does the guarantee actually decay?** That is the number nobody
quotes, and it decides whether conformal intervals are a safety property or a
decoration.

## Why this pairs with the drift experiment

[`drift-detector-overlap`](../drift-detector-overlap/) already measured how
far apart five pairs of NYC taxi months are, on the same cleaned features,
using the same overlapping coefficient. This experiment reuses those months
and that measurement, so degradation can be plotted **against a measured
distance rather than against a label like "far".**

| Month | Mean overlap with 2024-06 | Why it is in the set |
|---|---|---|
| 2024-06 | 1.000 (itself) | in-distribution control |
| 2024-05 | 0.986 | adjacent |
| 2023-06 | 0.982 | one year, same season |
| 2020-04 | - | the pandemic month, 238k trips |
| 2019-06 | 0.866 | pre-pandemic |

## The task

Predict **`fare_amount`** for a NYC yellow taxi trip, and put an interval
around it.

Regression rather than classification because the guarantee is about coverage
of a real value, and because the failure mode being looked for - intervals
that stay the same width while the world moves under them - is only visible
with a width to look at.

Features: `trip_distance`, `trip_duration_min`, `passenger_count`,
`pickup_hour`, `pu_location`, `do_location`.

`tip_amount` and `total_amount` are excluded, as in
[`hdc-vs-boosting`](../hdc-vs-boosting/): `total_amount` contains the fare
being predicted, and `tip_amount` is contaminated by how the trip was paid.

Cleaning is the same set of rules frozen in
[`drift-detector-overlap/CRITERIA.md`](../drift-detector-overlap/CRITERIA.md),
so all three experiments describe the same population of trips.

## Method

**Split conformal, absolute-residual score.**

1. Split 2024-06 into `fit` (100,000 trips), `calibrate` (50,000) and
   `eval` (50,000), disjoint and drawn once.
2. Fit a gradient-boosted regressor on `fit`.
3. On `calibrate`, compute the residuals `|y − ŷ|`.
4. The interval half-width is the `⌈(n+1)(1−α)⌉ / n` empirical quantile of
   those residuals. This is the finite-sample correction that makes the
   guarantee exact rather than asymptotic, and dropping it is the usual way
   the method gets implemented slightly wrong.
5. Apply that **same half-width** to every other month.

**α = 0.1**, so the target is 90% coverage. Fixed now.

The model is fitted once, on 2024-06, and never refitted. That is the point:
the question is what happens to a calibrated system as the world moves away
from it, not what happens if you keep retraining.

### The remedy arm

A second calibration is computed from a **small sample of the test month
itself** - 2,000 trips - and the same evaluation is repeated. This is what a
practitioner would actually do if they could label a little fresh data, and
the question is how little it takes.

## Metrics

- **Empirical coverage**: the share of eval trips whose true fare falls inside
  the interval. The guarantee says ≥ 0.90.
- **Mean interval width**, in dollars. A method can always buy coverage by
  widening; reporting width alongside coverage is what stops that.
- **Coverage gap**: `0.90 − coverage`. Positive means the guarantee is being
  violated.

## Pre-registered predictions

- **P1.** On held-out data from the calibration month itself, coverage is
  within **1 point of 90%**. If this fails the implementation is wrong and
  nothing else matters.
- **P2.** Coverage falls monotonically as the overlap with the calibration
  month falls, across 2024-05, 2023-06 and 2019-06.
- **P3.** On 2019-06 - the most distant month - coverage drops **below 85%**.
- **P4.** The **interval width does not change** across months, by
  construction, while coverage does. The failure is silent: nothing in the
  output of the method signals that the guarantee has stopped holding.
- **P5.** Recalibrating on **2,000 trips** from the test month restores
  coverage to within **1 point of 90%** on every month.

## What counts as a failure

- **P1 failing** invalidates everything else.
- **P3 failing** would mean the guarantee is far more robust to this much
  drift than expected, which would be a useful positive result and would
  argue *for* the method rather than against it.
- P2, P4 and P5 failing individually are reported as negatives on those
  claims.

## Known in advance, and not to be spun

**Under-coverage is not the only failure.** A calibration set drawn from a
harder month would produce intervals that are too *wide* elsewhere - coverage
above 90% at the cost of being useless. Both directions are reported; only one
of them is usually discussed.

**2020-04 is a small month.** The pandemic left 238,073 trips before cleaning,
against 3.5 million in a normal month, so its evaluation sample is smaller and
its coverage estimate noisier. The sample size is reported next to every
coverage figure rather than left implicit.

**One score function.** Absolute residual is the simplest conformal score.
Normalised and quantile-regression variants adapt their width to the input and
would behave differently under drift - probably better. A negative result here
is about the plain version, which is also the version most people implement
first.

**Coverage is marginal, not conditional.** Even when the overall rate is 90%,
particular kinds of trip can be covered far less often. This experiment does
not open that up, and the distinction is worth keeping in mind before reading
a passing coverage number as safety.

**The fare distribution itself changed.** Fares rose about 46% between 2019
and 2024. Part of the coverage loss is the model being wrong about the level,
not only the intervals being miscalibrated, and the two are not separated
here.

## Reproducibility

`SEED = 20260905`. `run_all.py` checks `config.yaml` against these values, and
the coverage of the in-distribution control is asserted before the other
months are scored.

## Deviations

Recorded in `README.md`. This file is not edited after this point.
