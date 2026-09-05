# Pre-registration: how much of a solar forecast is just the sunrise?

**Frozen 2026-09-05, before any model was fitted.**

Decisions below are made in advance. Results go in `README.md`; anything that
turns out wrong is recorded there as a deviation, and this file is **not
edited**.

## The question

Solar generation forecasting papers report R² above 0.90 routinely. The
question is what that number is made of.

German solar output is measured hourly. In the five years of it used here,
**54.8% of hours have the sun below 5 degrees of elevation, and 43.4% have
output of exactly zero.** A model that predicts zero at night and something
reasonable at noon will score very well on any metric computed over all hours,
without containing a single piece of forecast information.

So: **how much of a reported solar-forecast R² survives once the sun's
position is accounted for?** The sun's position at a given place and instant
has been computable for centuries. Nothing about predicting it is forecasting.

## Data

`datasets/de/opsd_solar/`. Measured hourly generation for Germany from the
four transmission system operators, published by Open Power System Data,
release 2020-10-06. Not a simulation - predicting a simulator is a different
and much easier problem.

- **43,696 hours**, 2015-01-01 to 2019-12-30.
- Target is the **capacity factor**: generation divided by installed capacity.
  Capacity grew 36% over the period (37.2 to 50.5 GW), and a model fitted to
  raw megawatts would be rewarded for learning that trend, which is not
  forecasting.
- **Solar elevation** is computed from the NOAA algorithm at Germany's
  centroid, and agrees with PVGIS's independent implementation to 0.13 degrees
  mean absolute error over a year.

### Splits, fixed now

| | Period | Use |
|---|---|---|
| Train | 2015-01-01 → 2017-12-31 | fitting |
| Validation | 2018-01-01 → 2018-12-31 | any choice that has to be made |
| **Test** | **2019-01-01 → 2019-12-30** | **touched once, at the end** |

Chronological, never shuffled. A random split of an hourly series puts the
hour before and the hour after the same target in the training set, which
turns forecasting into interpolation and inflates every number reported here.

## Forecasts

Two horizons: **1 hour ahead** and **24 hours ahead**. Everything is fitted
and evaluated separately for each.

At forecast time for target hour *t+h*, a model may use observations up to and
including hour *t*, plus the solar geometry of *t+h*, which is known exactly
in advance. Nothing else.

### The baselines are the point

| | Name | What it does | Forecast information |
|---|---|---|---|
| **B0** | training mean | predicts the mean capacity factor of the training set, always | none |
| **B1** | climatology | the average capacity factor for that (day-of-year, hour) in the training years | **none** |
| **B2** | smart persistence | carries the clear-sky index forward from hour *t* and rescales it to the clear-sky level of *t+h* | the last observation |

**B1 is the load-bearing one.** It is identical every year, cannot react to
weather, and knows nothing whatever about the day it is predicting. If it
scores well, the metric is measuring the calendar.

**B2 is the honest benchmark.** Smart persistence is what the solar
forecasting literature actually compares against, and it is the denominator of
every skill score reported here.

Clear-sky output is estimated from the training set alone as the 90th
percentile of capacity factor within each 1-degree elevation bin. Empirical
rather than a physical model, so it needs no site parameters, and estimated on
training data only so the test year cannot leak into it.

### Models

Same features for all three, no per-model tuning:

lags of capacity factor at *t*, *t−1*, *t−2*, *t−24*; the clear-sky index at
those lags; the elevation and clear-sky output at *t+h*; hour of day; day of
year encoded as sine and cosine.

| | Model | Configuration |
|---|---|---|
| **M1** | ridge regression | `alpha = 1.0`, features standardised |
| **M2** | gradient boosting | `HistGradientBoostingRegressor`, `max_iter=300`, `max_depth=6`, `learning_rate=0.06`, `early_stopping=False` |
| **M3** | echo state network | 400 units, spectral radius 0.9, leak 0.3, input scale 0.5, ridge readout `alpha = 1.0`, 200 hours of washout |

The echo state network is here because it is the cheap end of the recurrent
family: the reservoir is random and fixed, and only the linear readout is
fitted. If a randomly wired recurrent network matches a tuned gradient
booster, the interesting quantity is what the booster cost to get there.

`early_stopping=False` on the booster is set explicitly. The sklearn default
turns it on above 10,000 samples, which would make the model quietly different
between horizons and splits of different size.

## Metrics

- **R² over all hours** - the number the literature reports.
- **R² over daylight hours only** (elevation > 5 degrees) - the same number
  with the free part removed.
- **nRMSE**, normalised by installed capacity.
- **Skill against B2**: `1 - MSE_model / MSE_B2`. Zero means no better than
  carrying the last observation forward; negative means worse.

## Pre-registered predictions

- **P1.** B1 - climatology, containing no forecast information at all -
  achieves **R² ≥ 0.85 over all hours** on the test year, at both horizons.
- **P2.** Restricting to daylight hours drops **every** model's R² by at least
  **0.15**.
- **P3.** At h = 1, **no model exceeds a skill of 0.35** against smart
  persistence.
- **P4.** At h = 24, smart persistence has **negative skill against
  climatology** - carrying yesterday's weather forward a full day is worse
  than knowing nothing but the calendar.
- **P5.** The echo state network and the gradient booster differ by less than
  **0.10 in daylight R²** at h = 1.

## What counts as a failure

- **P1 failing** would mean the premise is wrong and the metric is not as
  inflated as claimed. That is the outcome that would make this experiment
  uninteresting, and it is reported as plainly as the alternative.
- **P3 failing** would mean the models genuinely beat persistence by a wide
  margin, which would be a positive result worth more than the framing.
- P2, P4 and P5 failing individually are reported as negatives on those
  specific claims.

## Known in advance, and not to be spun

**A national aggregate has no single location.** German panels run from 47.3
to 55.1 degrees north. Elevation is computed at the centroid, so the twilight
hours are wrong by up to about twenty minutes at the extremes. This blurs the
day/night boundary slightly and cannot manufacture the effect being measured,
which is about the middle of the day as much as the edges.

**Aggregate output is smoother than a single site.** Clouds over Bavaria do
not cover Schleswig-Holstein, so a national total averages away much of the
variance that makes single-site forecasting hard. Skill scores here will be
higher than a rooftop installation would see, not lower.

**Capacity is reported with a lag**, so a few hours have a capacity factor
above 1. They are clipped to 1, and the count is reported.

**This says nothing about forecasting other countries or other years.** One
country, five years, one aggregation level.

## Reproducibility

`SEED = 20260905` for the reservoir's random weights and every other draw.
`run_all.py` checks `config.yaml` against the frozen values before running,
and runs the leakage checks first.

## Deviations

Recorded in `README.md` as they happen. This file is not edited after this
point, including after the results are known.
