# Pre-registration: what one pass of addition buys

**Frozen 2026-09-05, before any model was fitted on the real task.**

Written in advance. Results go in `README.md`; deviations are recorded there
and this file is **not edited**.

## The question

Hyperdimensional computing represents every feature and every value as a
random vector in a space wide enough that any two random vectors are nearly
orthogonal, and then does arithmetic on them. A record is the elementwise
product of feature and value vectors, summed. A class is the sum of its
records. Prediction is whichever class vector is closest.

There is no gradient, no loss function and no epoch. **Training is one pass
and consists of addition.**

Three claims are made for it, and all three are testable:

1. **It learns from very few examples**, because a class prototype is
   meaningful after one record.
2. **It tolerates corruption of its own representation.** Information is
   spread across every dimension, so losing some should degrade it gently
   rather than break it.
3. **It is cheap**, because addition is cheap.

Against a gradient-boosted tree on a real task, what do those claims buy, and
what do they cost?

## The task

**Was this taxi ride paid in cash?** Binary, from the NYC yellow taxi records
already cached by `datasets/us/nyc_taxi`.

- Train pool: **2024-05**. Test: **2024-06**. A forward split, so nothing from
  the test month can inform the fit.
- Roughly **15.1% of trips are cash**, so the classes are imbalanced and
  plain accuracy is close to useless: predicting "card" every time scores
  about 0.849.

**ROC AUC is the primary metric.** Balanced accuracy and plain accuracy are
reported alongside, to show what each one hides.

> **This metric choice was changed before the experiment ran, and the reason
> is on the record.** The first draft of this file made balanced accuracy
> primary. An exploratory fit - one gradient booster, no HDC, no learning
> curve - showed it is degenerate here: at 15% prevalence and a fixed 0.5
> threshold, a booster with AUC 0.65 still scores balanced accuracy 0.51,
> because it almost never predicts the minority class. Every method would have
> come out at 0.51 and the comparison would have measured the threshold rather
> than the models. The change was made after that probe and before any run of
> this experiment, and it is disclosed here rather than presented as the
> original plan.

### Features, and the one that had to be removed

Seven: `trip_distance`, `fare_amount`, `passenger_count`,
`trip_duration_min`, `pickup_hour`, `pu_location`, `do_location`.

**`tip_amount` is excluded, and this is the most important decision here.** It
is exactly zero for 100% of cash trips and non-zero for 94% of card trips,
because the TLC records card tips and cannot see cash ones. The label is
written into the feature by the collection process. `total_amount` inherits
the same contamination, being fare plus extras plus tip, and is excluded too.

This is leakage of a different family from the temporal kind: nothing here
reads the future, and a chronological split does not catch it. The only
defence is knowing how the data was collected. `test_leakage.py` includes the
leaky feature deliberately and reports what it would have bought.

## Configurations

| | Model | Configuration |
|---|---|---|
| **B** | majority class | always predicts card |
| **M1** | logistic regression | standardised inputs, `max_iter=1000` |
| **M2** | gradient boosting | `HistGradientBoostingClassifier`, `max_iter=200`, `max_depth=6`, `early_stopping=False` |
| **M3** | hyperdimensional | `dim=10000`, 64 quantisation levels, **one pass**, no corrective epochs |

Quantisation uses percentiles of the training split, not equal-width bins:
fare and distance are heavily right-skewed and equal-width bins would put most
of the data in the first two.

Level vectors are built by progressive bit-flipping so that neighbouring
levels stay similar and the extremes come out orthogonal. Independent random
vectors per level would discard the ordering of a continuous feature.

`retrain_epochs = 0`. A corrective pass over misclassified records is
iterative learning, and folding it into "one-shot HDC" would be answering a
different question than the one asked.

## The three arms

**A. Learning curve.** Training sizes 100, 300, 1,000, 3,000, 10,000, 30,000,
100,000, 300,000. Five seeds at each size; the reported figure is the median.
Test set is a fixed 200,000-trip sample of June, identical for every cell.

**B. Robustness.** Two separate tests, because the two methods are not
corruptible in the same way:

- *Input corruption*, applied to both: replace a fraction of feature values
  with draws from that feature's training distribution. Fractions 0, 0.1, 0.2,
  0.4.
- *Representation corruption*, HDC only: zero a fraction of hypervector
  dimensions at prediction time. Fractions 0, 0.2, 0.5, 0.8. Gradient boosting
  has no analogue, and inventing one would be a comparison of two different
  things.

**C. Cost.** Wall-clock fit time and the number of parameters that have to be
kept after training.

## Pre-registered predictions

- **P1.** Predicting the majority class reaches plain accuracy ≥ 0.84, AUC of
  exactly 0.50, and balanced accuracy of exactly 0.50 - three numbers for the
  same useless predictor, only two of which say so.
- **P2.** At the largest training size, gradient boosting beats HDC by at
  least **0.03 AUC**.
- **P3.** At 300 or fewer training examples, HDC is within **0.03 AUC** of
  gradient boosting.
- **P4.** Zeroing **50% of HDC's hypervector dimensions costs less than 0.02
  AUC**.
- **P5.** HDC does **not** train faster than gradient boosting at the largest
  size. The "addition is cheap" claim is about the operation, not the total,
  and a 10,000-dimensional encode per row is not free.

## What counts as a failure

- **P2 failing** would mean HDC is competitive on ranking quality, which would
  make it a serious option rather than a curiosity, and would be the most
  interesting outcome available.
- **P4 failing** would break the claim the method is best known for.
- P1, P3 and P5 failing individually are reported as negatives on those
  specific claims.

## Known in advance

**One task, one month pair, one encoding.** Record-based encoding is the
simplest of several; n-gram and permutation-based encodings exist and are not
tested. A negative result here is about this encoding on this task.

**The honest task is hard.** With `tip_amount` removed the features carry
little information: a gradient booster on 1.5 million rows reaches AUC 0.65.
That is real signal and well above chance, but nobody should read this as a
solved problem. A method that reaches 0.62 here is not failing badly.

**The classes are imbalanced and the minority class is not oversampled.**
Small training samples are drawn without stratifying, so a 100-row sample may
contain very few cash trips. That is deliberate - it is what a real deployment
faces - and it is why the small-n end of the learning curve is noisy and is
reported as a median over five seeds.

**Location IDs are treated as numbers.** They are categorical, and neither the
quantiser nor the tree is told so. This handicaps both methods equally and is
noted rather than fixed, since fixing it well is a separate piece of work.

**Timings are wall-clock on one machine**, single run per cell, and include
encoding for HDC. They are indicative, not benchmarks.

## Reproducibility

`SEED = 20260905`. `run_all.py` checks `config.yaml` against these values and
runs the leakage demonstration before anything else.

## Deviations

Recorded in `README.md`. This file is not edited after this point, including
after the results are known.
