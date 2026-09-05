# Addendum: the 1-hour arm

**Pre-registered 2026-08-14, after the daily result was known.**

[`CRITERIA.md`](CRITERIA.md) was frozen for the daily experiment and is not
edited. This is a separate, later pre-registration for a second timeframe, and
it says so plainly because the order matters.

## What was already known when this was written

The daily arm **failed**: median grid Sharpe on the held-out period was −0.314
against buy-and-hold's 0.557, and none of the 18 parameter cells beat
buy-and-hold in both periods.

## The multiple-testing problem this creates

Running a second timeframe after the first one failed is exactly the behaviour
the daily experiment set out to expose: try enough variants, report the one
that worked. Testing two timeframes is **two tests**, and if the hourly arm
"works" while the daily one did not, the honest reading is not "it works on
1h" but "one of two tried timeframes looked good", which is a much weaker
claim.

So this is recorded now, in advance:

- This is **test 2 of 2** on the same pattern and the same asset.
- If the hourly arm succeeds and the daily arm failed, the write-up reports
  **both**, states that two timeframes were tried, and treats a single success
  out of two as weak evidence rather than a finding.
- A third timeframe, if ever run, is likewise disclosed.

## What changes from the daily arm

Only the data and the annualisation:

- **BTC/USDT 1-hour candles**, same source and span (2017-08-17 onward,
  78,712 bars).
- **Sharpe annualised by √8760** rather than √365. Leaving the daily constant
  in place would inflate the hourly ratio by √24 and make the two arms
  incomparable.
- Costs, execution rules, look-ahead rules, metrics, the 18-cell grid and the
  held-out boundary (2024-01-01) are **unchanged**.

The grid parameters are deliberately left at the same bar counts, so
`pivot_window = 5` now means five hours rather than five days. Re-tuning the
grid for the new timeframe would add a degree of freedom and make the two arms
answer different questions.

## Expectation, stated before running

Costs bite far harder here. The same pattern on hourly bars fires many more
times, and each round trip costs 0.30%. A strategy trading ten times as often
must be about ten times better before costs to break even against the daily
version. The expectation is therefore **failure by a wider margin**, but the
outcome is reported whichever way it falls.

## Decision rules

Identical to `CRITERIA.md`, applied to the hourly held-out period: the median
Sharpe across the 18-cell grid must beat buy-and-hold, **and** so must the 25th
percentile, or the strategy has not worked.
