# Pre-registration

Frozen 2026-09-06, before any strategy was backtested and before any
surrogate series was generated. Not edited afterwards. Deviations are
recorded in `README.md`.

## Read this first

This is a study of method, not trading advice. It searches a large number of
mechanical trading rules **in order to show what such a search produces even
when there is nothing to find.** Nothing here is a recommendation, a positive
result would not make one, and every user is responsible for their own
trading decisions.

## The question

[`btc-01-rsi-divergence`](../btc-01-rsi-divergence/) tested 18 parameter sets
of a well-known chart pattern. The best of them beat buy-and-hold on the
development period and then **lost 33% over a holdout period in which BTC
rose 49%.**

That is the standard shape of a backtesting failure, and the standard
explanation is "overfitting". This measures the explanation instead of
asserting it:

> **If you search N mechanical strategies on data in which no strategy can
> possibly have an edge, how good does the best one look?**

If the answer is "about as good as the best one looked on real Bitcoin", then
the search explains the result and no market structure needs to be invoked.

## Data, split and costs

All identical to `btc-01`, so the two experiments are directly comparable:

- Binance `BTCUSDT` daily bars, cached and checksummed, 2017-08-17 onwards.
- **Development** to 2023-12-31; **holdout** from 2024-01-01.
- 0.10% fee plus 0.05% slippage per side, charged on every position change.
- Signals act on the next bar's open; no same-bar execution.

Buy-and-hold over the development period returned **890.7%** and over the
holdout **49.1%**. Those two numbers are the reference for everything below,
and reproducing them is the implementation control.

## The strategy family

Moving-average crossover, long or flat: hold when the fast average is above
the slow one, otherwise hold nothing. It is the most-tried rule in technical
analysis and it has exactly two parameters, which makes the size of the
search transparent.

- fast: every integer from 2 to 40
- slow: every multiple of 5 from 5 to 200
- keep every pair with `slow > fast`

That is roughly **1,500 strategies** - a plausible afternoon of parameter
tuning, and small compared with what a grid search or a genetic optimiser
would try.

## The nulls

Surrogate price series in which **no strategy can have an edge by
construction**, because any exploitable time structure has been destroyed:

- **`shuffle`** - the daily log returns, randomly permuted. The set of
  returns is unchanged, so buy-and-hold's total return is *identical* on
  every surrogate; only their order differs. This is the pre-registered null.
- **`block`** - a moving-block bootstrap with 20-day blocks, which preserves
  short-range autocorrelation and volatility clustering. Reported alongside,
  as a harder null.

200 surrogates of each. The same ~1,500 strategies are run on every one.

## Predictions

- **P1** The backtester in this experiment reproduces `btc-01`'s buy-and-hold
  total return on both periods to four decimal places. Control: if it does
  not, the two experiments are not comparable and nothing else matters.
- **P2** On `shuffle` surrogates, where no edge exists, the best of ~1,500
  strategies still ends the development period with **at least double
  buy-and-hold's final equity**, in at least 90% of surrogates.
- **P3** On real data, the best strategy's development excess over
  buy-and-hold is **not above the 95th percentile** of the `shuffle` null.
  That is: the search explains the result without needing any real edge.
- **P4** The development-best strategy's holdout return is **below the median
  holdout return** of all strategies searched - the winner's curse.
- **P5** **Fewer than half** the strategies that beat buy-and-hold in
  development also beat it in the holdout.
- **P6** The *count* of strategies beating buy-and-hold in development on
  real data falls inside the central 90% of that count's `shuffle` null.

## What would overturn the story

**P3 failing** - the real best strategy landing beyond the null's 95th
percentile - would be evidence that the crossover family captures something
real in Bitcoin that shuffling destroys. That is the interesting outcome
available here and it would be reported as such, with the warning that a
single test at one threshold on one asset is weak evidence.

## Known in advance

**One family, one asset, one bar size.** Crossovers on daily BTC. Nothing
here generalises to other rules, other markets, or intraday data.

**Long or flat, never short.** Bitcoin rose enormously over the development
period, which makes buy-and-hold a hard benchmark and makes any time spent
out of the market expensive. That is deliberate: it is the benchmark a real
holder faces.

**The nulls destroy structure that may be real.** Volatility clustering and
momentum exist in real prices. `shuffle` removes both; `block` keeps the
first. If a strategy family exploits genuine momentum, `shuffle` will
understate what is achievable - which is why `block` is reported too.

**Surrogates preserve the drift.** Shuffled returns have the same sum, so
every surrogate has buy-and-hold's exact total return. The comparison is
therefore about *timing*, not about whether the asset went up.
