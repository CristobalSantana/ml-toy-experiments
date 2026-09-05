# BTC RSI divergence: pre-registered criteria

**Pre-registration date: 2026-08-14.** Written before any backtest was run and
frozen at that point. **No result may change this file.** If reality forces a
deviation, it is recorded in `README.md` under "Deviations", never by editing
here. The whole point is that the success conditions below are fixed before a
single equity curve is seen.

## The question

Does RSI divergence, one of the most widely taught patterns in technical
analysis, produce a better risk-adjusted return than simply buying and holding
BTC - after realistic trading costs?

This is a hypothesis to be **tested and reported in whichever direction it
falls**, including the likely outcome that it does not.

> This is a backtest, not financial advice, and it does not predict future
> returns. A strategy that worked on 2017-2026 BTC has no guarantee of working
> on anything else, including future BTC.

## Data

- **BTC/USDT daily candles from Binance** (public REST API, no authentication),
  from 2017-08-17 (the first day Binance lists the pair) to the retrieval date.
- Cached to disk on first download; a rerun never re-hits the API.
- Retrieval date and row count recorded in a committed manifest.

## The strategy under test

**Long only.** No shorts, no leverage, no funding-rate assumptions.

- **Entry**: a confirmed *regular bullish* RSI divergence - price makes a lower
  low while RSI makes a higher low.
- **Exit**: a confirmed *regular bearish* divergence (price higher high, RSI
  lower high), or a stop/timeout if one is defined in the config.
- Position is all-in or flat; no sizing rules, since sizing would add a second
  set of free parameters and blur what is being tested.

## Costs, fixed in advance

- **Fee: 0.10% per side** (Binance spot taker).
- **Slippage: 0.05% per side.**
- Total **0.30% per round trip**. Applied to every trade. A backtest without
  costs is not a backtest.

## Execution rules - the leakage that matters here

A divergence is built on *pivots*, and a pivot at bar `t` is only knowable
`N` bars later, once `N` subsequent bars have failed to exceed it. Detecting
pivots with a centred window and then trading them at `t` is look-ahead bias,
and it is the single most common way a divergence backtest produces returns
that cannot be earned.

1. **A pivot is confirmed only at `t + N`**, where `N` is the right-hand window.
   No signal may reference a bar later than the decision bar.
2. **Signals execute on the NEXT bar's open**, never on the close of the bar
   that produced them.
3. These are asserted in code and abort the run if violated.

## Metrics

Reported for the strategy **and** for buy-and-hold over the identical period:

- **Sharpe ratio** (annualised, daily returns, zero risk-free rate) - primary.
- Total return and CAGR.
- **Maximum drawdown**.
- Number of trades, hit rate, average holding period.
- Exposure: the fraction of days the strategy is in the market. A strategy that
  is flat 80% of the time is not comparable to buy-and-hold without saying so.

## Parameter grid, and the anti-cherry-picking rule

Divergence has free parameters. The grid tested, fixed now:

- RSI period: **7, 14, 21**
- Pivot window `N` (bars either side): **3, 5, 8**
- Maximum bars between the two pivots forming a divergence: **30, 60**

That is 18 combinations. **The headline result is the median across the whole
grid, not the best cell.** Reporting the best of 18 is how a strategy that does
nothing is made to look profitable; the full grid is reported so the dispersion
is visible.

## Held-out period - not touched until the end

- **Development: 2017-08-17 to 2023-12-31.**
- **Held out: 2024-01-01 to the retrieval date.** Not used for parameter
  choice, for debugging, or for any decision. Evaluated once, at the end.

## Decision rules, declared before any result

Let `S_strat` be the **median Sharpe across the 18-cell grid** on the held-out
period, `S_bh` buy-and-hold's Sharpe on the same period, and `IQR` the
interquartile range of grid Sharpes.

- **The strategy works** if `S_strat > S_bh` **and** the grid's 25th percentile
  also exceeds `S_bh`. That is: it beats buy-and-hold, and not only for a
  lucky corner of the parameter space.
- **The strategy fails** if `S_strat <= S_bh`. This is the expected outcome and
  will be reported as plainly as the alternative.
- **Inconclusive** if `S_strat > S_bh` but the 25th percentile falls below it -
  meaning the result depends on which parameters you happened to pick, which is
  itself the finding.

A secondary check, reported either way: whether the strategy's **maximum
drawdown** is smaller than buy-and-hold's. A strategy can lose on return and
still be worth something if it dramatically reduces drawdown, and that would be
dishonest to hide.

## What this file does not fix

Implementation details free to evolve: plotting, how the data is cached, code
structure. Frozen: the strategy definition, the cost assumptions, the execution
and look-ahead rules, the metrics, the parameter grid, the held-out period, and
the decision rules above.
