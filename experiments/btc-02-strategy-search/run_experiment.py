"""
run_experiment.py -- Search ~1,400 strategies on Bitcoin, then search the
same ~1,400 on data where none of them can work.

    python run_experiment.py

  1. the real search, development and holdout   -> outputs/real.csv
  2. the same search on 200 shuffle surrogates
     and 200 block surrogates                   -> outputs/nulls.csv
  3. the comparison                             -> outputs/verdict.json

The control is fatal. If this backtester does not reproduce btc-01's
buy-and-hold return, the two experiments are not measuring the same thing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import search as S

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATA = HERE.parent / "btc-01-rsi-divergence" / "data" / "btcusdt_1d.csv"

BTC01_DEV = 8.907474          # btc-01/outputs/buy_and_hold.csv
BTC01_HOLDOUT = 0.490849


def excess(strategy: float, benchmark: float) -> float:
    """Final equity relative to buy-and-hold's, minus one.

    Ratios rather than differences: over a period in which the asset went up
    890%, a difference in percentage points is dominated by the asset and
    says almost nothing about the strategy.
    """
    return (1.0 + strategy) / (1.0 + benchmark) - 1.0


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])

    d = pd.read_csv(DATA, parse_dates=["date"])
    dev = d[d["date"] <= cfg["split"]["dev_end"]].reset_index(drop=True)
    hol = d[d["date"] >= cfg["split"]["holdout_start"]].reset_index(drop=True)
    do, dc = dev["open"].to_numpy(), dev["close"].to_numpy()
    ho, hc = hol["open"].to_numpy(), hol["close"].to_numpy()

    pairs = S.grid(cfg["grid"]["fast_max"], cfg["grid"]["slow_max"],
                   cfg["grid"]["slow_step"])
    print(f"data       {len(d):,} daily bars, "
          f"{d['date'].min().date()} to {d['date'].max().date()}")
    print(f"split      development {len(dev):,} bars, holdout {len(hol):,}")
    print(f"strategies {len(pairs):,} moving-average crossovers\n")

    # ---- P1: the control, before anything else --------------------------
    bh_dev = S.buy_and_hold(do, dc)
    bh_hol = S.buy_and_hold(ho, hc)
    print(f"control - buy-and-hold against btc-01's published figures")
    print(f"  development {bh_dev:.6f}  btc-01 {BTC01_DEV:.6f}")
    print(f"  holdout     {bh_hol:.6f}  btc-01 {BTC01_HOLDOUT:.6f}")
    if (abs(bh_dev - BTC01_DEV) > 5e-5 or abs(bh_hol - BTC01_HOLDOUT) > 5e-5):
        sys.exit("\nFAIL - this backtester does not reproduce btc-01's "
                 "buy-and-hold. The two experiments are not comparable and "
                 "nothing below would mean anything.")
    print("  OK - same data, same split, same number\n")

    # ---- the real search ------------------------------------------------
    rd = S.run_grid(do, dc, pairs)
    rh = S.run_grid(ho, hc, pairs)
    real = pd.DataFrame({
        "fast": pairs[:, 0], "slow": pairs[:, 1],
        "dev_return": rd.total_return, "holdout_return": rh.total_return,
        "dev_excess": excess(rd.total_return, bh_dev),
        "holdout_excess": excess(rh.total_return, bh_hol),
        "dev_trades": rd.n_trades, "dev_exposure": rd.exposure,
    })
    real.to_csv(OUT / "real.csv", index=False)

    best = real.loc[real["dev_return"].idxmax()]
    beat_dev = real["dev_return"] > bh_dev
    beat_both = beat_dev & (real["holdout_return"] > bh_hol)
    print("the real search")
    print(f"  buy-and-hold          development {bh_dev:7.3f}   "
          f"holdout {bh_hol:7.3f}")
    print(f"  best in development   development {best['dev_return']:7.3f}   "
          f"holdout {best['holdout_return']:7.3f}   "
          f"(fast {int(best['fast'])}, slow {int(best['slow'])})")
    print(f"  median strategy       development "
          f"{real['dev_return'].median():7.3f}   "
          f"holdout {real['holdout_return'].median():7.3f}")
    print(f"  beat buy-and-hold in development: {int(beat_dev.sum()):,} "
          f"of {len(real):,} ({beat_dev.mean():.1%})")
    print(f"  ... and in the holdout too:       {int(beat_both.sum()):,} "
          f"({beat_both.sum() / max(int(beat_dev.sum()), 1):.1%} of them)\n")

    # ---- the nulls ------------------------------------------------------
    rows = []
    for kind in ("shuffle", "block"):
        started = time.time()
        for i in range(cfg["nulls"]["n_surrogates"]):
            if kind == "shuffle":
                so, sc = S.shuffle_surrogate(do, dc, rng)
            else:
                so, sc = S.block_surrogate(do, dc, rng,
                                           cfg["nulls"]["block_size"])
            g = S.run_grid(so, sc, pairs)
            bh = S.buy_and_hold(so, sc)
            ex = excess(g.total_return, bh)
            rows.append({
                "kind": kind, "surrogate": i,
                "buy_and_hold": bh,
                "best_return": float(g.total_return.max()),
                "best_excess": float(ex.max()),
                "median_excess": float(np.median(ex)),
                "n_beating": int((g.total_return > bh).sum()),
            })
        n = cfg["nulls"]["n_surrogates"]
        s = pd.DataFrame(rows)
        s = s[s["kind"] == kind]
        print(f"{kind:<8} {n} surrogates in {time.time() - started:.0f}s   "
              f"best excess: median {s['best_excess'].median():6.3f}, "
              f"95th pct {s['best_excess'].quantile(0.95):6.3f}   "
              f"strategies beating buy-and-hold: median "
              f"{s['n_beating'].median():.1f}", flush=True)

    nulls = pd.DataFrame(rows)
    nulls.to_csv(OUT / "nulls.csv", index=False)

    # ---- the verdict ----------------------------------------------------
    real_best_excess = float(excess(best["dev_return"], bh_dev))
    real_n_beating = int(beat_dev.sum())
    sh = nulls[nulls["kind"] == "shuffle"]
    q = cfg["significance_percentile"] / 100.0

    verdict = {
        "n_strategies": int(len(real)),
        "buy_and_hold": {"dev": bh_dev, "holdout": bh_hol},
        "real": {
            "best_dev_return": float(best["dev_return"]),
            "best_dev_excess": real_best_excess,
            "best_holdout_return": float(best["holdout_return"]),
            "best_params": [int(best["fast"]), int(best["slow"])],
            "median_holdout_return": float(real["holdout_return"].median()),
            "n_beating_dev": real_n_beating,
            "n_beating_both": int(beat_both.sum()),
        },
    }
    for kind in ("shuffle", "block"):
        s = nulls[nulls["kind"] == kind]
        verdict[kind] = {
            "best_excess_median": float(s["best_excess"].median()),
            "best_excess_q95": float(s["best_excess"].quantile(q)),
            "share_at_least_double": float((s["best_excess"] >= 1.0).mean()),
            "n_beating_q05": float(s["n_beating"].quantile(0.05)),
            "n_beating_q95": float(s["n_beating"].quantile(0.95)),
            "p_value": float((s["best_excess"] >= real_best_excess).mean()),
        }

    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2),
                                      encoding="utf-8")

    print("\nthe comparison")
    print(f"  real data, best strategy's development excess over "
          f"buy-and-hold: {real_best_excess:+.3f}")
    for kind in ("shuffle", "block"):
        v = verdict[kind]
        print(f"  {kind:<8} null: median {v['best_excess_median']:+.3f}, "
              f"95th percentile {v['best_excess_q95']:+.3f}   "
              f"-> p = {v['p_value']:.3f}")
    print(f"\nWrote real.csv, nulls.csv, verdict.json -> {OUT}")


if __name__ == "__main__":
    main()
