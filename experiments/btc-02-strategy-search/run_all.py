"""
run_all.py -- The whole experiment, end to end.

    python run_all.py

  0. config.yaml still matches the frozen pre-registration
  1. implementation checks, the first of which is that this backtester
     reproduces btc-01's published buy-and-hold return
  2. the real search, the two nulls, and the comparison
  3. figures

Step 1 is fatal, and so is the control inside step 2. If this file's
backtester is not btc-01's backtester, the two experiments are not measuring
the same thing and the comparison between them - which is the reason this
experiment exists - would be meaningless.

About three minutes.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PY = sys.executable
BTC01 = HERE.parent / "btc-01-rsi-divergence"

FROZEN = {
    "seed": 20260906,
    "split.dev_end": "2023-12-31",
    "split.holdout_start": "2024-01-01",
    "costs.fee": 0.0010,
    "costs.slippage": 0.0005,
    "grid.fast_max": 40,
    "grid.slow_max": 200,
    "grid.slow_step": 5,
    "nulls.n_surrogates": 200,
    "nulls.block_size": 20,
    "significance_percentile": 95,
}

STEPS = [
    ("implementation checks", ["test_search.py"]),
    ("the search, the nulls, the comparison", ["run_experiment.py"]),
    ("figures", ["make_figures.py"]),
]


def check_config() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    bad = []
    for path, expected in FROZEN.items():
        node = cfg
        for key in path.split("."):
            node = node[key]
        if node != expected:
            bad.append(f"  {path}: config has {node!r}, CRITERIA froze {expected!r}")
    if bad:
        sys.exit("config.yaml no longer matches the pre-registration:\n"
                 + "\n".join(bad)
                 + "\n\nRestore the frozen values, or record the change as a "
                   "deviation in README.md - but do not edit CRITERIA.md.")
    print("  config matches CRITERIA.md")


def check_costs_match_btc01() -> None:
    """The cost model has to be btc-01's, and it is written down twice.

    Once in this experiment's config.yaml and once as a constant in
    btc-01/backtest.py. Two copies of a number drift; this compares them
    rather than trusting that they still agree.
    """
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    src = (BTC01 / "backtest.py").read_text(encoding="utf-8")
    found = {}
    for line in src.splitlines():
        for name, key in (("FEE", "fee"), ("SLIPPAGE", "slippage")):
            if line.startswith(f"{name} "):
                found[key] = float(line.split("=")[1].split("#")[0].strip())
    mismatch = [f"  {k}: this experiment {cfg['costs'][k]}, btc-01 {v}"
                for k, v in found.items() if cfg["costs"][k] != v]
    if mismatch:
        sys.exit("cost model has drifted from btc-01:\n" + "\n".join(mismatch))
    print(f"  cost model matches btc-01 "
          f"({found['fee']:.2%} fee + {found['slippage']:.2%} slippage per side)")


def check_data() -> None:
    csv = BTC01 / "data" / "btcusdt_1d.csv"
    if not csv.exists():
        sys.exit(f"missing {csv}\n\nrun first:\n"
                 f"  python ../btc-01-rsi-divergence/data_loader.py")
    print("  daily bars present (shared with btc-01, checksummed there)")


def main() -> None:
    started = time.time()
    print(f"{'=' * 70}\n[0/{len(STEPS)}] pre-registration check\n{'=' * 70}")
    check_config()
    check_costs_match_btc01()
    check_data()
    for i, (label, args) in enumerate(STEPS, start=1):
        print(f"\n{'=' * 70}\n[{i}/{len(STEPS)}] {label}\n{'=' * 70}", flush=True)
        r = subprocess.run([PY, *args], cwd=HERE)
        if r.returncode != 0:
            sys.exit(f"\n{label} failed (exit {r.returncode}). Stopping.")
    print(f"\n{'=' * 70}")
    print(f"Done in {(time.time() - started) / 60:.1f} min. "
          f"Results in {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
