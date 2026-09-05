"""
run_all.py -- The whole experiment, end to end.

    python run_all.py

Runs, in this order:

  1. the look-ahead checks, including the deliberately broken versions
  2. the daily arm  - 18-cell grid on development, then on the held-out period
  3. the hourly arm - the same grid, pre-registered separately in CRITERIA-1h.md
  4. the figures

The order matters. The leakage checks come first and the run stops if any of
them fails: a grid that took a peek at future bars produces numbers worth
nothing, and it is better to find that out before spending the time than to
read a plausible-looking Sharpe afterwards.

Costs, execution rules, the grid and the held-out boundary are frozen in
CRITERIA.md and are not read from a config file - see the README for why.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

STEPS = [
    ("look-ahead checks", ["test_leakage.py"]),
    ("daily arm", ["run_experiment.py"]),
    ("hourly arm", ["run_experiment.py", "--interval", "1h"]),
    ("figures", ["make_figures.py"]),
]


def main() -> None:
    started = time.time()
    for i, (label, args) in enumerate(STEPS, start=1):
        print(f"\n{'=' * 70}\n[{i}/{len(STEPS)}] {label}\n{'=' * 70}", flush=True)
        r = subprocess.run([PY, *args], cwd=HERE)
        if r.returncode != 0:
            sys.exit(f"\n{label} failed (exit {r.returncode}). Stopping: every "
                     f"step after this one would be reporting numbers that "
                     f"depend on it.")

    mins = (time.time() - started) / 60
    print(f"\n{'=' * 70}")
    print(f"Done in {mins:.1f} min. Results in {HERE / 'outputs'}")
    print("The pre-registered decision for each arm is in decision.md and "
          "decision_1h.md.")


if __name__ == "__main__":
    main()
