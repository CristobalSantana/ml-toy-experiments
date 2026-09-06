"""
run_all.py -- The whole experiment, end to end.

    python run_all.py

  0. config.yaml still matches the frozen pre-registration
  1. implementation checks on synthetic data, where exchangeability is true
     by construction and any failure is the code rather than the world
  2. calibrate on 2024-06, then walk away from it
  3. figures

Step 1 is fatal, and so is the in-distribution control inside step 2. A
conformal predictor that cannot cover its own held-out data is broken, and
every number measured on a drifted month would be measuring the bug.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PY = sys.executable

FROZEN = {
    "seed": 20260905,
    "alpha": 0.1,
    "calibration_month": "2024-06",
    "months": ["2024-06", "2024-05", "2023-06", "2020-04", "2019-06"],
    "splits.n_fit": 100000,
    "splits.n_calibrate": 50000,
    "splits.n_eval": 50000,
    "recalibration.n_fresh": 2000,
    "model.max_iter": 300,
    "model.early_stopping": False,
}

STEPS = [
    ("implementation checks", ["test_conformal.py"]),
    ("calibrate once, then drift", ["run_experiment.py"]),
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


def main() -> None:
    started = time.time()
    print(f"{'=' * 70}\n[0/{len(STEPS)}] pre-registration check\n{'=' * 70}")
    check_config()
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
