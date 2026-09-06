"""
run_all.py -- The whole experiment, end to end.

    python run_all.py

  0. config.yaml still matches the frozen pre-registration
  1. the leakage demonstration - the excluded feature, and what it would buy
  2. the three arms
  3. figures

Step 1 is fatal. If tip_amount stops being a leak, the task is a different
task and the write-up describes something that no longer exists.
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
    "task.train_month": "2024-05",
    "task.test_month": "2024-06",
    "task.test_size": 200000,
    "learning_curve.train_sizes": [100, 300, 1000, 3000, 10000, 30000, 100000, 300000],
    "learning_curve.seeds": 5,
    "robustness.n_train": 30000,
    "models.gbdt.max_iter": 200,
    "models.gbdt.early_stopping": False,
    "models.hdc.dim": 10000,
    "models.hdc.levels": 64,
    "models.hdc.retrain_epochs": 0,
}

STEPS = [
    ("leakage demonstration", ["test_leakage.py"]),
    ("three arms", ["run_experiment.py"]),
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
