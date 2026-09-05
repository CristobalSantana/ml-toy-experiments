"""
run_all.py -- The whole experiment, end to end.

    python run_all.py

  1. config matches the pre-registration
  2. detector self-checks, including the in-sample version built to fail
  3. synthetic sweep      (arm A, the long one)
  4. real taxi pairs      (arm B; stops if the null pair is flagged)
  5. sample-size sweep    (arm C)
  6. figures

Steps 1 and 2 come first and are fatal. A detector that flags two samples from
one distribution produces numbers that look like findings and are not, and it
is cheaper to stop here than to read a plausible curve afterwards.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PY = sys.executable

# What CRITERIA.md froze on 2026-09-05. If config.yaml drifts from this, the
# run stops: a configuration file that can be edited between runs is a way to
# change the rules after seeing the results.
FROZEN = {
    "seed": 20260905,
    "detector.max_iter": 100,
    "detector.max_depth": 6,
    "detector.learning_rate": 0.1,
    "detector.test_size": 0.5,
    "detector.alpha": 0.05,
    "synthetic.trials_per_cell": 100,
    "synthetic.n_per_side": [500, 5000, 50000],
    "synthetic.n_features": [1, 8],
    "real.n_per_side": 50000,
    "real.ovl_bins": 100,
}

STEPS = [
    ("detector self-checks", ["test_detector.py"]),
    ("arm A - synthetic sweep", ["run_synthetic.py"]),
    ("arm B - real taxi pairs", ["run_real.py"]),
    ("arm C - sample size", ["run_sample_size.py"]),
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
                 + "\n\nEither restore the frozen values or record the change as "
                   "a deviation in README.md - but do not edit CRITERIA.md.")
    print("  config matches CRITERIA.md")


def main() -> None:
    started = time.time()
    print(f"{'=' * 70}\n[0/{len(STEPS)}] pre-registration check\n{'=' * 70}")
    check_config()

    for i, (label, args) in enumerate(STEPS, start=1):
        print(f"\n{'=' * 70}\n[{i}/{len(STEPS)}] {label}\n{'=' * 70}", flush=True)
        r = subprocess.run([PY, *args], cwd=HERE)
        if r.returncode != 0:
            sys.exit(f"\n{label} failed (exit {r.returncode}). Stopping: every "
                     f"step after this one would depend on it.")

    print(f"\n{'=' * 70}")
    print(f"Done in {(time.time() - started) / 60:.1f} min. "
          f"Results in {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
