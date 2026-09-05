"""
run_all.py -- The whole experiment, end to end.

    python run_all.py

  0. config.yaml still matches the frozen pre-registration
  1. leakage checks, including the builder broken on purpose
  2. fit and score everything, both horizons
  3. figures

Steps 0 and 1 are fatal. A forecast built on a feature that reads ahead looks
better than anything honest, so there is no point computing the rest until the
features are known to be causal.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PY = sys.executable

# Frozen in CRITERIA.md on 2026-09-05.
FROZEN = {
    "seed": 20260905,
    "splits.train_end": "2017-12-31",
    "splits.val_end": "2018-12-31",
    "horizons": [1, 24],
    "daylight_elevation_deg": 5.0,
    "clear_sky.quantile": 0.90,
    "clear_sky.elevation_bin_deg": 1.0,
    "features.cf_lags": [1, 2, 3, 24],
    "models.gbdt.max_iter": 300,
    "models.gbdt.early_stopping": False,
    "models.esn.n_reservoir": 400,
    "models.esn.spectral_radius": 0.9,
}

STEPS = [
    ("leakage checks", ["test_leakage.py"]),
    ("fit and score", ["run_experiment.py"]),
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
