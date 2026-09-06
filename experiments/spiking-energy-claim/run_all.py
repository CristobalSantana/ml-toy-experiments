"""
run_all.py -- The whole experiment, end to end.

    python run_all.py

  0. config.yaml still matches the frozen pre-registration
  1. implementation checks: is it actually a spiking network, and is the
     operation counter measuring rather than assuming?
  2. baselines, the dense control, the T sweep, the accounting
  3. figures

Step 1 is fatal, and so is the dense control inside step 2. A network whose
"spikes" are not binary would produce every number in this experiment, and a
task the dense network cannot beat smart persistence on is not a forecasting
task, so matching a spiking network to it would mean nothing.

About forty minutes on four CPU threads, almost all of it the T = 32 cells.
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
    "splits.train_end": "2017-12-31",
    "splits.val_end": "2018-12-31",
    "daylight_elevation_deg": 5.0,
    "network.n_hidden": 64,
    "network.beta": 0.9,
    "network.threshold": 1.0,
    "sweep.n_steps": [1, 2, 4, 8, 16, 32],
    "sweep.encodings": ["direct", "rate"],
    "sweep.seeds": 3,
    "training.epochs": 120,
    "training.patience": 15,
    "training.batch": 256,
    "training.lr": 0.005,
    "matching.rmse_tolerance": 0.02,
}

STEPS = [
    ("implementation checks", ["test_snn.py"]),
    ("baselines, sweep, accounting", ["run_experiment.py"]),
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
    if "best_effort" in cfg:
        b = cfg["best_effort"]
        print(f"  plus one post-hoc arm (H={b['n_hidden']}, T={b['n_steps']}), "
              f"added after the sweep and scored on nothing")


def check_dataset() -> None:
    raw = HERE.parent.parent / "datasets" / "de" / "opsd_solar" / "raw"
    files = sorted(raw.glob("*.csv")) if raw.exists() else []
    if not files:
        sys.exit("missing the solar dataset\n\nrun first:\n"
                 "  python ../../datasets/de/opsd_solar/load.py")
    print(f"  solar dataset present ({files[0].name})")


def main() -> None:
    started = time.time()
    print(f"{'=' * 70}\n[0/{len(STEPS)}] pre-registration check\n{'=' * 70}")
    check_config()
    check_dataset()
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
