"""
run_all.py -- The whole experiment, end to end.

    python run_all.py

  0. config.yaml still matches the frozen pre-registration, and the
     generator's data is present
  1. implementation checks: is the spectral layer grid-independent, is the
     control what it claims, is the data band-limited as stated?
  2. train at 64 points (and the FNO at 256), evaluate on four grids
  3. figures
  4. two post-hoc checks, added after the results were in

Step 1 is fatal, and so is the control inside step 2. An FNO whose spectral
layer secretly depended on the number of grid points would train exactly
as well and fail the super-resolution test for a reason unrelated to it.

About five hours on a six-core laptop CPU, three trainings at a time.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PY = sys.executable
GEN = HERE.parent.parent / "generators" / "burgers_1d" / "outputs"

FROZEN = {
    "seed": 20261003,
    "data.viscosities": [0.1, 0.03, 0.01],
    "data.n_train": 1000,
    "data.train_resolution": 64,
    "data.high_resolution": 256,
    "data.eval_resolutions": [64, 128, 256, 512],
    "training.epochs": 300,
    "training.batch": 20,
    "training.lr": 0.001,
    "training.lr_halve_every": 60,
    "training.weight_decay": 0.0001,
    "training.seeds": 3,
    "fno.width": 32,
    "fno.modes": 16,
    "fno.layers": 4,
    "cnn.channels": 63,
    "cnn.kernel": 5,
    "cnn.dilations": [1, 2, 4, 8, 16, 4, 1],
}

STEPS = [
    ("implementation checks", ["test_models.py"]),
    ("train, evaluate on four grids", ["run_experiment.py"]),
    ("figures", ["make_figures.py"]),
    ("post-hoc checks (added after the results; not pre-registered)", ["posthoc.py"]),
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


def check_generator() -> None:
    if not (GEN / "burgers.npz").exists():
        sys.exit(f"missing {GEN / 'burgers.npz'}\n\nrun first:\n"
                 f"  python ../../generators/burgers_1d/generate.py")
    print("  reference solutions present")


def main() -> None:
    started = time.time()
    print(f"{'=' * 70}\n[0/{len(STEPS)}] pre-registration check\n{'=' * 70}")
    check_config()
    check_generator()
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
