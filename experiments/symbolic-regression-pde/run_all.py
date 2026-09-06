"""
run_all.py -- The whole experiment, end to end.

    python run_all.py

  0. config.yaml still matches the frozen pre-registration
  1. implementation checks, on problems whose answer is known by construction
  2. the noise sweep, the smoothing sweep, the threshold sweep, and the
     comparison against two black boxes
  3. figures

Step 1 is fatal, and so is the noiseless control inside step 2. If the method
cannot recover the equation from a field with no noise in it at all, the
implementation is wrong and every noise threshold below it would be a
property of the bug.

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

FROZEN = {
    "seed": 20260905,
    "domain.n_burn": 20,
    "domain.n_edge": 3,
    "sweep.seeds": 5,
    "sweep.threshold": 0.01,
    "sweep.threshold_sensitivity": [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1],
    "smoothing.polyorder": 3,
    "smoothing.windows": [5, 11, 21, 41],
    "recovery.coefficient_tolerance": 0.05,
    "recovery.seeds_required": 3,
    "extrapolation.train_fraction": 0.5,
    "extrapolation.gbdt.max_iter": 200,
    "extrapolation.gbdt.max_depth": 6,
    "extrapolation.gbdt.early_stopping": False,
}

# CRITERIA froze one sigma per decade. The grid actually run keeps all of
# them and adds 2x / 5x points, because P2 asks whether the transition is
# narrower than a decade and one point per decade cannot answer that. The
# check below is that none of the frozen levels went missing.
FROZEN_SIGMA = [0.0, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2]

STEPS = [
    ("implementation checks", ["test_sindy.py"]),
    ("noise, smoothing, threshold, black boxes", ["run_experiment.py"]),
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

    missing = [s for s in FROZEN_SIGMA if s not in cfg["sweep"]["sigma"]]
    if missing:
        bad.append(f"  sweep.sigma: frozen levels missing from the grid: {missing}")

    if bad:
        sys.exit("config.yaml no longer matches the pre-registration:\n"
                 + "\n".join(bad)
                 + "\n\nRestore the frozen values, or record the change as a "
                   "deviation in README.md - but do not edit CRITERIA.md.")
    print(f"  config matches CRITERIA.md "
          f"({len(cfg['sweep']['sigma'])} sigma levels, all "
          f"{len(FROZEN_SIGMA)} frozen ones present)")


def check_generator() -> None:
    npz = (HERE.parent.parent / "generators" / "diffusion_1d" / "outputs"
           / "diffusion_1d_solution.npz")
    if not npz.exists():
        sys.exit(f"missing {npz}\n\nrun first:\n"
                 f"  python ../../generators/diffusion_1d/generate.py")
    print("  reference solution present")


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
