"""
run_all.py -- The whole experiment, end to end.

    python run_all.py

  0. config.yaml still matches the frozen pre-registration
  1. implementation checks: is the symplectic gradient wired correctly, and
     does the integrator reproduce the generator?
  2. fit both models, roll them out, break the inductive bias, extrapolate
  3. figures

Step 1 is fatal, and so is the control inside step 2. An HNN with the two
partial derivatives swapped trains just as well and describes a pendulum
that swings the wrong way; nothing in a loss curve would show it.

About twenty-five minutes on CPU, most of it the HNN's rollouts, which
need autograd on every one of 10,000 RK4 steps.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PY = sys.executable
GEN = HERE.parent.parent / "generators" / "hamiltonian_pendulum" / "outputs"

FROZEN = {
    "seed": 20260919,
    "split.n_train_trajectories": 140,
    "split.energy_train_max": 0.5,
    "split.energy_test_range": [0.5, 0.9],
    "training.n_states": 20000,
    "training.epochs": 300,
    "training.batch": 256,
    "training.lr": 0.001,
    "training.seeds": 3,
    "network.n_hidden": 64,
    "rollout.dt": 0.01,
    "rollout.t_end": 100.0,
    "rollout.n_initial_conditions": 30,
}

STEPS = [
    ("implementation checks", ["test_models.py"]),
    ("fit, roll out, break, extrapolate", ["run_experiment.py"]),
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


def check_generator() -> None:
    for name in ("ideal", "damped"):
        if not (GEN / f"pendulum_{name}.npz").exists():
            sys.exit(f"missing {GEN / f'pendulum_{name}.npz'}\n\nrun first:\n"
                     f"  python ../../generators/hamiltonian_pendulum/generate.py")
    print("  reference trajectories present")


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
