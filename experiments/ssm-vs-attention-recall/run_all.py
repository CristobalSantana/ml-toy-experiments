"""
run_all.py -- The whole experiment, end to end.

    python run_all.py

  0. config.yaml still matches the frozen pre-registration, and the
     generator's data is present
  1. implementation checks: is the chunked SSM the fixed-state recurrence,
     can any model see past the current token, does the state stay fixed?
  2. train six models x three seeds, measure recall at five pair counts
  3. figures
  4. three post-hoc analyses, added after the results were in

Step 1 is fatal, and so is the control inside step 2. An SSM trained
through an algorithm that computed something other than its recurrence
could recall more than a fixed state allows, and the capacity it showed
would say nothing about fixed states.

About six and a half hours on a six-core laptop CPU, three trainings at a
time - the selective SSMs are most of it.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PY = sys.executable
GEN = HERE.parent.parent / "generators" / "associative_recall" / "outputs"

FROZEN = {
    "seed": 20261003,
    "data.n_keys": 256,
    "data.n_values": 256,
    "data.pair_counts": [4, 8, 16, 32, 64],
    "model.d_model": 64,
    "model.n_layers": 2,
    "model.n_heads": 4,
    "ssm.d_states": [4, 8, 16, 32],
    "ssm.lti_d_state": 32,
    "training.steps": 7500,
    "training.batch": 64,
    "training.lr": 0.001,
    "training.warmup": 300,
    "training.weight_decay": 0.1,
    "training.grad_clip": 1.0,
    "training.eval_every": 750,
    "training.seeds": 3,
}

STEPS = [
    ("implementation checks", ["test_models.py"]),
    ("train, measure recall", ["run_experiment.py"]),
    ("figures", ["make_figures.py"]),
    ("post-hoc analyses (added after the results; not pre-registered)", ["posthoc.py"]),
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
    if not (GEN / "recall_test.npz").exists():
        sys.exit(f"missing {GEN / 'recall_test.npz'}\n\nrun first:\n"
                 f"  python ../../generators/associative_recall/generate.py")
    print("  test sets present")


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
