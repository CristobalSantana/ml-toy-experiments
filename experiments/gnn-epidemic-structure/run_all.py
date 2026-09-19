"""
run_all.py -- The whole experiment, end to end.

    python run_all.py

  0. config.yaml still matches the frozen pre-registration
  1. implementation checks: are the hand-computed features right, did the
     generator's epidemic threshold match theory, and does an L-layer GNN
     see exactly L hops?
  2. every model on Erdős–Rényi, the depth sweep, structural drift to
     Barabási–Albert, and what the embedding encodes
  3. figures

Step 1 is fatal, and so is the control inside step 2. If hand-computed
structure cannot predict infection, the task is not about structure.

About ten minutes on CPU.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PY = sys.executable
GEN = HERE.parent.parent / "generators" / "sir_on_graph" / "outputs"

FROZEN = {
    "seed": 20260919,
    "task.min_final_size": 0.10,
    "task.train_share": 0.8,
    "task.val_share_of_train": 0.2,
    "gnn.n_hidden": 64,
    "gnn.layers": [1, 2, 3, 4, 6, 8],
    "gnn.epochs": 60,
    "gnn.lr": 0.003,
    "tabular.n_hidden": 64,
    "tabular.epochs": 40,
    "tabular.batch": 512,
    "tabular.lr": 0.003,
    "seeds": 3,
}

STEPS = [
    ("implementation checks", ["test_graph.py"]),
    ("models, depth, drift, embedding", ["run_experiment.py"]),
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
    for name in ("erdos_renyi", "barabasi_albert"):
        if not (GEN / f"sir_{name}.npz").exists():
            sys.exit(f"missing {GEN / f'sir_{name}.npz'}\n\nrun first:\n"
                     f"  python ../../generators/sir_on_graph/generate.py")
    print("  epidemics present")


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
