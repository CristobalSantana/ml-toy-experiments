"""
run_all.py -- Reproduce Experiment 02 end to end, from a fixed seed.

    python run_all.py                # everything
    python run_all.py --skip-cv      # reuse existing CV results

Order matters: the cross-validation must finish before the final holdout is
touched, and nothing else should compete for the CPU while timings are being
measured - training time is a reported metric here, and a background job
inflates it (this happened once during development and the results were
discarded and re-run).

Prerequisites, both manual because the sources are credential-gated:
  datasets/ch/sii_cadastre/raw/<comuna>/*.zip     SII Detalle Catastral
  datasets/ch/bcch_ipv/raw/*.xlsx                 BCCh IPV cuadros
  experiments/02-plusvalia/.env                   TABPFN_TOKEN=<key>
See each dataset's README for the exact download steps.

Expected runtime on 6 CPU cores: ~4.5 hours, dominated by the CV sweep.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PY = sys.executable

STEPS = [
    ("Build SII cadastre table", REPO_ROOT / "datasets/ch/sii_cadastre/load.py", []),
    ("Build BCCh IPV table", REPO_ROOT / "datasets/ch/bcch_ipv/load.py", []),
    ("Verify leakage checks fire", HERE / "test_leakage.py", []),
    ("Cross-validation, both arms", HERE / "run_cross_sectional.py", ["--arm", "both"]),
    ("Aggregate + evaluate decision rules", HERE / "analyze.py", []),
    ("Drift analysis", HERE / "drift.py", []),
    ("Interpretability (SHAP, PDP/ALE, PySR)", HERE / "interpret.py", []),
    ("FINAL: frozen holdout", HERE / "final_holdout.py", []),
    ("Figures", HERE / "make_figures.py", []),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--skip-cv", action="store_true",
                    help="reuse outputs/cv_results.csv instead of re-running the sweep")
    args = ap.parse_args()

    t_start = time.perf_counter()
    for i, (label, script, extra) in enumerate(STEPS, 1):
        if args.skip_cv and script.name == "run_cross_sectional.py":
            print(f"\n[{i}/{len(STEPS)}] SKIPPED (--skip-cv): {label}")
            continue
        print(f"\n{'=' * 72}\n[{i}/{len(STEPS)}] {label}\n  {script.name}\n{'=' * 72}",
              flush=True)
        t0 = time.perf_counter()
        r = subprocess.run([PY, "-u", str(script), *extra], cwd=script.parent)
        if r.returncode != 0:
            sys.exit(f"\nFAILED at step {i} ({label}) with exit code {r.returncode}")
        print(f"  done in {time.perf_counter() - t0:.0f}s", flush=True)

    print(f"\nAll steps completed in {(time.perf_counter() - t_start)/60:.0f} min.")
    print(f"Results and figures -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
