"""
final_holdout.py -- The single final evaluation on the frozen holdout.

CRITERIA.md froze 10% of manzanas at the start and forbade touching them for
training, tuning, model selection, drift-window choice or feature decisions.
This script is the one time they are used. It runs once, at the end, after
every other decision has already been made and committed.

Each model is trained on the entire development split and evaluated on the
holdout, over the pre-registered seeds, so the reported figure is a mean ±
spread rather than one run.

TabPFN cannot appear here: the development split is 806k rows against its
5,000-row CPU ceiling. That absence is the result, not a gap - a model that
cannot ingest the data cannot be deployed on it, and the regime-limited arm
already measured it where it can run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

import features as F  # noqa: E402
import leakage  # noqa: E402
import models as M  # noqa: E402

OUTPUT_DIR = EXPERIMENT_ROOT / "outputs"
RESULTS = OUTPUT_DIR / "final_holdout.csv"


def main() -> None:
    cfg = yaml.safe_load((EXPERIMENT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, groups, holdout, _ = F.build(cfg, verbose=True)
    dev = ~holdout
    Xd, yd = X[dev], y[dev]
    Xh, yh = X[holdout], y[holdout]

    # The holdout must share no manzana with development - verified, not assumed.
    leakage.check_group_disjoint(groups, np.where(dev)[0], np.where(holdout)[0],
                                 "final holdout")
    print(f"\nFINAL EVALUATION (the frozen holdout is being used for the first time)")
    print(f"  train on {len(Xd)} development rows -> evaluate on {len(Xh)} holdout rows "
          f"({groups[holdout].nunique()} manzanas)\n")

    rows = []
    for name in M.CLASSICAL:
        for seed in cfg["split"]["seeds"]:
            res = M.fit_evaluate(name, Xd, yd, Xh, yh, seed=seed, fold=-1,
                                 arm="final_holdout")
            rows.append(res.as_row())
            print(f"  {name:<14} seed={seed} MAE={res.mae:.4f} R2={res.r2:.3f} "
                  f"MdAPE={res.mdape_uf_m2:.1f}% fit={res.fit_seconds:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(RESULTS, index=False)

    df = pd.DataFrame(rows)
    summary = df.groupby("model").agg(
        mae=("mae", "mean"), mae_sd=("mae", "std"),
        r2=("r2", "mean"), mdape=("mdape_uf_m2", "mean"),
        fit_s=("fit_seconds", "mean"), peak_mem_mb=("peak_memory_mb", "mean"),
    ).sort_values("mae")
    print("\n=== FINAL HOLDOUT RESULT (mean over seeds) ===")
    print(summary.round(4).to_string())
    summary.to_csv(OUTPUT_DIR / "final_holdout_summary.csv")
    print(f"\nWrote {RESULTS.name} and final_holdout_summary.csv -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
