"""
run_cross_sectional.py -- Phase 5: the grouped cross-validation for Experiment 02.

Runs the pre-registered protocol from CRITERIA.md: grouped K-fold by manzana,
5 folds x 5 seeds, on the development split only. The frozen holdout is never
touched here - it is loaded solely to be excluded.

Two arms, because TabPFN's documented ceiling is 10,000 training rows while the
development set has ~806,000:

  full            the five classical models on all development rows - the
                  realistic valuation problem.
  regime_limited  all six models on a subsample inside TabPFN's envelope, so
                  the accuracy/cost comparison against it is like-for-like
                  rather than against models that saw 80x more data.

Results are appended to outputs/cv_results.csv after every fit, so a crash
five hours in does not discard everything before it.

    python run_cross_sectional.py                 # both arms
    python run_cross_sectional.py --arm full
    python run_cross_sectional.py --models ridge,lightgbm
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

import features  # noqa: E402
import leakage  # noqa: E402
import models  # noqa: E402

OUTPUT_DIR = EXPERIMENT_ROOT / "outputs"
RESULTS_CSV = OUTPUT_DIR / "cv_results.csv"


def load_token_from_dotenv() -> None:
    """Read TABPFN_TOKEN from a local .env if present.

    The token is a personal API key tied to a licence acceptance, so it lives
    in a gitignored .env and never in the repo or in this file.
    """
    env = EXPERIMENT_ROOT / ".env"
    if not env.exists() or os.environ.get("TABPFN_TOKEN"):
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("TABPFN_TOKEN"):
            os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")


def append_result(row: dict) -> None:
    """Write incrementally: a long run must survive an interruption."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(RESULTS_CSV, mode="a", header=not RESULTS_CSV.exists(), index=False)


def already_done() -> set[tuple]:
    """(model, arm, seed, fold) combinations already in the results file, so a
    resumed run skips them instead of duplicating work."""
    if not RESULTS_CSV.exists():
        return set()
    df = pd.read_csv(RESULTS_CSV)
    return set(zip(df["model"], df["arm"], df["seed"], df["fold"]))


def subsample_within_groups(X, y, groups, n_rows: int, seed: int):
    """Take ~n_rows while keeping whole manzanas together.

    Sampling rows independently would split manzanas across the arm boundary
    and quietly weaken the grouping guarantee, so we sample *manzanas* until
    the row budget is met.
    """
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.sort(groups.unique()))
    sizes = groups.value_counts()
    take, total = [], 0
    for g in uniq:
        if total >= n_rows:
            break
        take.append(g)
        total += int(sizes[g])
    mask = groups.isin(set(take))
    return X[mask], y[mask], groups[mask]


def run_arm(arm: str, X, y, groups, cfg, model_names: list[str], n_jobs: int = -1) -> None:
    seeds = cfg["split"]["seeds"]
    n_folds = cfg["split"]["n_folds"]
    done = already_done()

    for seed in seeds:
        if arm == "regime_limited":
            Xa, ya, ga = subsample_within_groups(
                X, y, groups, models.TABPFN_MAX_SAMPLES, seed)
        else:
            Xa, ya, ga = X, y, groups

        gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(gkf.split(Xa, ya, groups=ga)):
            # Rule 3 is verified on every actual split, not assumed from the
            # splitter's contract.
            leakage.check_group_disjoint(ga, tr, te, f"{arm} seed={seed} fold={fold}")

            for name in model_names:
                key = (name, arm, seed, fold)
                if key in done:
                    print(f"  skip {name} {arm} seed={seed} fold={fold} (already done)", flush=True)
                    continue
                if name == "tabpfn" and len(tr) > models.TABPFN_MAX_SAMPLES:
                    print(f"  skip tabpfn: {len(tr)} train rows exceeds its "
                          f"{models.TABPFN_MAX_SAMPLES} limit", flush=True)
                    continue

                t0 = time.perf_counter()
                try:
                    res = models.fit_evaluate(
                        name, Xa.iloc[tr], ya.iloc[tr], Xa.iloc[te], ya.iloc[te],
                        seed=seed, fold=fold, arm=arm, n_jobs=n_jobs,
                        notes=("subsampled to TabPFN's documented limit"
                               if arm == "regime_limited" else ""),
                    )
                except Exception as e:  # noqa: BLE001
                    # Record the failure instead of losing the whole run to it.
                    print(f"  FAIL {name} {arm} seed={seed} fold={fold}: "
                          f"{type(e).__name__}: {str(e)[:160]}", flush=True)
                    append_result({"model": name, "arm": arm, "seed": seed, "fold": fold,
                                   "notes": f"FAILED: {type(e).__name__}: {str(e)[:200]}"})
                    continue

                append_result(res.as_row())
                print(f"  {name:<14} {arm:<15} seed={seed} fold={fold} "
                      f"MAE={res.mae:.4f} R2={res.r2:.3f} "
                      f"fit={res.fit_seconds:7.1f}s mem={res.peak_memory_mb:6.0f}MB "
                      f"({time.perf_counter()-t0:.0f}s total)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--arm", choices=["full", "regime_limited", "both"], default="both")
    ap.add_argument("--models", default=None, help="comma-separated subset")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    load_token_from_dotenv()
    cfg = features.load_config()

    X, y, groups, holdout, _ = features.build(cfg, verbose=True)
    dev = ~holdout
    Xd, yd, gd = X[dev], y[dev], groups[dev]
    print(f"\ndevelopment set: {len(Xd)} rows, {gd.nunique()} manzanas "
          f"(holdout of {int(holdout.sum())} rows stays untouched)\n", flush=True)

    arms = ["full", "regime_limited"] if args.arm == "both" else [args.arm]
    for arm in arms:
        default = models.CLASSICAL if arm == "full" else models.ALL_MODELS
        names = args.models.split(",") if args.models else default
        if arm == "regime_limited" and "tabpfn" in names and not os.environ.get("TABPFN_TOKEN"):
            print("NOTE: TABPFN_TOKEN not set - TabPFN will fail and be recorded as such.\n"
                  "      Put it in experiments/02-plusvalia/.env to include it.\n", flush=True)
        print(f"=== arm: {arm} | models: {names} ===", flush=True)
        run_arm(arm, Xd, yd, gd, cfg, names, n_jobs=args.n_jobs)

    print(f"\nResults -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
