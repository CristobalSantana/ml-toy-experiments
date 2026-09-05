"""
run_real.py -- Arm B: the same detector on NYC taxi months.

    python run_real.py

Five pairs, fixed in CRITERIA.md before anything ran. Pair 0 is two random
halves of a single month: no drift by construction. It is not one result among
five - it is the control that decides whether the other four mean anything. If
a detector flags two halves of the same month, nothing it says about 2019
against 2024 is worth reading, and `run_all.py` stops.

Every pair is compared at the same window size, so the numbers sit on one
scale. The synthetic arm has already shown what the window does on its own;
letting it vary here would mix the two effects.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402
import yaml                                     # noqa: E402

from detector import c2st, empirical_overlap    # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATASET = HERE.parents[1] / "datasets" / "us" / "nyc_taxi"
sys.path.insert(0, str(DATASET))
from load import FEATURES, load_month           # noqa: E402


def _subsample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    return df.iloc[idx][FEATURES].to_numpy(dtype=float)


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    det, real, seed0 = cfg["detector"], cfg["real"], cfg["seed"]
    n, bins = real["n_per_side"], real["ovl_bins"]
    OUT.mkdir(parents=True, exist_ok=True)

    cache: dict[str, pd.DataFrame] = {}

    def month(m: str) -> pd.DataFrame:
        if m not in cache:
            cache[m] = load_month(m, verbose=False)
        return cache[m]

    rows, per_feature = [], []
    for spec in real["pairs"]:
        rng = np.random.default_rng(seed0 + spec["id"])
        if spec["split"]:
            # the null: one month, cut in two at random
            df = month(spec["a"])
            perm = rng.permutation(len(df))
            half = min(n, len(df) // 2)
            A = df.iloc[perm[:half]][FEATURES].to_numpy(dtype=float)
            B = df.iloc[perm[half:2 * half]][FEATURES].to_numpy(dtype=float)
        else:
            A = _subsample(month(spec["a"]), n, rng)
            B = _subsample(month(spec["b"]), n, rng)

        ovls = {f: empirical_overlap(A[:, i], B[:, i], bins=bins)
                for i, f in enumerate(FEATURES)}
        r = c2st(A, B, seed=seed0 + spec["id"], max_iter=det["max_iter"],
                 max_depth=det["max_depth"], learning_rate=det["learning_rate"],
                 early_stopping=det["early_stopping"], test_size=det["test_size"],
                 alpha=det["alpha"])

        rows.append({"pair": spec["id"], "name": spec["name"],
                     "a": spec["a"], "b": spec["b"], "is_null": spec["split"],
                     "n_per_side": len(A), "accuracy": r.accuracy,
                     "p_value": r.p_value, "drift_called": r.drift,
                     "mean_overlap": float(np.mean(list(ovls.values()))),
                     "min_overlap": float(np.min(list(ovls.values()))),
                     "min_overlap_feature": min(ovls, key=ovls.get)})
        for f, v in ovls.items():
            per_feature.append({"pair": spec["id"], "feature": f, "overlap": v})

        print(f"  pair {spec['id']}  {spec['name']:<32} "
              f"acc {r.accuracy:.3f}  p {r.p_value:<9.2e} "
              f"{'DRIFT' if r.drift else 'no drift':<8}  "
              f"mean OVL {rows[-1]['mean_overlap']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "real_pairs.csv", index=False)
    pd.DataFrame(per_feature).to_csv(OUT / "real_overlap_by_feature.csv", index=False)

    # ---- the control, checked here and not left to the write-up ----
    null = df[df["is_null"]].iloc[0]
    print(f"\nControl: pair 0 is two halves of {null['a']}, no drift by construction.")
    if bool(null["drift_called"]):
        sys.exit(
            f"\nFAIL - the null pair was flagged as drift "
            f"(accuracy {null['accuracy']:.3f}, p = {null['p_value']:.2e}).\n"
            f"Nothing the detector says about the other four pairs means "
            f"anything until this is explained.")
    print(f"  OK - not flagged (accuracy {null['accuracy']:.3f}, "
          f"p = {null['p_value']:.2f})")

    # ---- P5: does the ordering match? ----
    drifted = df[~df["is_null"]].copy()
    by_detector = drifted.sort_values("accuracy", ascending=False)["pair"].tolist()
    by_overlap = drifted.sort_values("mean_overlap")["pair"].tolist()
    print(f"\nP5 - ordering by detector accuracy : {by_detector}")
    print(f"     ordering by (1 - mean overlap): {by_overlap}")
    print(f"     {'MATCH' if by_detector == by_overlap else 'MISMATCH'}"
          f"  (pre-registered expectation: 1 < 2 < 3 ~ 4)")


if __name__ == "__main__":
    main()
