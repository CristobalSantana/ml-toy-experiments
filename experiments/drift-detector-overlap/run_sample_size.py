"""
run_sample_size.py -- Arm C: the same shift, seen through windows of growing size.

    python run_sample_size.py

Takes one pair of adjacent months - the smallest real drift in the set - and
asks a per-feature Kolmogorov-Smirnov test how many of the seven features are
"significantly" different, at windows from a thousand rows to a million.

The drift does not change between those runs. Only the window does. If the
count climbs anyway, the alert is being driven by how much data was collected
rather than by how much the world moved, which is prediction P6.

A null control runs alongside: two random halves of a single month, swept over
the same window sizes. There the count must stay near the false-alarm rate the
threshold implies, at every n. A test that also climbs under the null is
broken rather than merely oversensitive, and the difference between those two
diagnoses is the reason the control is here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATASET = HERE.parents[1] / "datasets" / "us" / "nyc_taxi"
sys.path.insert(0, str(DATASET))
from load import FEATURES, load_month           # noqa: E402


def _flagged(A: pd.DataFrame, B: pd.DataFrame, alpha: float) -> tuple[int, dict]:
    """How many of the seven features a KS test calls different."""
    ps = {f: float(stats.ks_2samp(A[f].to_numpy(), B[f].to_numpy()).pvalue)
          for f in FEATURES}
    return sum(p < alpha for p in ps.values()), ps


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    ss, real, seed0 = cfg["sample_size"], cfg["real"], cfg["seed"]
    alpha, repeats = ss["alpha"], ss["repeats"]
    OUT.mkdir(parents=True, exist_ok=True)

    spec = next(p for p in real["pairs"] if p["id"] == ss["pair"])
    print(f"drifted pair: {spec['name']}  ({spec['a']} vs {spec['b']})")
    a_all, b_all = load_month(spec["a"], verbose=False), load_month(spec["b"], verbose=False)
    control_all = load_month("2024-06", verbose=False)
    print(f"  rows available: {len(a_all):,} / {len(b_all):,}\n")

    rows = []
    for n in ss["n_grid"]:
        # "control", not "null": pandas read_csv turns the string "null"
        # into NaN by default, so a column holding it silently loses the
        # arm on the way back in and the figure drops the series.
        for arm, src in (("drifted", (a_all, b_all)), ("control", None)):
            counts = []
            for rep in range(repeats):
                rng = np.random.default_rng(seed0 + n + rep * 7919 + (arm == "control") * 13)
                if arm == "drifted":
                    A_df, B_df = src
                    if min(len(A_df), len(B_df)) < n:
                        continue
                    A = A_df.iloc[rng.choice(len(A_df), n, replace=False)]
                    B = B_df.iloc[rng.choice(len(B_df), n, replace=False)]
                else:
                    if len(control_all) < 2 * n:
                        continue
                    idx = rng.choice(len(control_all), 2 * n, replace=False)
                    A = control_all.iloc[idx[:n]]
                    B = control_all.iloc[idx[n:]]
                k, _ = _flagged(A, B, alpha)
                counts.append(k)
            if not counts:
                continue
            rows.append({"n_per_side": n, "arm": arm,
                         "mean_flagged": float(np.mean(counts)),
                         "min_flagged": int(np.min(counts)),
                         "max_flagged": int(np.max(counts)),
                         "repeats": len(counts), "n_features": len(FEATURES)})
            print(f"  n={n:>9,}  {arm:<8}  flagged "
                  f"{np.mean(counts):>4.1f} / {len(FEATURES)}   "
                  f"(min {np.min(counts)}, max {np.max(counts)}, "
                  f"{len(counts)} repeats)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sample_size.csv", index=False)
    print(f"\nWrote sample_size.csv")

    d = df[df.arm == "drifted"].sort_values("n_per_side")["mean_flagged"].to_numpy()
    monotone = bool(np.all(np.diff(d) >= -1e-9))
    print(f"\nP6 - flagged count rises with n on the drifted pair: "
          f"{'YES' if monotone else 'NO'}  ({', '.join(f'{v:.1f}' for v in d)})")
    nul = df[df.arm == "control"]["mean_flagged"].to_numpy()
    print(f"     control stays flat: "
          f"{', '.join(f'{v:.1f}' for v in nul)}  (expected near "
          f"{alpha * len(FEATURES):.1f} of {len(FEATURES)})")


if __name__ == "__main__":
    main()
