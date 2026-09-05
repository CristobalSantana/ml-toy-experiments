"""
run_synthetic.py -- Arm A: how detector quality falls as the overlap rises.

    python run_synthetic.py

For each (overlap, window size, feature count) cell, `trials_per_cell` trials
are run. Half draw two samples that genuinely differ at that overlap; half
draw two samples from the same distribution. The detector calls each one, and
the cell's score is the Matthews correlation of those calls against the truth.

Balanced by construction, so a detector that always answers "drift" scores
0.5 accuracy - and 0 MCC, which is the honest description of it.

At OVL = 1.00 the two distributions are identical, so every trial is a null
and MCC is 0 by construction. That cell is not a data point on the curve; it
is the calibration check the pre-registration calls P2, and its false-alarm
rate is reported separately.

Cost: 20 overlaps x 3 window sizes x 2 feature counts x 100 trials = 12,000
classifier fits, about 100 minutes in series. Trials are independent, so they
are farmed out across cores with one thread each - sklearn's own threading
would otherwise fight the process pool for the same cores.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")   # before sklearn is imported

import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402
import yaml                                     # noqa: E402
from joblib import Parallel, delayed            # noqa: E402

from detector import c2st, gaussian_pair, mcc   # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def _trial(ovl: float, n: int, d: int, drifted: bool, seed: int,
           det: dict) -> tuple[bool, bool, float]:
    """One trial: draw a pair, ask the detector, return (truth, call, accuracy).

    A "null" trial draws both samples at OVL = 1.0, which is a separation of
    zero - the same distribution twice, not a small difference.
    """
    rng = np.random.default_rng(seed)
    a, b = gaussian_pair(n, d, ovl=(ovl if drifted else 1.0), rng=rng)
    r = c2st(a, b, seed=seed, max_iter=det["max_iter"], max_depth=det["max_depth"],
             learning_rate=det["learning_rate"], early_stopping=det["early_stopping"],
             test_size=det["test_size"], alpha=det["alpha"])
    return drifted, r.drift, r.accuracy


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    det, syn, seed0 = cfg["detector"], cfg["synthetic"], cfg["seed"]
    OUT.mkdir(parents=True, exist_ok=True)

    ovls = np.round(np.arange(syn["ovl_min"], syn["ovl_max"] + 1e-9,
                              syn["ovl_step"]), 4)
    cells = [(o, n, d) for d in syn["n_features"]
             for n in syn["n_per_side"] for o in ovls]
    trials = syn["trials_per_cell"]

    print(f"{len(cells)} cells x {trials} trials = {len(cells)*trials:,} fits")
    print(f"overlap {ovls[0]:.2f} .. {ovls[-1]:.2f}   n {syn['n_per_side']}   "
          f"features {syn['n_features']}\n")

    rows, started = [], time.time()
    for k, (ovl, n, d) in enumerate(cells, start=1):
        # deterministic per-cell seeds, so a rerun reproduces every trial
        base = seed0 + int(round(ovl * 1000)) * 100_003 + n * 31 + d
        jobs = [delayed(_trial)(ovl, n, d, i < trials // 2, base + i, det)
                for i in range(trials)]
        res = Parallel(n_jobs=-1, batch_size=4)(jobs)

        truth = np.array([t for t, _, _ in res])
        call = np.array([c for _, c, _ in res])
        acc = np.array([a for _, _, a in res])

        false_alarm = float(call[~truth].mean()) if (~truth).any() else float("nan")
        power = float(call[truth].mean()) if truth.any() else float("nan")
        rows.append({"overlap": float(ovl), "n_per_side": n, "n_features": d,
                     "mcc": mcc(truth, call), "false_alarm_rate": false_alarm,
                     "detection_rate": power,
                     "mean_accuracy_drifted": float(acc[truth].mean()),
                     "mean_accuracy_null": float(acc[~truth].mean()),
                     "trials": trials})

        el = time.time() - started
        eta = el / k * (len(cells) - k)
        print(f"  [{k:>3}/{len(cells)}] ovl {ovl:.2f}  n {n:>6}  d {d}   "
              f"MCC {rows[-1]['mcc']:>6.3f}   FA {false_alarm:.2f}   "
              f"eta {eta/60:>5.1f} min", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "synthetic.csv", index=False)
    print(f"\nWrote synthetic.csv in {(time.time()-started)/60:.1f} min")

    # P2: the calibration cell, checked here rather than left to the write-up
    null_cells = df[np.isclose(df["overlap"], 1.0)]
    print("\nP2 - false-alarm rate at OVL = 1.00 (must sit in [0.01, 0.10]):")
    for _, r in null_cells.iterrows():
        ok = 0.01 <= r["false_alarm_rate"] <= 0.10
        print(f"  {'OK  ' if ok else 'FAIL'} n={int(r['n_per_side']):>6} "
              f"d={int(r['n_features'])}   {r['false_alarm_rate']:.3f}")


if __name__ == "__main__":
    main()
