"""
run_experiment.py -- Calibrate once, then walk away from the calibration month.

    python run_experiment.py

The model is fitted on 2024-06 and never refitted. The interval half-width is
set on 2024-06 and never reset. Then the same frozen system is pointed at
months that are progressively less like it, and the coverage it promised is
measured against the coverage it delivers.

A second pass repeats the evaluation after recalibrating on a small sample of
each test month, which is what a practitioner with a little fresh labelled
data would actually do.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

import data as D
from conformal import calibrate, coverage_report

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def empirical_overlap(x: np.ndarray, y: np.ndarray, bins: int = 100) -> float:
    """Same estimator as drift-detector-overlap, so the two experiments put
    their months on one comparable axis."""
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    if hi <= lo:
        return 1.0
    e = np.linspace(lo, hi, bins + 1)
    hx, _ = np.histogram(x, bins=e, density=True)
    hy, _ = np.histogram(y, bins=e, density=True)
    return float(np.minimum(hx, hy).sum() * (e[1] - e[0]))


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    seed, alpha, sp = cfg["seed"], cfg["alpha"], cfg["splits"]
    feats, target = cfg["features"], D.TARGET
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- the calibration month -----------------------------------------
    cal_month = cfg["calibration_month"]
    print(f"calibration month {cal_month}")
    base = D.load_month(cal_month, verbose=True)
    fit_df, cal_df, eval_df = D.three_way_split(
        base, sp["n_fit"], sp["n_calibrate"], sp["n_eval"], seed)

    model = HistGradientBoostingRegressor(random_state=seed, **cfg["model"])
    model.fit(fit_df[feats].to_numpy(), fit_df[target].to_numpy())

    q = calibrate(cal_df[target].to_numpy(),
                  model.predict(cal_df[feats].to_numpy()), alpha)
    print(f"  half-width from {len(cal_df):,} calibration trips: ${q:.2f}\n")

    # ---- P1: the in-distribution control, checked before anything else --
    ctrl = coverage_report(eval_df[target].to_numpy(),
                           model.predict(eval_df[feats].to_numpy()),
                           q, alpha, lower_bound=0.0)
    print(f"control ({cal_month}, held out): coverage {ctrl['coverage']:.4f}  "
          f"target {ctrl['target']:.2f}")
    if abs(ctrl["coverage"] - (1 - alpha)) > 0.01:
        sys.exit(
            f"\nFAIL - coverage on held-out data from the calibration month is "
            f"{ctrl['coverage']:.4f}, more than a point from the {1-alpha:.2f} "
            f"the method guarantees. The implementation is wrong and nothing "
            f"measured on the other months would mean anything.")
    print("  OK - the guarantee holds where its assumption does\n")

    # ---- every month, with the frozen width and with a fresh one --------
    ref_feature = cal_df["trip_distance"].to_numpy()
    rows = []
    for month in cfg["months"]:
        df = D.load_month(month) if month != cal_month else eval_df
        ev = D.sample(df, sp["n_eval"], seed + 1)
        X, y = ev[feats].to_numpy(), ev[target].to_numpy()
        pred = model.predict(X)

        ov = float(np.mean([
            empirical_overlap(cal_df[f].to_numpy(), ev[f].to_numpy())
            for f in feats]))

        frozen = coverage_report(y, pred, q, alpha, lower_bound=0.0)

        # the remedy: a small labelled sample from this month, nothing else
        fresh = D.sample(df, cfg["recalibration"]["n_fresh"], seed + 2)
        q_fresh = calibrate(fresh[target].to_numpy(),
                            model.predict(fresh[feats].to_numpy()), alpha)
        recal = coverage_report(y, pred, q_fresh, alpha, lower_bound=0.0)

        rows.append({
            "month": month, "mean_overlap": ov, "n_eval": frozen["n"],
            "mean_fare": float(y.mean()),
            "coverage_frozen": frozen["coverage"],
            "gap_frozen": frozen["coverage_gap"],
            "width_frozen": frozen["mean_width"],
            "coverage_recal": recal["coverage"],
            "width_recal": recal["mean_width"],
            "half_width_recal": q_fresh,
        })
        print(f"  {month}  overlap {ov:.3f}  n={frozen['n']:>6,}  "
              f"mean fare ${y.mean():5.2f}   "
              f"coverage {frozen['coverage']:.4f} (width ${frozen['mean_width']:5.2f})"
              f"   after recalibration {recal['coverage']:.4f} "
              f"(width ${recal['mean_width']:5.2f})", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "coverage.csv", index=False)

    # residual distributions, for the figure that shows why the width stopped
    # being the right width
    resid = {}
    for month in cfg["months"]:
        df = D.load_month(month) if month != cal_month else eval_df
        ev = D.sample(df, min(20000, len(df)), seed + 3)
        r = np.abs(ev[target].to_numpy() - model.predict(ev[feats].to_numpy()))
        resid[month] = np.quantile(r, np.linspace(0, 1, 201)).tolist()
    (OUT / "residual_quantiles.json").write_text(
        json.dumps({"quantiles": np.linspace(0, 1, 201).tolist(),
                    "months": resid, "half_width": q}, indent=2), encoding="utf-8")

    print(f"\nWrote coverage.csv, residual_quantiles.json -> {OUT}")
    worst = res.loc[res["coverage_frozen"].idxmin()]
    print(f"\nworst month: {worst['month']}  coverage "
          f"{worst['coverage_frozen']:.4f} against a promised {1-alpha:.2f} "
          f"- a shortfall of {worst['gap_frozen']*100:.1f} points")


if __name__ == "__main__":
    main()
