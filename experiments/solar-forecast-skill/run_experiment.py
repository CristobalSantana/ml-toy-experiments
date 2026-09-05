"""
run_experiment.py -- Fit everything, score it two ways, and report the gap.

    python run_experiment.py

For each horizon the same table is produced: R2 over all hours, R2 over
daylight hours only, and skill against smart persistence. The distance between
the first and the last columns is the whole point of the experiment.

Nothing is tuned on the test year. Any choice that had to be made was made on
2018 and is recorded in config.yaml.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import features as F
from models import (MODELS, ClearSky, baseline_climatology,
                    baseline_smart_persistence, baseline_train_mean)

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATASET = HERE.parents[1] / "datasets" / "de" / "opsd_solar"
sys.path.insert(0, str(DATASET))
from load import load as load_solar          # noqa: E402


def r2(y: np.ndarray, p: np.ndarray) -> float:
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(((y - p) ** 2).mean()))


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    seed = cfg["seed"]
    sp, cs_cfg = cfg["splits"], cfg["clear_sky"]
    OUT.mkdir(parents=True, exist_ok=True)

    df = load_solar(verbose=True)
    df = F.add_time_parts(df)

    # ---- clear sky, fitted on the training split only ------------------
    train_mask = df["time"] <= sp["train_end"]
    cs = ClearSky(bin_deg=cs_cfg["elevation_bin_deg"],
                  quantile=cs_cfg["quantile"]).fit(
        df.loc[train_mask, "elevation"].to_numpy(),
        df.loc[train_mask, "cf"].to_numpy())
    df["cs"] = cs.predict(df["elevation"].to_numpy())
    df["k"] = cs.index(df["elevation"].to_numpy(), df["cf"].to_numpy())
    print(f"\nclear-sky envelope fitted on {int(train_mask.sum()):,} training hours "
          f"(q{cs_cfg['quantile']:.2f} per {cs_cfg['elevation_bin_deg']:g}-degree bin)")

    day_thr = cfg["daylight_elevation_deg"]
    rows, curves = [], []

    for h in cfg["horizons"]:
        frame = F.build(df, h, cfg["features"]["cf_lags"],
                        cfg["features"]["use_clear_sky_index"],
                        cfg["features"]["use_target_geometry"])
        tt = pd.DatetimeIndex(frame["target_time"])
        frame["target_doy"] = tt.dayofyear

        tr = frame["target_time"] <= sp["train_end"]
        va = (frame["target_time"] > sp["train_end"]) & (frame["target_time"] <= sp["val_end"])
        te = (frame["target_time"] > sp["val_end"])
        print(f"\n=== horizon {h}h ===")
        print(f"  train {int(tr.sum()):,}   val {int(va.sum()):,}   test {int(te.sum()):,}")

        cols = F.feature_columns(frame)
        Xtr, ytr = frame.loc[tr, cols].to_numpy(), frame.loc[tr, "y"].to_numpy()
        Xte, yte = frame.loc[te, cols].to_numpy(), frame.loc[te, "y"].to_numpy()
        test = frame.loc[te]
        day = (test["target_elevation"] > day_thr).to_numpy()

        preds: dict[str, np.ndarray] = {}

        # ---- baselines --------------------------------------------------
        preds["B0 train mean"] = baseline_train_mean(ytr, len(yte))
        clim_train = pd.DataFrame({"doy": pd.DatetimeIndex(
            frame.loc[tr, "target_time"]).dayofyear,
            "hour": frame.loc[tr, "target_hour"].to_numpy(),
            "cf": ytr})
        preds["B1 climatology"] = baseline_climatology(
            clim_train, pd.DataFrame({"doy": test["target_doy"].to_numpy(),
                                      "hour": test["target_hour"].to_numpy()}))
        preds["B2 smart persistence"] = baseline_smart_persistence(
            test["cf_now"].to_numpy(), test["cs_now"].to_numpy(),
            test["target_cs"].to_numpy())

        # ---- learned models ---------------------------------------------
        timings = {}
        for key, spec in cfg["models"].items():
            t0 = time.perf_counter()
            m = MODELS[key](seed=seed, **spec).fit(Xtr, ytr)
            timings[key] = time.perf_counter() - t0
            preds[f"M {key}"] = m.predict(Xte)
            print(f"  fitted {key:<6} in {timings[key]:>6.1f} s", flush=True)

        mse_b2 = float(((yte - preds["B2 smart persistence"]) ** 2).mean())
        mse_b1 = float(((yte - preds["B1 climatology"]) ** 2).mean())

        for name, p in preds.items():
            mse = float(((yte - p) ** 2).mean())
            rows.append({
                "horizon": h, "model": name,
                "r2_all": r2(yte, p),
                "r2_day": r2(yte[day], p[day]),
                "nrmse_all": rmse(yte, p),
                "nrmse_day": rmse(yte[day], p[day]),
                "skill_vs_persistence": 1.0 - mse / mse_b2,
                "skill_vs_climatology": 1.0 - mse / mse_b1,
                "fit_seconds": timings.get(name.replace("M ", ""), float("nan")),
                "n_test": int(len(yte)), "n_test_day": int(day.sum()),
            })

        # a fortnight of the test year, for the figure that shows what the
        # numbers actually look like hour by hour
        if h == 1:
            sel = (test["target_time"] >= "2019-06-10") & (test["target_time"] < "2019-06-17")
            c = pd.DataFrame({"time": test.loc[sel, "target_time"].to_numpy(),
                              "actual": yte[sel.to_numpy()]})
            for name, p in preds.items():
                c[name] = p[sel.to_numpy()]
            curves.append(c)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "scores.csv", index=False)
    if curves:
        curves[0].to_csv(OUT / "week_h1.csv", index=False)

    # the clear-sky curve itself, for the figure and for anyone checking it
    el = np.arange(-5, 65, 0.5)
    pd.DataFrame({"elevation": el, "clear_sky_cf": cs.predict(el)}).to_csv(
        OUT / "clear_sky.csv", index=False)

    # a small summary of what the data is made of, quoted in the write-up
    (OUT / "data_summary.json").write_text(json.dumps({
        "hours": int(len(df)),
        "first": str(df["time"].min()), "last": str(df["time"].max()),
        "night_share": float((df["elevation"] <= day_thr).mean()),
        "zero_share": float((df["cf"] == 0).mean()),
        "mean_cf": float(df["cf"].mean()),
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    for h in cfg["horizons"]:
        print(f"\nHORIZON {h}h        R2 all    R2 day   skill vs persistence")
        for _, r in res[res.horizon == h].iterrows():
            print(f"  {r['model']:<24} {r['r2_all']:>7.3f}  {r['r2_day']:>7.3f}"
                  f"  {r['skill_vs_persistence']:>10.3f}")
    print(f"\nWrote scores.csv, week_h1.csv, clear_sky.csv -> {OUT}")


if __name__ == "__main__":
    main()
