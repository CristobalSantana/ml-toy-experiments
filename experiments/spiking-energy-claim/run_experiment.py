"""
run_experiment.py -- Match a spiking network to a dense one, then count what
each of them actually did.

    python run_experiment.py

  1. baselines and the dense control        -> outputs/baselines.csv
  2. the T sweep, both encodings, 3 seeds   -> outputs/sweep.csv
  3. the accounting, under two conventions  -> outputs/accounting.json

The control is fatal. If the dense network cannot beat smart persistence,
the task is not a forecasting task and matching a spiking network to it
would be matching two things that are both useless.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge

import data as D
from snn import MLP, SNN, rmse, train

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def as_tensors(*arrays):
    return [torch.from_numpy(np.ascontiguousarray(a)) for a in arrays]


def evaluate(model, X, y, day) -> float:
    """Daylight RMSE. Night hours are excluded because predicting zero in the
    dark is not forecasting, and including them would let any model look good
    by getting the easy half right."""
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).numpy()
    return rmse(pred[day], y[day])


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)

    net, tr_cfg = cfg["network"], cfg["training"]
    H = net["n_hidden"]

    print("task")
    task_std = D.make_task(cfg["splits"], scaling="standard")
    task_mm = D.make_task(cfg["splits"], scaling="minmax", verbose=False)
    n_in = task_std.n_features
    day = task_std.day_te
    print()

    # ---- baselines ------------------------------------------------------
    rows = []
    for name, pred in D.baselines(task_std).items():
        rows.append({"model": name, "rmse": rmse(pred[day], task_std.yte[day]),
                     "ops_total": 0.0, "note": "reference"})

    r = Ridge(alpha=1.0).fit(task_std.Xtr, task_std.ytr)
    rows.append({"model": "ridge",
                 "rmse": rmse(r.predict(task_std.Xte)[day], task_std.yte[day]),
                 "ops_total": float(n_in), "note": "reference"})

    # ---- the dense control ----------------------------------------------
    Xtr, ytr, Xva, yva = as_tensors(task_std.Xtr, task_std.ytr,
                                    task_std.Xva, task_std.yva)
    mlp_rmse, mlp_ops = [], None
    for s in range(cfg["sweep"]["seeds"]):
        torch.manual_seed(cfg["seed"] + s)
        m = train(MLP(n_in, H), Xtr, ytr, Xva, yva, seed=cfg["seed"] + s,
                  epochs=tr_cfg["epochs"], batch=tr_cfg["batch"],
                  lr=tr_cfg["lr"], patience=tr_cfg["patience"])
        mlp_rmse.append(evaluate(m, task_std.Xte, task_std.yte, day))
        mlp_ops = m.count_ops(None)
    mlp_med = float(np.median(mlp_rmse))
    rows.append({"model": "mlp", "rmse": mlp_med,
                 "ops_total": mlp_ops.total, "note": "dense control"})

    base = pd.DataFrame(rows)
    base.to_csv(OUT / "baselines.csv", index=False)
    print("references and the dense control (daylight RMSE)")
    for _, x in base.iterrows():
        print(f"  {x['model']:<20} {x['rmse']:.5f}   "
              f"{x['ops_total']:>8,.0f} operations")

    sp = float(base.loc[base["model"] == "smart_persistence", "rmse"].iloc[0])
    if not mlp_med < sp:
        sys.exit(
            f"\nFAIL - the dense network ({mlp_med:.5f}) does not beat smart "
            f"persistence ({sp:.5f}). The task is not a forecasting task, and "
            f"matching a spiking network to it would mean nothing.")
    print(f"  OK - the dense control beats smart persistence by "
          f"{(1 - mlp_med / sp) * 100:.1f}%\n")

    # ---- the sweep ------------------------------------------------------
    print("spiking sweep (daylight RMSE, and operations per prediction)")
    tol = cfg["matching"]["rmse_tolerance"]
    sweep = []
    for enc in cfg["sweep"]["encodings"]:
        task = task_std if enc == "direct" else task_mm
        Xtr, ytr, Xva, yva = as_tensors(task.Xtr, task.ytr, task.Xva, task.yva)
        Xte = torch.from_numpy(task.Xte)
        for T in cfg["sweep"]["n_steps"]:
            started = time.time()
            for s in range(cfg["sweep"]["seeds"]):
                torch.manual_seed(cfg["seed"] + s)
                g = torch.Generator().manual_seed(cfg["seed"] + s)
                m = SNN(n_in, H, T, beta=net["beta"],
                        threshold=net["threshold"], encoding=enc)
                m = train(m, Xtr, ytr, Xva, yva, seed=cfg["seed"] + s,
                          epochs=tr_cfg["epochs"], batch=tr_cfg["batch"],
                          lr=tr_cfg["lr"], patience=tr_cfg["patience"])
                with torch.no_grad():
                    pred = m(Xte).numpy()
                ops = m.count_ops(Xte[:2000], g)
                sweep.append({"encoding": enc, "n_steps": T, "seed": s,
                              "rmse": rmse(pred[day], task.yte[day]),
                              **ops.as_dict()})
            d = pd.DataFrame(sweep)
            d = d[(d["encoding"] == enc) & (d["n_steps"] == T)]
            med = d["rmse"].median()
            print(f"  {enc:<7} T={T:>3}  rmse {med:.5f} "
                  f"({med / mlp_med:5.2f}x the dense net)   "
                  f"{d['ops_total'].median():>9,.0f} operations   "
                  f"hidden spike rate {d['spike_rate'].median():.3f}   "
                  f"[{time.time() - started:.0f}s]", flush=True)

    sw = pd.DataFrame(sweep)
    sw.to_csv(OUT / "sweep.csv", index=False)

    # ---- the accounting -------------------------------------------------
    acc = {"mlp_rmse": mlp_med, "mlp_ops_total": mlp_ops.total,
           "mlp_ops_input": mlp_ops.input_layer,
           "mlp_ops_hidden": mlp_ops.hidden,
           "smart_persistence_rmse": sp, "tolerance": tol, "encodings": {}}

    print("\nmatched comparison")
    for enc in cfg["sweep"]["encodings"]:
        g = (sw[sw["encoding"] == enc].groupby("n_steps")
             .median(numeric_only=True).reset_index())
        ok = g[g["rmse"] <= mlp_med * (1 + tol)]
        if not len(ok):
            print(f"  {enc}: never reached the dense network's accuracy")
            acc["encodings"][enc] = {"matched": False,
                                     "best_rmse": float(g["rmse"].min())}
            continue
        row = ok.iloc[0]                      # smallest T that matches
        acc["encodings"][enc] = {
            "matched": True,
            "n_steps": int(row["n_steps"]),
            "rmse": float(row["rmse"]),
            "ops_total": float(row["ops_total"]),
            "ops_input": float(row["ops_input"]),
            "ops_hidden": float(row["ops_hidden"]),
            "spike_rate": float(row["spike_rate"]),
            "input_spike_rate": float(row["input_spike_rate"]),
            # P3: everything counted
            "ratio_total": float(row["ops_total"] / mlp_ops.total),
            # P5: only the sparse layer counted, as energy comparisons often do
            "ratio_hidden_only": float(row["ops_hidden"] / mlp_ops.hidden),
        }
        a = acc["encodings"][enc]
        print(f"  {enc}: matches at T={a['n_steps']} "
              f"(rmse {a['rmse']:.5f} against {mlp_med:.5f})")
        print(f"    counting everything:        "
              f"{a['ops_total']:>9,.0f} against {mlp_ops.total:>7,.0f}"
              f"   = {a['ratio_total']:.2f}x the dense network")
        print(f"    counting the sparse layer:  "
              f"{a['ops_hidden']:>9,.0f} against {mlp_ops.hidden:>7,.0f}"
              f"   = {a['ratio_hidden_only']:.2f}x")

    # ---- best effort (post-hoc) -----------------------------------------
    # Added after the sweep showed no T reaching the dense network's accuracy
    # at the frozen width. If a spiking network with eight times the width and
    # the longest T still loses, the gap is the architecture rather than the
    # budget - and that is worth knowing even though nothing is scored on it.
    be = cfg["best_effort"]
    print(f"\nbest effort (post-hoc): H={be['n_hidden']}, T={be['n_steps']}, "
          f"{be['encoding']} coding")
    task = task_std if be["encoding"] == "direct" else task_mm
    Xtr, ytr, Xva, yva = as_tensors(task.Xtr, task.ytr, task.Xva, task.yva)
    Xte = torch.from_numpy(task.Xte)
    be_rows = []
    for s in range(cfg["sweep"]["seeds"]):
        torch.manual_seed(cfg["seed"] + s)
        g = torch.Generator().manual_seed(cfg["seed"] + s)
        m = SNN(n_in, be["n_hidden"], be["n_steps"], beta=net["beta"],
                threshold=net["threshold"], encoding=be["encoding"])
        m = train(m, Xtr, ytr, Xva, yva, seed=cfg["seed"] + s,
                  epochs=tr_cfg["epochs"], batch=tr_cfg["batch"],
                  lr=tr_cfg["lr"], patience=tr_cfg["patience"])
        with torch.no_grad():
            pred = m(Xte).numpy()
        ops = m.count_ops(Xte[:2000], g)
        be_rows.append({"seed": s, "rmse": rmse(pred[day], task.yte[day]),
                        **ops.as_dict()})
        print(f"    seed {s}: rmse {be_rows[-1]['rmse']:.5f}", flush=True)
    b = pd.DataFrame(be_rows)
    b.to_csv(OUT / "best_effort.csv", index=False)
    acc["best_effort"] = {
        "n_hidden": be["n_hidden"], "n_steps": be["n_steps"],
        "encoding": be["encoding"],
        "rmse": float(b["rmse"].median()),
        "ops_total": float(b["ops_total"].median()),
        "spike_rate": float(b["spike_rate"].median()),
        "ratio_rmse": float(b["rmse"].median() / mlp_med),
        "ratio_ops": float(b["ops_total"].median() / mlp_ops.total),
    }
    a = acc["best_effort"]
    print(f"  rmse {a['rmse']:.5f} = {a['ratio_rmse']:.2f}x the dense network, "
          f"at {a['ops_total']:,.0f} operations = {a['ratio_ops']:.0f}x")

    (OUT / "accounting.json").write_text(json.dumps(acc, indent=2),
                                         encoding="utf-8")
    print(f"\nWrote baselines.csv, sweep.csv, best_effort.csv, "
          f"accounting.json -> {OUT}")


if __name__ == "__main__":
    main()
