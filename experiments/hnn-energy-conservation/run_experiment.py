"""
run_experiment.py -- Train both models on the field, then let them run.

    python run_experiment.py

  A  ideal pendulum: fit, rollout, and the learned H     -> outputs/ideal.csv,
                                                            learned_energy.csv
  B  damped pendulum: fit                                 -> outputs/damped.csv
  C  energy extrapolation: train low, evaluate higher     -> outputs/extrapolation.csv
  D  one rollout per model, for the figures               -> outputs/rollout_example.npz

The control is fatal. If a 4,500-parameter network cannot fit a smooth 2D
field to 1%, nothing downstream measures the architecture.
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

import models as M

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
GEN = HERE.parent.parent / "generators" / "hamiltonian_pendulum" / "outputs"

torch.set_default_dtype(torch.float64)


def load(name: str) -> dict:
    z = np.load(GEN / f"pendulum_{name}.npz")
    # stored as float32 to keep the files small; the models run in float64
    # because the HNN's field is a second derivative and float32 is not
    # enough to measure a 1e-6 energy drift on top of it
    return {k: z[k].astype(np.float64)
            for k in ("t", "q", "p", "dq", "dp", "H", "q0", "p0")} | {
        "dt": float(z["dt"]), "save_every": int(z["save_every"]),
        "gamma": float(z["gamma"])}


def sample_states(d: dict, cols: np.ndarray, n: int, rng, h_max=None):
    """n states drawn from the given trajectories, with their exact field.

    `h_max` restricts to states below an energy, for the extrapolation arm.
    """
    q, p = d["q"][:, cols], d["p"][:, cols]
    dq, dp = d["dq"][:, cols], d["dp"][:, cols]
    H = d["H"][:, cols]
    mask = np.ones(q.shape, bool) if h_max is None else (H < h_max)
    idx = np.flatnonzero(mask.ravel())
    pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
    X = np.stack([q.ravel()[pick], p.ravel()[pick]], -1)
    Y = np.stack([dq.ravel()[pick], dp.ravel()[pick]], -1)
    return torch.tensor(X), torch.tensor(Y)


def field_rmse(model, X, Y) -> float:
    with torch.enable_grad():
        f = model.field(X.clone().requires_grad_(True)).detach()
    return float(((f - Y) ** 2).mean().sqrt())


def fit_pair(Xtr, Ytr, cfg, seed):
    """One MLP and one HNN, same data, same budget, same seed."""
    tr, H = cfg["training"], cfg["network"]["n_hidden"]
    out = {}
    for name, cls in (("mlp", M.MLP), ("hnn", M.HNN)):
        torch.manual_seed(seed)
        m = cls(H)
        M.train(m, Xtr, Ytr, epochs=tr["epochs"], batch=tr["batch"],
                lr=tr["lr"], seed=seed)
        out[name] = m
    return out


def rollout_scores(model, d, cols, cfg):
    """Energy drift and state error of a long rollout from held-out ICs."""
    ro = cfg["rollout"]
    k = d["save_every"]
    n_steps = int(round(ro["t_end"] / ro["dt"]))
    x0 = torch.tensor(np.stack([d["q0"][cols], d["p0"][cols]], -1))
    traj = M.rollout(model, x0, n_steps, ro["dt"], save_every=k)   # (T, N, 2)
    ref = torch.tensor(np.stack([d["q"][:, cols], d["p"][:, cols]], -1))
    n = min(len(traj), len(ref))
    traj, ref = traj[:n], ref[:n]

    H = M.true_energy(traj)
    drift = (H - H[0]).abs()                                  # (T, N)
    # angle error on the circle, so a full rotation does not count as 2 pi off
    dq = torch.remainder(traj[..., 0] - ref[..., 0] + np.pi, 2 * np.pi) - np.pi
    dp = traj[..., 1] - ref[..., 1]
    state_err = (dq ** 2 + dp ** 2).sqrt()
    return {
        "energy_drift_final": float(drift[-1].median()),
        "energy_drift_max": float(drift.max(0).values.median()),
        "state_error_final": float(state_err[-1].median()),
        "state_error_mean": float(state_err.mean(0).median()),
        "blew_up": int((~torch.isfinite(traj[-1]).all(-1)).sum()),
    }, traj, H


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    sp, tr = cfg["split"], cfg["training"]

    ideal, damped = load("ideal"), load("damped")
    n_traj = ideal["q"].shape[1]
    perm = rng.permutation(n_traj)
    train_cols, test_cols = perm[:sp["n_train_trajectories"]], perm[sp["n_train_trajectories"]:]
    roll_cols = test_cols[:cfg["rollout"]["n_initial_conditions"]]
    print(f"{n_traj} trajectories: {len(train_cols)} for training, "
          f"{len(test_cols)} held out, {len(roll_cols)} rolled out\n")

    # ---- arm A: the ideal pendulum -------------------------------------
    print("=== arm A: ideal pendulum ===")
    Xte, Yte = sample_states(ideal, test_cols, tr["n_states"], rng)
    field_rms = float((Yte ** 2).mean().sqrt())
    rows, energy_rows = [], []
    example = {}
    for s in range(tr["seeds"]):
        seed = cfg["seed"] + s
        Xtr, Ytr = sample_states(ideal, train_cols, tr["n_states"],
                                 np.random.default_rng(seed))
        t0 = time.time()
        models = fit_pair(Xtr, Ytr, cfg, seed)
        for name, m in models.items():
            rmse = field_rmse(m, Xte, Yte)
            scores, traj, H = rollout_scores(m, ideal, roll_cols, cfg)
            rows.append({"seed": s, "model": name, "field_rmse": rmse,
                         "field_rmse_rel": rmse / field_rms, **scores})
            if s == 0:
                example[name] = (traj.numpy(), H.numpy())
            print(f"  seed {s} {name}: field rmse {rmse:.2e} "
                  f"({rmse / field_rms:.2%} of field RMS)   "
                  f"energy drift at t=100: {scores['energy_drift_final']:.2e}   "
                  f"state error: {scores['state_error_final']:.3f}", flush=True)

        # P3: the learned energy against the true one, on the training region
        h = models["hnn"]
        with torch.no_grad():
            Ht = h.energy(Xtr).numpy()
        Htrue = M.true_energy(Xtr).numpy()
        A = np.stack([Htrue, np.ones_like(Htrue)], 1)
        (a, b), *_ = np.linalg.lstsq(A, Ht, rcond=None)
        resid = Ht - (a * Htrue + b)
        r2 = 1 - resid.var() / Ht.var()
        energy_rows.append({"seed": s, "slope": float(a), "intercept": float(b),
                            "r2": float(r2)})
        print(f"  seed {s} learned H_theta = {a:.4f} H + {b:+.4f}, R2 {r2:.5f}"
              f"   [{time.time() - t0:.0f}s]", flush=True)
        if s == 0:
            # the learned energy on a grid, for the level-set figure; and the
            # weights, so the figure can be redrawn without retraining
            qq, pp = np.meshgrid(np.linspace(-np.pi, np.pi, 121),
                                 np.linspace(-2.6, 2.6, 121))
            grid = torch.tensor(np.stack([qq.ravel(), pp.ravel()], -1))
            with torch.no_grad():
                Hg = h.energy(grid).numpy().reshape(qq.shape)
            np.savez_compressed(OUT / "learned_energy_grid.npz", q=qq, p=pp,
                                H_theta=Hg, H_true=M.true_energy(grid).numpy().reshape(qq.shape),
                                slope=a, intercept=b)
            torch.save({k: v.state_dict() for k, v in models.items()},
                       OUT / "models_seed0.pt")

    A_df = pd.DataFrame(rows)
    A_df.to_csv(OUT / "ideal.csv", index=False)
    pd.DataFrame(energy_rows).to_csv(OUT / "learned_energy.csv", index=False)

    # ---- P1: the control ------------------------------------------------
    med = A_df.groupby("model")["field_rmse_rel"].median()
    print(f"\ncontrol: median held-out field RMSE  mlp {med['mlp']:.2%}  "
          f"hnn {med['hnn']:.2%} of field RMS")
    if not (med < 0.01).all():
        sys.exit("\nFAIL - a model could not fit the ideal field to 1%. The "
                 "setup is broken and nothing below measures the architecture.")
    print("  OK - both fit the field\n")

    # ---- arm B: the damped pendulum ------------------------------------
    print("=== arm B: damped pendulum (the inductive bias is now wrong) ===")
    Xte_d, Yte_d = sample_states(damped, test_cols, tr["n_states"], rng)
    rms_d = float((Yte_d ** 2).mean().sqrt())
    rows = []
    for s in range(tr["seeds"]):
        seed = cfg["seed"] + s
        Xtr, Ytr = sample_states(damped, train_cols, tr["n_states"],
                                 np.random.default_rng(seed))
        models = fit_pair(Xtr, Ytr, cfg, seed)
        for name, m in models.items():
            rmse = field_rmse(m, Xte_d, Yte_d)
            rows.append({"seed": s, "model": name, "field_rmse": rmse,
                         "field_rmse_rel": rmse / rms_d})
        print(f"  seed {s}: field rmse  mlp {rows[-2]['field_rmse']:.2e}   "
              f"hnn {rows[-1]['field_rmse']:.2e}   "
              f"ratio {rows[-1]['field_rmse'] / rows[-2]['field_rmse']:.1f}x",
              flush=True)
    B_df = pd.DataFrame(rows)
    B_df.to_csv(OUT / "damped.csv", index=False)

    # ---- arm C: extrapolation in energy --------------------------------
    print("\n=== arm C: train on H < 0.5, evaluate on 0.5 < H < 0.9 ===")
    lo, hi = sp["energy_test_range"]
    H0 = ideal["H"][0]
    ext_cols = np.array([c for c in test_cols if lo < H0[c] < hi])
    # Post-hoc extension, disclosed in README. The pre-registered set is the
    # held-out trajectories in the band, and there are only two. But energy
    # is conserved, so a trajectory with H0 > 0.5 contributes no state at all
    # to a training set restricted to H < 0.5 - every trajectory in the band
    # is unseen by these models, whichever split it came from. Both sets are
    # evaluated and both are reported; P5 is scored on the pre-registered one.
    all_cols = np.flatnonzero((H0 > lo) & (H0 < hi))
    print(f"  {len(ext_cols)} held-out trajectories in the evaluation band "
          f"(pre-registered); {len(all_cols)} in the band overall (extension)")
    rows = []
    for s in range(tr["seeds"]):
        seed = cfg["seed"] + s
        Xtr, Ytr = sample_states(ideal, train_cols, tr["n_states"],
                                 np.random.default_rng(seed),
                                 h_max=sp["energy_train_max"])
        assert float(M.true_energy(Xtr).max()) < sp["energy_train_max"]
        models = fit_pair(Xtr, Ytr, cfg, seed)
        Xe, Ye = sample_states(ideal, ext_cols, tr["n_states"],
                               np.random.default_rng(seed + 100))
        for name, m in models.items():
            rmse = field_rmse(m, Xe, Ye)
            for set_name, cols in (("pre-registered", ext_cols), ("extended", all_cols)):
                scores, _, _ = rollout_scores(m, ideal, cols, cfg)
                rows.append({"seed": s, "model": name, "set": set_name,
                             "n_trajectories": len(cols), "field_rmse": rmse, **scores})
        pre = {r["model"]: r for r in rows if r["seed"] == s and r["set"] == "pre-registered"}
        ext = {r["model"]: r for r in rows if r["seed"] == s and r["set"] == "extended"}
        print(f"  seed {s}: state error at t=100  pre-registered (n={len(ext_cols)}): "
              f"mlp {pre['mlp']['state_error_final']:.3f}  hnn {pre['hnn']['state_error_final']:.3f}"
              f"   extended (n={len(all_cols)}): "
              f"mlp {ext['mlp']['state_error_final']:.3f}  hnn {ext['hnn']['state_error_final']:.3f}",
              flush=True)
    C_df = pd.DataFrame(rows)
    C_df.to_csv(OUT / "extrapolation.csv", index=False)

    # ---- arm D: one example rollout, for the figures --------------------
    np.savez_compressed(OUT / "rollout_example.npz",
                        t=ideal["t"][:len(example["mlp"][0])],
                        mlp_traj=example["mlp"][0], mlp_H=example["mlp"][1],
                        hnn_traj=example["hnn"][0], hnn_H=example["hnn"][1],
                        ref_q=ideal["q"][:, roll_cols], ref_p=ideal["p"][:, roll_cols],
                        ref_H=ideal["H"][:, roll_cols])

    # ---- summary --------------------------------------------------------
    a = A_df.groupby("model").median(numeric_only=True)
    b = B_df.groupby("model").median(numeric_only=True)
    c = C_df[C_df["set"] == "pre-registered"].groupby("model").median(numeric_only=True)
    c_ext = C_df[C_df["set"] == "extended"].groupby("model").median(numeric_only=True)
    e = pd.DataFrame(energy_rows).median(numeric_only=True)
    summary = {
        "field_rms_ideal": field_rms,
        "P1_field_rmse_rel": {"mlp": float(a.loc["mlp", "field_rmse_rel"]),
                              "hnn": float(a.loc["hnn", "field_rmse_rel"])},
        "P2_energy_drift_final": {"mlp": float(a.loc["mlp", "energy_drift_final"]),
                                  "hnn": float(a.loc["hnn", "energy_drift_final"])},
        "P2_ratio": float(a.loc["mlp", "energy_drift_final"] / a.loc["hnn", "energy_drift_final"]),
        "P3_learned_energy": {"slope": float(e["slope"]), "r2": float(e["r2"])},
        "P4_damped_field_rmse": {"mlp": float(b.loc["mlp", "field_rmse"]),
                                 "hnn": float(b.loc["hnn", "field_rmse"])},
        "P4_ratio": float(b.loc["hnn", "field_rmse"] / b.loc["mlp", "field_rmse"]),
        "P5_extrapolation_state_error": {"mlp": float(c.loc["mlp", "state_error_final"]),
                                         "hnn": float(c.loc["hnn", "state_error_final"]),
                                         "n_trajectories": int(c["n_trajectories"].iloc[0])},
        "P5_extended_posthoc": {"mlp": float(c_ext.loc["mlp", "state_error_final"]),
                                "hnn": float(c_ext.loc["hnn", "state_error_final"]),
                                "n_trajectories": int(c_ext["n_trajectories"].iloc[0])},
        "P2_energy_drift_max_over_time": {"mlp": float(a.loc["mlp", "energy_drift_max"]),
                                          "hnn": float(a.loc["hnn", "energy_drift_max"])},
        "ideal_state_error_final": {"mlp": float(a.loc["mlp", "state_error_final"]),
                                    "hnn": float(a.loc["hnn", "state_error_final"])},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nmedians over seeds")
    print(f"  P2  energy drift at t=100:  mlp {summary['P2_energy_drift_final']['mlp']:.2e}  "
          f"hnn {summary['P2_energy_drift_final']['hnn']:.2e}  ratio {summary['P2_ratio']:.0f}x")
    print(f"  P3  H_theta = {e['slope']:.4f} H + c,  R2 {e['r2']:.5f}")
    print(f"  P4  damped field rmse:  hnn / mlp = {summary['P4_ratio']:.1f}x")
    print(f"  P5  extrapolation state error at t=100 (pre-registered, "
          f"n={summary['P5_extrapolation_state_error']['n_trajectories']}):  "
          f"mlp {summary['P5_extrapolation_state_error']['mlp']:.3f}  "
          f"hnn {summary['P5_extrapolation_state_error']['hnn']:.3f}")
    print(f"      extended post-hoc set (n={summary['P5_extended_posthoc']['n_trajectories']}):  "
          f"mlp {summary['P5_extended_posthoc']['mlp']:.3f}  "
          f"hnn {summary['P5_extended_posthoc']['hnn']:.3f}")
    print(f"  P2, max-over-time reading:  mlp {summary['P2_energy_drift_max_over_time']['mlp']:.2e}  "
          f"hnn {summary['P2_energy_drift_max_over_time']['hnn']:.2e}  ratio "
          f"{summary['P2_energy_drift_max_over_time']['mlp']/summary['P2_energy_drift_max_over_time']['hnn']:.1f}x")
    print(f"\nWrote ideal.csv, learned_energy.csv, damped.csv, extrapolation.csv, "
          f"rollout_example.npz, summary.json -> {OUT}")


if __name__ == "__main__":
    main()
