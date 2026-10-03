"""
run_experiment.py -- Train, evaluate on four grids, score the predictions.

    python run_experiment.py

Twenty-seven trainings - three viscosities, three seeds, and three
(model, training grid) pairs: the FNO and the CNN at 64 points, the FNO at
256 - farmed out three at a time with two threads each. The control (P1)
trains first and is checked before anything else runs.

Writes to outputs/:
  errors.csv        one row per training x evaluation grid x mode
  curves.csv        training loss per epoch
  examples.npz      a few held-out predictions, for the figures
  summary.json      every prediction, scored
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

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATA = HERE.parent.parent / "generators" / "burgers_1d" / "outputs" / "burgers.npz"
STORED = 512
N_EXAMPLES = 16


def load_cfg() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def load_data() -> dict:
    z = np.load(DATA)
    return {k: z[k] for k in z.files}


def on_grid(a: np.ndarray, n: int) -> torch.Tensor:
    """A nested subsample of the stored 512-point grid."""
    return torch.tensor(a[:, ::STORED // n], dtype=torch.float32)


def interpolate_linear(u: torch.Tensor, n: int) -> torch.Tensor:
    """Periodic linear interpolation from u's grid onto an n-point grid."""
    m = u.shape[-1]
    pos = torch.arange(n, dtype=torch.float64) * m / n
    i0 = torch.floor(pos).long()
    w = (pos - i0).to(u.dtype)
    return u[..., i0 % m] * (1 - w) + u[..., (i0 + 1) % m] * w


def interpolate_fourier(u: torch.Tensor, n: int) -> torch.Tensor:
    """Trigonometric interpolation: pad the spectrum with zeros. Reported
    beside the linear one; not what P4 is scored on."""
    m = u.shape[-1]
    uh = torch.fft.rfft(u.double())
    out = torch.zeros(u.shape[0], n // 2 + 1, dtype=uh.dtype)
    out[:, :m // 2] = uh[:, :m // 2]
    out[:, m // 2] = uh[:, m // 2] / 2            # split the Nyquist mode
    return (torch.fft.irfft(out, n=n) * n / m).to(u.dtype)


# --------------------------------------------------------------------------
# one training run
# --------------------------------------------------------------------------

def run_one(task: dict) -> dict:
    sys.path.insert(0, str(HERE))
    import models as M
    torch.set_num_threads(2)
    cfg, nu, seed, name, res = task["cfg"], task["nu"], task["seed"], task["model"], task["res"]
    tr = cfg["training"]
    d = load_data()
    n_train = cfg["data"]["n_train"]
    u0, uT = d["u0"], d[f"uT_nu{nu:g}"]
    x_tr, y_tr = on_grid(u0[:n_train], res), on_grid(uT[:n_train], res)

    torch.manual_seed(cfg["seed"] + 1000 * seed)
    model = M.build(name, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=tr["lr_halve_every"], gamma=0.5)
    g = torch.Generator().manual_seed(cfg["seed"] + 1000 * seed + 7)
    started, curve = time.time(), []
    for epoch in range(tr["epochs"]):
        perm = torch.randperm(n_train, generator=g)
        total = 0.0
        model.train()
        for i in range(0, n_train, tr["batch"]):
            idx = perm[i:i + tr["batch"]]
            opt.zero_grad()
            loss = M.relative_l2(model(x_tr[idx]), y_tr[idx]).mean()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        sched.step()
        curve.append(total / n_train)
    seconds = time.time() - started

    model.eval()
    rows, examples = [], {}
    with torch.no_grad():
        own = model(on_grid(u0[n_train:], res))                 # on its own grid
        for n in cfg["data"]["eval_resolutions"]:
            truth = on_grid(uT[n_train:], n)
            zero_shot = model(on_grid(u0[n_train:], n))
            preds = {"zero_shot": zero_shot}
            if n > res:
                preds["interp_linear"] = interpolate_linear(own, n)
                preds["interp_fourier"] = interpolate_fourier(own, n)
            for mode, p in preds.items():
                e = M.relative_l2(p, truth)
                rows.append({"nu": nu, "seed": seed, "model": name, "train_res": res,
                             "eval_res": n, "mode": mode, "error": float(e.mean()),
                             "error_median": float(e.median())})
            if seed == 0:
                examples[f"{name}{res}_zero_shot_{n}"] = zero_shot[:N_EXAMPLES].numpy()
    return {"rows": rows, "examples": examples, "nu": nu, "seed": seed, "model": name,
            "res": res, "curve": curve, "seconds": seconds,
            "params": M.n_params(model)}


def floors(cfg: dict, d: dict) -> list[dict]:
    """What a perfect 64-point answer would score on a finer grid, if it
    were interpolated: the truth itself, sampled at 64 and interpolated.
    No model involved."""
    n_train, rows = cfg["data"]["n_train"], []
    for nu in cfg["data"]["viscosities"]:
        coarse = on_grid(d[f"uT_nu{nu:g}"][n_train:], cfg["data"]["train_resolution"])
        for n in [r for r in cfg["data"]["eval_resolutions"]
                  if r > cfg["data"]["train_resolution"]]:
            truth = on_grid(d[f"uT_nu{nu:g}"][n_train:], n)
            for mode, f in (("interp_linear", interpolate_linear),
                            ("interp_fourier", interpolate_fourier)):
                diff = f(coarse, n) - truth
                e = torch.linalg.vector_norm(diff, dim=-1) / torch.linalg.vector_norm(truth, dim=-1)
                rows.append({"nu": nu, "eval_res": n, "mode": mode,
                             "error": float(e.mean())})
    return rows


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def med(df: pd.DataFrame, **where) -> float:
    sel = df
    for k, v in where.items():
        sel = sel[sel[k] == v]
    per_seed = sel.groupby("seed")["error"].first()
    assert len(per_seed) > 0, where
    return float(per_seed.median())


def score(df: pd.DataFrame, cfg: dict) -> dict:
    lo, hi = cfg["data"]["train_resolution"], cfg["data"]["high_resolution"]
    nus = cfg["data"]["viscosities"]
    smooth, sharp = max(nus), min(nus)
    fno = lambda nu, n, mode="zero_shot", res=lo: med(df, nu=nu, model="fno", train_res=res,
                                                      eval_res=n, mode=mode)
    cnn = lambda nu, n, mode="zero_shot": med(df, nu=nu, model="cnn", train_res=lo,
                                              eval_res=n, mode=mode)
    s = {}
    p1 = fno(smooth, lo)
    s["P1"] = {"fno_error_64": p1, "passed": p1 < 0.01}
    ratios = {f"{nu:g}": fno(nu, lo) / cnn(nu, lo) for nu in nus}
    s["P2"] = {"fno_over_cnn_at_64": ratios,
               "fno": {f"{nu:g}": fno(nu, lo) for nu in nus},
               "cnn": {f"{nu:g}": cnn(nu, lo) for nu in nus},
               "passed": all(r <= 0.5 for r in ratios.values())}
    f_ratio = fno(smooth, hi) / fno(smooth, lo)
    c_ratio = cnn(smooth, hi) / cnn(smooth, lo)
    s["P3"] = {"fno_256_over_64": f_ratio, "cnn_256_over_64": c_ratio,
               "passed": f_ratio <= 1.25 and c_ratio >= 3}
    zs, it = fno(sharp, hi), fno(sharp, hi, "interp_linear")
    s["P4"] = {"zero_shot": zs, "interp_linear": it,
               "interp_fourier": fno(sharp, hi, "interp_fourier"),
               "zero_shot_over_interp": zs / it, "passed": zs >= 0.9 * it}
    trained_hi = fno(sharp, hi, res=hi)
    s["P5"] = {"trained_at_256": trained_hi, "zero_shot_from_64": zs,
               "ratio": trained_hi / zs, "passed": trained_hi <= 0.5 * zs}
    return s


def main() -> None:
    from joblib import Parallel, delayed
    cfg = load_cfg()
    OUT.mkdir(exist_ok=True)
    d = load_data()
    lo, hi = cfg["data"]["train_resolution"], cfg["data"]["high_resolution"]
    seeds = range(cfg["training"]["seeds"])
    nus = cfg["data"]["viscosities"]
    smooth = max(nus)
    pairs = [("fno", lo), ("cnn", lo), ("fno", hi)]
    tasks = [{"cfg": cfg, "nu": nu, "seed": s, "model": m, "res": r}
             for nu in nus for s in seeds for m, r in pairs]
    control = [t for t in tasks if t["nu"] == smooth and t["model"] == "fno" and t["res"] == lo]
    rest = [t for t in tasks if t not in control]
    started = time.time()

    def label(r):
        return f"nu {r['nu']:<5g} seed {r['seed']}  {r['model']} @ {r['res']:<3}"

    print(f"control first: {len(control)} trainings", flush=True)
    results = Parallel(n_jobs=3, verbose=0)(delayed(run_one)(t) for t in control)
    for r in results:
        own = [x for x in r["rows"] if x["eval_res"] == lo and x["mode"] == "zero_shot"][0]
        print(f"  {label(r)}  error at 64: {own['error']:.4f}   ({r['seconds'] / 60:.1f} min)")
    p1 = float(np.median([[x for x in r["rows"] if x["eval_res"] == lo][0]["error"]
                          for r in results]))
    if p1 >= 0.01:
        sys.exit(f"\nCONTROL FAILED (P1): FNO error at 64 points, nu = {smooth:g}, is "
                 f"{p1:.4f} - at least 1%. The setup is broken; stopping.")
    print(f"  P1 control passed: median {p1:.4f}\n", flush=True)

    print(f"the other {len(rest)} trainings, three at a time", flush=True)
    for r in Parallel(n_jobs=3, verbose=0, return_as="generator")(delayed(run_one)(t)
                                                                  for t in rest):
        own = [x for x in r["rows"] if x["eval_res"] == r["res"] and x["mode"] == "zero_shot"][0]
        print(f"  {label(r)}  error on own grid: {own['error']:.4f}   "
              f"({r['seconds'] / 60:.1f} min)", flush=True)
        results.append(r)

    df = pd.DataFrame([row for r in results for row in r["rows"]])
    df.to_csv(OUT / "errors.csv", index=False)
    pd.DataFrame(floors(cfg, d)).to_csv(OUT / "floors.csv", index=False)
    pd.DataFrame([{"nu": r["nu"], "seed": r["seed"], "model": r["model"],
                   "train_res": r["res"], "epoch": e, "loss": l}
                  for r in results for e, l in enumerate(r["curve"])]).to_csv(
        OUT / "curves.csv", index=False)
    ex = {"x512": d["x"], "u0": d["u0"][cfg["data"]["n_train"]:][:N_EXAMPLES]}
    for nu in nus:
        ex[f"truth_nu{nu:g}"] = d[f"uT_nu{nu:g}"][cfg["data"]["n_train"]:][:N_EXAMPLES]
    for r in results:
        for k, v in r["examples"].items():
            ex[f"nu{r['nu']:g}_{k}"] = v
    np.savez_compressed(OUT / "examples.npz", **ex)

    s = score(df, cfg)
    s["params"] = {r["model"]: r["params"] for r in results}
    s["minutes"] = (time.time() - started) / 60
    (OUT / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")

    print("\nmedian over seeds of the mean relative L2 error on 200 held-out samples")
    table = (df[df["mode"] == "zero_shot"]
             .groupby(["nu", "model", "train_res", "eval_res"])["error"].median()
             .unstack("eval_res"))
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\ninterpolated from the 64-point output")
    t2 = (df[df["mode"] != "zero_shot"]
          .groupby(["nu", "model", "mode", "eval_res"])["error"].median().unstack("eval_res"))
    print(t2.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    for k in ("P1", "P2", "P3", "P4", "P5"):
        print(f"  {k}: {'passed' if s[k]['passed'] else 'FAILED'}   "
              + json.dumps({a: b for a, b in s[k].items() if a != 'passed'}))
    print(f"\n{s['minutes']:.0f} min -> {OUT}")


if __name__ == "__main__":
    main()
