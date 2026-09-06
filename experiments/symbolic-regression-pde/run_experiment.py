"""
run_experiment.py -- Where does symbolic regression stop recovering the
equation, and what does it do at the moment it stops?

    python run_experiment.py

Four arms:

  A  noise sweep, raw finite differences        -> outputs/sweep.csv
  B  the same sweep, smoothed first             -> outputs/smoothed.csv
  C  threshold sensitivity                      -> outputs/threshold.csv
  D  the equation against two black boxes, on
     time none of them was fitted on            -> outputs/extrapolation.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

import sindy as S
from data import load_field, add_noise

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def trim_for(cfg, window: int | None) -> tuple[int, int]:
    """How much of the domain to throw away.

    Without smoothing: `n_edge` points per side in x, because a central
    stencil cannot be evaluated at a Neumann wall without ghost nodes.

    With smoothing: more. Savitzky-Golay pads the array at the edges, so a
    band of `window//2` points either side is a polynomial fit to invented
    data. Feeding that to the regression is what produced the first version
    of arm B, in which smoothing appeared to destroy recovery even on a
    noiseless field.
    """
    d = cfg["domain"]
    if window is None or not cfg["smoothing"].get("widen_trim", True):
        return d["n_burn"], d["n_edge"]
    half = window // 2
    return d["n_burn"] + half, max(d["n_edge"], half + 3)


def fit_once(U, f, cfg, threshold, window=None):
    """One field in, one equation out."""
    if window is not None:
        U = S.smooth(U, window, cfg["smoothing"]["polyorder"])
    burn, edge = trim_for(cfg, window)
    smp = S.assemble(U, f.dx, f.dt, burn, edge)
    return S.stlsq(smp.Theta, smp.u_t, threshold), smp.names


def verdict(xi, names, true_coef, tol):
    """Did it find the equation, and if not, how did it fail?

    `right_term_set` and `kept_true_term` are recorded separately so the
    failure mode can be read off: dropping the true term and burying it under
    spurious ones are different failures, and CRITERIA predicts which happens.
    """
    terms = S.active_terms(xi, names)
    coef = xi[names.index(S.TRUE_TERM)]
    err = abs(coef - true_coef) / true_coef
    right_set = terms == [S.TRUE_TERM]
    return {
        "terms": "|".join(terms) if terms else "(empty)",
        "n_terms": len(terms),
        "coef": float(coef),
        "coef_rel_error": float(err),
        "kept_true_term": bool(S.TRUE_TERM in terms),
        "right_term_set": bool(right_set),
        "recovered": bool(right_set and err < tol),
    }


def cliff(df: pd.DataFrame, cfg) -> float:
    """Largest sigma recovered in at least `seeds_required` of the seeds.

    -1.0 means nothing was recovered at all, which is a different statement
    from 0.0, meaning only the noiseless field was.
    """
    need = cfg["recovery"]["seeds_required"]
    g = df.groupby("sigma")["recovered"].sum()
    ok = g[g >= need]
    return float(ok.index.max()) if len(ok) else -1.0


def show_cliff(c: float) -> str:
    if c < 0:
        return "never recovered, at any noise level"
    if c == 0:
        return "only with no noise at all"
    return f"survives up to sigma {c:.0e}"


def sweep(f, cfg, label, **kw) -> pd.DataFrame:
    """One pass over every sigma and seed, holding `kw` fixed."""
    tol = cfg["recovery"]["coefficient_tolerance"]
    rows = []
    for sigma in cfg["sweep"]["sigma"]:
        for s in range(cfg["sweep"]["seeds"]):
            U = add_noise(f.U, sigma, cfg["seed"] + s)
            xi, names = fit_once(U, f, cfg,
                                 kw.get("threshold", cfg["sweep"]["threshold"]),
                                 window=kw.get("window"))
            rows.append({**{k: v for k, v in kw.items() if v is not None},
                         "sigma": sigma, "seed": s,
                         **verdict(xi, names, f.true_coef, tol)})
    d = pd.DataFrame(rows)
    print(f"  {label:<26} {show_cliff(cliff(d, cfg))}", flush=True)
    return d


def arm_a(f, cfg) -> pd.DataFrame:
    print("=== arm A: noise sweep, raw finite differences ===")
    d = sweep(f, cfg, "raw finite differences")
    for sigma in cfg["sweep"]["sigma"]:
        r = d[d["sigma"] == sigma]
        print(f"    sigma {sigma:<9.0e} recovered {int(r['recovered'].sum())}/5"
              f"   kept u_xx {int(r['kept_true_term'].sum())}/5"
              f"   median terms {r['n_terms'].median():.0f}"
              f"   median coef error {r['coef_rel_error'].median():.4g}")
    return d


def arm_b(f, cfg) -> pd.DataFrame:
    print("\n=== arm B: Savitzky-Golay before differentiating ===")
    return pd.concat([sweep(f, cfg, f"window {w:>3}", window=w)
                      for w in cfg["smoothing"]["windows"]], ignore_index=True)


def arm_c(f, cfg) -> pd.DataFrame:
    print("\n=== arm C: does the answer depend on the threshold? ===")
    return pd.concat([sweep(f, cfg, f"threshold {t:<9.0e}", threshold=t)
                      for t in cfg["sweep"]["threshold_sensitivity"]],
                     ignore_index=True)


def arm_d(f, cfg, sigma: float) -> pd.DataFrame:
    """P5: fit on the first half of the time domain, score on both halves.

    Three models, all fitted on the same rows:

      equation       STLSQ on the library
      gbdt+library   a gradient booster given the same library columns
      gbdt+raw u     a gradient booster given only the raw 5-point stencil
                     of u, which is what "a black box on this data" means
                     before anyone has decided which derivatives matter

    All three are scored against the *clean* du/dt: the question is which
    recovered the physics, not which reproduced the noise it was shown. Both
    halves use one common denominator so the two columns can be compared.
    """
    print(f"\n=== arm D: fit on the first half of time, score on both "
          f"(sigma={sigma:.0e}) ===")
    d, ex = cfg["domain"], cfg["extrapolation"]
    nb, ne, k = d["n_burn"], d["n_edge"], ex["stencil"]
    rows = []
    for s in range(cfg["sweep"]["seeds"]):
        U = add_noise(f.U, sigma, cfg["seed"] + s)
        smp = S.assemble(U, f.dx, f.dt, nb, ne, U_clean=f.U)

        nt, nx = U.shape
        st = np.stack([U[nb:nt - 1, ne + j - k // 2: nx - ne + j - k // 2]
                       for j in range(k)], axis=-1).reshape(-1, k)
        assert st.shape[0] == smp.Theta.shape[0]

        lo, hi = smp.t_index.min(), smp.t_index.max()
        cut = lo + ex["train_fraction"] * (hi - lo)
        early, late = smp.t_index <= cut, smp.t_index > cut

        xi = S.stlsq(smp.Theta[early], smp.u_t[early], cfg["sweep"]["threshold"])
        gl = HistGradientBoostingRegressor(random_state=cfg["seed"] + s,
                                           **ex["gbdt"]).fit(smp.Theta[early],
                                                             smp.u_t[early])
        gr = HistGradientBoostingRegressor(random_state=cfg["seed"] + s,
                                           **ex["gbdt"]).fit(st[early],
                                                             smp.u_t[early])
        denom = float(np.sqrt(np.mean(smp.u_t ** 2)))

        def rel(pred, m):
            return float(np.sqrt(np.mean((pred - smp.u_t[m]) ** 2)) / denom)

        for half, m in (("trained on", early), ("held out (later)", late)):
            rows.append({
                "seed": s, "half": half,
                "equation": rel(smp.Theta[m] @ xi, m),
                "gbdt_library": rel(gl.predict(smp.Theta[m]), m),
                "gbdt_raw": rel(gr.predict(st[m]), m),
                "unseen_u_share": float((st[m] > st[early].max()).mean()),
                "terms": "|".join(S.active_terms(xi, smp.names)),
            })
    r = pd.DataFrame(rows)
    print(f"  {'':<18}{'equation':>10}{'gbdt+library':>14}{'gbdt+raw u':>12}")
    for half in ("trained on", "held out (later)"):
        h = r[r["half"] == half]
        print(f"  {half:<18}{h['equation'].median():>10.4f}"
              f"{h['gbdt_library'].median():>14.4f}"
              f"{h['gbdt_raw'].median():>12.4f}")
    late = r[r["half"] == "held out (later)"]
    print(f"  share of held-out u values above anything seen in training: "
          f"{late['unseen_u_share'].median():.1%}")
    return r


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    f = load_field()

    print(f"field {f.shape[0]} x {f.shape[1]}   "
          f"true coefficient D*T/L^2 = {f.true_coef:.10f}\n")

    # ---- P1: the control, checked before anything else ------------------
    xi, names = fit_once(f.U, f, cfg, cfg["sweep"]["threshold"])
    v = verdict(xi, names, f.true_coef, 0.01)
    print("control (no noise):")
    print(f"  found      {S.describe(xi, names)}")
    print(f"  truth      u_t = +{f.true_coef:.4f} u_xx")
    print(f"  coefficient error {v['coef_rel_error']:.3e}")
    if not (v["right_term_set"] and v["coef_rel_error"] < 0.01):
        sys.exit(
            "\nFAIL - the method does not recover the equation from noiseless "
            "data. The implementation is wrong, and every number below would "
            "be measuring the bug rather than the physics.")
    print("  OK - the equation is recoverable from this grid\n")

    a = arm_a(f, cfg); a.to_csv(OUT / "sweep.csv", index=False)
    b = arm_b(f, cfg); b.to_csv(OUT / "smoothed.csv", index=False)
    c = arm_c(f, cfg); c.to_csv(OUT / "threshold.csv", index=False)

    raw_cliff = cliff(a, cfg)
    dd = arm_d(f, cfg, max(raw_cliff, 0.0))
    dd.to_csv(OUT / "extrapolation.csv", index=False)

    best_w = max(cfg["smoothing"]["windows"],
                 key=lambda w: cliff(b[b["window"] == w], cfg))
    summary = {
        "true_coef": f.true_coef,
        "control_coef": float(xi[names.index(S.TRUE_TERM)]),
        "control_rel_error": v["coef_rel_error"],
        "cliff_raw": raw_cliff,
        "cliff_smoothed": {str(w): cliff(b[b["window"] == w], cfg)
                           for w in cfg["smoothing"]["windows"]},
        "best_window": best_w,
        "cliff_by_threshold": {str(t): cliff(c[c["threshold"] == t], cfg)
                               for t in cfg["sweep"]["threshold_sensitivity"]},
        "arm_d_sigma": max(raw_cliff, 0.0),
        "tolerance": cfg["recovery"]["coefficient_tolerance"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2),
                                      encoding="utf-8")

    gain = cliff(b[b["window"] == best_w], cfg) / raw_cliff if raw_cliff > 0 else 0
    print(f"\nsmoothing at its best window ({best_w}) buys a factor of "
          f"{gain:.0f} in noise tolerance")
    print(f"Wrote sweep.csv, smoothed.csv, threshold.csv, extrapolation.csv, "
          f"summary.json -> {OUT}")


if __name__ == "__main__":
    main()
