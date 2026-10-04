"""
posthoc.py -- Three analyses added after the results were in. Not
pre-registered, and reported as such.

    python posthoc.py

The pre-registered capacity measure, K90, is censored below 4 pairs for
any run that never learned the task at all, and most selective-SSM runs
did not. That makes the scaling prediction (P4) unmeasurable, and it
leaves open whether those runs had too small a state or simply never
learned to use it. These three analyses separate the two.

  1. Pairs recalled, corrected for guessing. A run that has learned which
     tokens are values but not which key each belongs to scores the guess
     level g (answering with a random value from the sequence). Its
     accuracy above that, rescaled, counts the pairs it actually holds:
         recalled = N_pairs x (accuracy - g) / (1 - g)
  2. Did each run learn, stall, or run out of budget? From the training
     curves: a run is "learned" if it ends at 95% or more at 4 pairs,
     "stuck" if it ends within 0.05 of the guess level at 4 pairs, and
     otherwise "partial" - with the change over its last quarter of
     training, to show whether it was still climbing.
  3. Capacity among the runs that learned. K90 for the "learned" runs only,
     per state size.

Writes outputs/posthoc.json and outputs/posthoc_runs.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_experiment import k90

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
TESTS = HERE.parent.parent / "generators" / "associative_recall" / "outputs" / "recall_test.npz"


def guess_level() -> dict:
    z = np.load(TESTS)
    out = {}
    for n in z["pair_counts"]:
        tok, tgt = z[f"tokens_{n}"].astype(int), z[f"targets_{n}"].astype(int)
        vals, ans = tok[:, 1:2 * n:2], tgt[:, 2 * n:]
        out[int(n)] = float((vals[:, None, :] == ans[:, :, None]).sum(-1).mean() / n)
    return out


def main() -> None:
    acc = pd.read_csv(OUT / "accuracy.csv")
    cur = pd.read_csv(OUT / "curves.csv")
    g = guess_level()
    counts = sorted(g)
    last = cur.step.max()
    quarter = cur.step[cur.step <= 0.75 * last].max()

    rows = []
    for (name, seed), d in acc.groupby(["name", "seed"]):
        a = dict(zip(d.pairs, d.accuracy))
        recalled = {k: k * (a[k] - g[k]) / (1 - g[k]) for k in counts}
        c4 = cur[(cur.name == name) & (cur.seed == seed) & (cur.pairs == counts[0])]
        end4 = float(c4[c4.step == last].accuracy.iloc[0])
        q4 = float(c4[c4.step == quarter].accuracy.iloc[0])
        if a[counts[0]] >= 0.95:
            status = "learned"
        elif a[counts[0]] - g[counts[0]] < 0.05:
            status = "stuck"
        else:
            status = "partial"
        value, censored = k90(a)
        rows.append({"name": name, "seed": seed, "status": status,
                     "accuracy_at_4": a[counts[0]], "accuracy_at_64": a[counts[-1]],
                     "recalled_max": max(recalled.values()),
                     "recalled_at_64": recalled[counts[-1]],
                     "curve_4pairs_last_quarter_change": end4 - q4,
                     "k90": value, "k90_censored": censored})
    runs = pd.DataFrame(rows)
    order = {n: i for i, n in enumerate(["attention", "ssm-4", "ssm-8", "ssm-16", "ssm-32", "lti-32"])}
    runs = runs.sort_values(["name", "seed"], key=lambda s: s.map(order) if s.name == "name" else s)
    runs.to_csv(OUT / "posthoc_runs.csv", index=False)

    print("guess level (a random value from the sequence): "
          + ", ".join(f"{k} pairs {v:.3f}" for k, v in g.items()))
    print("\nper run")
    print(runs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    learned = runs[runs.status == "learned"]
    by_size = {name: {"learned": int((d.status == "learned").sum()), "runs": int(len(d)),
                      "k90_of_learned": d[d.status == "learned"].k90.round(1).tolist(),
                      "recalled_at_64_of_learned":
                          d[d.status == "learned"].recalled_at_64.round(1).tolist()}
               for name, d in runs.groupby("name")}
    print("\nruns that learned the task (95%+ at 4 pairs), and their capacity")
    for name in sorted(by_size, key=lambda n: order.get(n, 99)):
        b = by_size[name]
        print(f"  {name:<10} {b['learned']}/{b['runs']} learned   K90 {b['k90_of_learned']}   "
              f"pairs recalled at 64: {b['recalled_at_64_of_learned']}")
    out = {"guess_level": {str(k): v for k, v in g.items()}, "by_model": by_size,
           "n_selective_runs": int(runs.name.str.startswith("ssm").sum()),
           "n_selective_learned": int((learned.name.str.startswith("ssm")).sum())}
    (OUT / "posthoc.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n-> {OUT / 'posthoc.json'}")


if __name__ == "__main__":
    main()
