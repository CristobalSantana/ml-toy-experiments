"""
run_experiment.py -- Train every model, measure recall at every pair count,
score the predictions.

    python run_experiment.py

Eighteen trainings - six models, three seeds - three at a time with two
threads each. The control (P1) trains first: the Transformer and the
largest selective SSM, checked at 4 pairs before anything else runs.

Writes to outputs/:
  accuracy.csv   one row per model x seed x pair count
  curves.csv     test accuracy (200 sequences per count) during training
  memory.csv     inference memory per layer, per model and length
  summary.json   every prediction, scored
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
GEN = HERE.parent.parent / "generators" / "associative_recall"


def load_cfg() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def load_tests() -> dict:
    z = np.load(GEN / "outputs" / "recall_test.npz")
    return {int(n): (torch.tensor(z[f"tokens_{n}"].astype(np.int64)),
                     torch.tensor(z[f"targets_{n}"].astype(np.int64)))
            for n in z["pair_counts"]}


def specs(cfg: dict) -> list[dict]:
    out = [{"name": "attention", "mixer": "attention"}]
    out += [{"name": f"ssm-{n}", "mixer": "ssm", "d_state": n} for n in cfg["ssm"]["d_states"]]
    out += [{"name": f"lti-{cfg['ssm']['lti_d_state']}", "mixer": "lti",
             "d_state": cfg["ssm"]["lti_d_state"]}]
    return out


def query_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy at the query positions only (targets of -1 elsewhere)."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=-1)


@torch.no_grad()
def accuracy(model, tokens: torch.Tensor, targets: torch.Tensor, batch: int = 250) -> float:
    hits = total = 0
    for i in range(0, len(tokens), batch):
        tok, tgt = tokens[i:i + batch], targets[i:i + batch]
        first = int((tgt[0] != -1).nonzero()[0])          # where the queries start
        pred = model(tok, start=first).argmax(-1)
        tgt = tgt[:, first:]
        q = tgt != -1
        hits += int((pred[q] == tgt[q]).sum())
        total += int(q.sum())
    return hits / total


# --------------------------------------------------------------------------
# one training run
# --------------------------------------------------------------------------

def run_one(task: dict) -> dict:
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(GEN))
    import models as M
    from generate import sample
    torch.set_num_threads(2)
    cfg, spec, seed = task["cfg"], task["spec"], task["seed"]
    tr, gen = cfg["training"], cfg["data"]
    tests = load_tests()
    vocab = gen["n_keys"] + gen["n_values"]
    counts = gen["pair_counts"]

    torch.manual_seed(cfg["seed"] + 1000 * seed)
    model = M.build(spec, vocab, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"],
                            betas=(0.9, 0.98))
    warm, steps = tr["warmup"], tr["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    # the same training sequences for every model of this seed; a stream the
    # test sets never use (their second seed word is 0)
    rng = np.random.default_rng([cfg["seed"], 1, seed])
    started, curve = time.time(), []
    for step in range(steps):
        n = counts[step % len(counts)]
        tok, tgt = sample(rng, tr["batch"], n, gen["n_keys"], gen["n_values"])
        model.train()
        opt.zero_grad()
        loss = query_loss(model(torch.from_numpy(tok), start=2 * n),
                          torch.from_numpy(tgt[:, 2 * n:]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
        opt.step()
        sched.step()
        if (step + 1) % tr["eval_every"] == 0 or step + 1 == steps:
            model.eval()
            for k in counts:
                tok_t, tgt_t = tests[k]
                curve.append({"step": step + 1, "pairs": k,
                              "accuracy": accuracy(model, tok_t[:200], tgt_t[:200])})
    seconds = time.time() - started
    model.eval()
    acc = {k: accuracy(model, *tests[k]) for k in counts}
    return {"name": spec["name"], "mixer": spec["mixer"], "d_state": spec.get("d_state"),
            "seed": seed, "accuracy": acc, "curve": curve, "seconds": seconds,
            "params": M.n_params(model),
            "memory": {3 * k: model.memory_floats(3 * k) // cfg["model"]["n_layers"]
                       for k in counts}}


# --------------------------------------------------------------------------
# capacity and scoring
# --------------------------------------------------------------------------

def k90(acc: dict, threshold: float = 0.9) -> tuple[float, str]:
    """Pair count where accuracy falls to `threshold`, interpolated in
    log2(pairs). Returns (value, censoring): "" for a measured value,
    "below" when already under the threshold at the smallest count, "above"
    when it never falls under it."""
    ks = sorted(acc)
    if acc[ks[0]] < threshold:
        return float(ks[0]), "below"
    for lo, hi in zip(ks, ks[1:]):
        if acc[hi] < threshold:
            f = (acc[lo] - threshold) / (acc[lo] - acc[hi])
            return float(2 ** (math.log2(lo) + f * (math.log2(hi) - math.log2(lo)))), ""
    return float(ks[-1]), "above"


def median_k90(rows: list[tuple[float, str]]) -> tuple[float, str]:
    """Median over seeds, censored values at their bound. The median is
    censored if the seed it lands on is."""
    rows = sorted(rows, key=lambda r: r[0])
    return rows[len(rows) // 2]


def score(df: pd.DataFrame, cap: pd.DataFrame, cfg: dict) -> dict:
    med = lambda name, k: float(df[(df.name == name) & (df.pairs == k)].accuracy.median())
    counts = cfg["data"]["pair_counts"]
    big = max(cfg["ssm"]["d_states"])
    s = {}
    p1 = {"attention": med("attention", counts[0]), f"ssm-{big}": med(f"ssm-{big}", counts[0])}
    s["P1"] = {"accuracy_at_4": p1, "passed": all(v >= 0.95 for v in p1.values())}
    att = {str(k): med("attention", k) for k in counts}
    s["P2"] = {"attention_accuracy": att, "passed": all(v >= 0.95 for v in att.values())}
    at64 = {f"ssm-{n}": med(f"ssm-{n}", counts[-1]) for n in cfg["ssm"]["d_states"]}
    s["P3"] = {"accuracy_at_64": at64, "passed": all(v < 0.9 for v in at64.values())}
    ks = {n: median_k90([(r.k90, r.censored) for r in cap[cap.name == f"ssm-{n}"].itertuples()])
          for n in cfg["ssm"]["d_states"]}
    free = {n: v for n, (v, c) in ks.items() if c == ""}
    if len(free) >= 3:
        x = np.log2(list(free)); y = np.log2(list(free.values()))
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = float("nan")
    s["P4"] = {"median_k90": {str(n): {"value": v, "censored": c} for n, (v, c) in ks.items()},
               "uncensored_sizes": len(free), "slope": slope,
               "passed": len(free) >= 3 and 0.5 <= slope <= 1.5}
    lti = median_k90([(r.k90, r.censored) for r in
                      cap[cap.name == f"lti-{cfg['ssm']['lti_d_state']}"].itertuples()])
    sel = ks[cfg["ssm"]["lti_d_state"]]
    s["P5"] = {"lti_k90": {"value": lti[0], "censored": lti[1]},
               "selective_k90": {"value": sel[0], "censored": sel[1]},
               "passed": lti[0] <= sel[0] / 4}
    return s


def main() -> None:
    from joblib import Parallel, delayed
    cfg = load_cfg()
    OUT.mkdir(exist_ok=True)
    seeds = range(cfg["training"]["seeds"])
    big = max(cfg["ssm"]["d_states"])
    all_specs = specs(cfg)
    tasks = [{"cfg": cfg, "spec": sp, "seed": s} for sp in all_specs for s in seeds]
    control = [t for t in tasks if t["spec"]["name"] in ("attention", f"ssm-{big}")]
    rest = [t for t in tasks if t not in control]
    counts = cfg["data"]["pair_counts"]
    started = time.time()

    def report(r):
        accs = "  ".join(f"{k}:{r['accuracy'][k]:.3f}" for k in counts)
        print(f"  {r['name']:<10} seed {r['seed']}  {accs}   ({r['seconds'] / 60:.1f} min)",
              flush=True)

    print(f"control first: {len(control)} trainings", flush=True)
    results = []
    for r in Parallel(n_jobs=3, return_as="generator")(delayed(run_one)(t) for t in control):
        report(r); results.append(r)
    p1 = {name: float(np.median([r["accuracy"][counts[0]] for r in results if r["name"] == name]))
          for name in ("attention", f"ssm-{big}")}
    if min(p1.values()) < 0.95:
        sys.exit(f"\nCONTROL FAILED (P1): accuracy at {counts[0]} pairs {p1}. "
                 "A model that cannot recall four pairs is not trained; stopping.")
    print(f"  P1 control passed: {p1}\n", flush=True)

    print(f"the other {len(rest)} trainings, three at a time", flush=True)
    for r in Parallel(n_jobs=3, return_as="generator")(delayed(run_one)(t) for t in rest):
        report(r); results.append(r)

    df = pd.DataFrame([{"name": r["name"], "mixer": r["mixer"], "d_state": r["d_state"],
                        "seed": r["seed"], "pairs": k, "accuracy": a}
                       for r in results for k, a in r["accuracy"].items()])
    df.to_csv(OUT / "accuracy.csv", index=False)
    cap = pd.DataFrame([{"name": r["name"], "seed": r["seed"],
                         **dict(zip(("k90", "censored"), k90(r["accuracy"])))}
                        for r in results])
    cap.to_csv(OUT / "capacity.csv", index=False)
    pd.DataFrame([{"name": r["name"], "seed": r["seed"], **c}
                  for r in results for c in r["curve"]]).to_csv(OUT / "curves.csv", index=False)
    mem = pd.DataFrame([{"name": r["name"], "length": L, "floats_per_layer": m,
                         "params": r["params"]}
                        for r in results if r["seed"] == 0 for L, m in r["memory"].items()])
    mem.to_csv(OUT / "memory.csv", index=False)

    s = score(df, cap, cfg)
    s["params"] = {r["name"]: r["params"] for r in results if r["seed"] == 0}
    s["minutes"] = (time.time() - started) / 60
    (OUT / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")

    print("\nmedian accuracy over seeds")
    order = [sp["name"] for sp in all_specs]
    table = df.groupby(["name", "pairs"]).accuracy.median().unstack("pairs").loc[order]
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nK90 per seed")
    print(cap.to_string(index=False))
    print()
    for k in ("P1", "P2", "P3", "P4", "P5"):
        print(f"  {k}: {'passed' if s[k]['passed'] else 'FAILED'}   "
              + json.dumps({a: b for a, b in s[k].items() if a != 'passed'}))
    print(f"\n{s['minutes']:.0f} min -> {OUT}")


if __name__ == "__main__":
    main()
