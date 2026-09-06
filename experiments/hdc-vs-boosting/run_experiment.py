"""
run_experiment.py -- The three arms: learning curve, robustness, cost.

    python run_experiment.py

Everything is scored on one fixed sample of June 2024, identical for every
cell, so the numbers in different rows are comparable to each other and not
just to themselves.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import data as D
from hdc import HDClassifier

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


# --------------------------------------------------------------------------
# a common interface, so the arms do not care which method they hold
# --------------------------------------------------------------------------

class Majority:
    name = "majority class"

    def fit(self, X, y):
        self.c = int(pd.Series(y).mode().iloc[0])
        return self

    def predict(self, X):
        return np.full(len(X), self.c)

    def scores(self, X):
        return np.zeros(len(X))          # no ranking at all: AUC 0.5

    n_parameters = 1


class LogReg:
    name = "logistic regression"

    def __init__(self, seed=0, **kw):
        self.s = StandardScaler()
        self.m = LogisticRegression(**kw)

    def fit(self, X, y):
        self.m.fit(self.s.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.m.predict(self.s.transform(X))

    def scores(self, X):
        return self.m.decision_function(self.s.transform(X))

    @property
    def n_parameters(self):
        return int(self.m.coef_.size + 1)


class GBDT:
    name = "gradient boosting"

    def __init__(self, seed=0, **kw):
        self.m = HistGradientBoostingClassifier(random_state=seed, **kw)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)

    def scores(self, X):
        return self.m.predict_proba(X)[:, 1]

    @property
    def n_parameters(self):
        # nodes across every tree: what has to be kept to predict
        return int(sum(p[0].get_n_leaf_nodes() * 2 - 1
                       for p in self.m._predictors))


class HDC:
    name = "hyperdimensional"

    def __init__(self, seed=0, **kw):
        self.m = HDClassifier(seed=seed, **kw)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X, dim_mask=None):
        return self.m.predict(X, dim_mask=dim_mask)

    def scores(self, X):
        return self.m.decision_scores(X)

    def predict_and_scores(self, X):
        return self.m.predict_and_scores(X)

    @property
    def n_parameters(self):
        return int(self.m.n_parameters)


def make(kind: str, cfg: dict, seed: int):
    if kind == "majority":
        return Majority()
    if kind == "logreg":
        return LogReg(seed=seed, **cfg["models"]["logreg"])
    if kind == "gbdt":
        return GBDT(seed=seed, **cfg["models"]["gbdt"])
    if kind == "hdc":
        return HDC(seed=seed, **cfg["models"]["hdc"])
    raise ValueError(kind)


def score_all(model, Xte, yte) -> dict:
    # HDC pays for the encoding, so it gets both out of one pass; the others
    # are cheap enough that the distinction does not matter.
    if hasattr(model, "predict_and_scores"):
        p, s = model.predict_and_scores(Xte)
    else:
        p, s = model.predict(Xte), model.scores(Xte)
    return {"auc": float(roc_auc_score(yte, s)),
            "balanced_accuracy": float(balanced_accuracy_score(yte, p)),
            "accuracy": float(accuracy_score(yte, p))}


def corrupt(X: np.ndarray, pool: np.ndarray, frac: float,
            rng: np.random.Generator) -> np.ndarray:
    """Replace a share of values with draws from the training distribution.

    Not Gaussian noise: a feature replaced by a plausible value from the same
    column is the realistic failure - a sensor reporting a stale reading, a
    field defaulted upstream - and it cannot be detected as out of range.
    """
    if frac <= 0:
        return X
    out = X.copy()
    mask = rng.random(X.shape) < frac
    for j in range(X.shape[1]):
        k = int(mask[:, j].sum())
        if k:
            out[mask[:, j], j] = rng.choice(pool[:, j], size=k, replace=True)
    return out


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    seed, task = cfg["seed"], cfg["task"]
    OUT.mkdir(parents=True, exist_ok=True)

    pool, test = D.build(task["train_month"], task["test_month"],
                         task["test_size"], seed)
    Xte = test[D.FEATURES].to_numpy(dtype=np.float64)
    yte = test["is_cash"].to_numpy()

    kinds = ["majority", "logreg", "gbdt", "hdc"]

    # ---------------- arm A: learning curve ------------------------------
    print("\n=== arm A: learning curve ===")
    rows = []
    for n in cfg["learning_curve"]["train_sizes"]:
        for s in range(cfg["learning_curve"]["seeds"]):
            tr = D.subsample(pool, n, seed + s)
            Xtr = tr[D.FEATURES].to_numpy(dtype=np.float64)
            ytr = tr["is_cash"].to_numpy()
            if len(np.unique(ytr)) < 2:      # a tiny sample can be one class
                continue
            for k in kinds:
                t0 = time.perf_counter()
                m = make(k, cfg, seed + s).fit(Xtr, ytr)
                fit_s = time.perf_counter() - t0
                r = score_all(m, Xte, yte)
                rows.append({"n_train": n, "seed": s, "model": m.name,
                             "fit_seconds": fit_s,
                             "n_parameters": int(m.n_parameters), **r})
        got = [r for r in rows if r["n_train"] == n]
        line = "  ".join(
            f"{k[:4]} {np.median([r['auc'] for r in got if r['model'].startswith(nm)]):.3f}"
            for k, nm in zip(kinds, ["majority", "logistic", "gradient", "hyper"]))
        print(f"  n={n:>7,}   AUC   {line}", flush=True)

    lc = pd.DataFrame(rows)
    lc.to_csv(OUT / "learning_curve.csv", index=False)

    # ---------------- arm B: robustness ----------------------------------
    print("\n=== arm B: robustness ===")
    n_rb = cfg["robustness"]["n_train"]
    tr = D.subsample(pool, n_rb, seed)
    Xtr = tr[D.FEATURES].to_numpy(dtype=np.float64)
    ytr = tr["is_cash"].to_numpy()
    fitted = {k: make(k, cfg, seed).fit(Xtr, ytr) for k in ["logreg", "gbdt", "hdc"]}

    rb = []
    rng = np.random.default_rng(seed)
    for frac in cfg["robustness"]["input_corruption"]:
        Xc = corrupt(Xte, Xtr, frac, rng)
        for k, m in fitted.items():
            r = score_all(m, Xc, yte)
            rb.append({"kind": "input", "fraction": frac, "model": m.name, **r})
        print(f"  input corrupted {frac:>4.0%}   " + "   ".join(
            f"{m.name.split()[0][:4]} {rb[-len(fitted) + i]['auc']:.3f}"
            for i, m in enumerate(fitted.values())), flush=True)

    hd = fitted["hdc"]
    for frac in cfg["robustness"]["dimension_dropout"]:
        mask = (rng.random(cfg["models"]["hdc"]["dim"]) >= frac).astype(np.float64)
        p = hd.predict(Xte, dim_mask=mask)
        rb.append({"kind": "dimension_dropout", "fraction": frac,
                   "model": hd.name, "auc": float("nan"),
                   "balanced_accuracy": float(balanced_accuracy_score(yte, p)),
                   "accuracy": float(accuracy_score(yte, p))})
        print(f"  {frac:>4.0%} of HDC dimensions switched off   "
              f"balanced acc {rb[-1]['balanced_accuracy']:.4f}", flush=True)

    pd.DataFrame(rb).to_csv(OUT / "robustness.csv", index=False)

    # ---------------- arm C: cost ----------------------------------------
    big = lc["n_train"].max()
    cost = (lc[lc.n_train == big].groupby("model")[["fit_seconds", "n_parameters", "auc"]]
            .median().reset_index().sort_values("auc", ascending=False))
    cost.to_csv(OUT / "cost.csv", index=False)
    print(f"\n=== arm C: cost at n = {big:,} ===")
    for _, r in cost.iterrows():
        print(f"  {r['model']:<22} AUC {r['auc']:.4f}   "
              f"fit {r['fit_seconds']:>7.2f} s   "
              f"kept parameters {int(r['n_parameters']):>10,}")

    print(f"\nWrote learning_curve.csv, robustness.csv, cost.csv -> {OUT}")


if __name__ == "__main__":
    main()
