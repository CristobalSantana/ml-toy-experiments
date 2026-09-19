"""
run_experiment.py -- Who gets infected: message passing against features a
person would write down.

    python run_experiment.py

  A  Erdős–Rényi: every model, held-out graphs        -> outputs/erdos_renyi.csv
  B  the GNN's depth sweep                             -> outputs/depth.csv
  C  structural drift: trained on ER, scored on BA     -> outputs/drift.csv
  D  what the GNN's embedding encodes                  -> outputs/embedding.csv

The control is fatal. If hand-computed structure cannot predict infection,
the task is not about structure and nothing downstream measures it.
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
from sklearn.metrics import roc_auc_score

import graph as G

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
GEN = HERE.parent.parent / "generators" / "sir_on_graph" / "outputs"

# the names in the output have accents, and a redirected stdout on Windows
# defaults to a code page that cannot write them
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def tabular(graphs, mu=None, sd=None):
    """Stack hand-computed features over graphs; standardise on training stats."""
    X = np.concatenate([G.node_features(g) for g in graphs])
    y = np.concatenate([g.infected for g in graphs]).astype(np.float32)
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-9
    return (torch.tensor((X - mu) / sd, dtype=torch.float32),
            torch.tensor(y), mu, sd)


def fit_tabular(kind, Xtr, ytr, cfg, seed, cols=None):
    """kind: 'logreg' or 'mlp'; cols restricts the feature set."""
    torch.manual_seed(seed)
    X = Xtr if cols is None else Xtr[:, cols]
    m = (G.FeatureLogReg(X.shape[1]) if kind == "logreg"
         else G.FeatureMLP(X.shape[1], cfg["tabular"]["n_hidden"]))
    G.train_tabular(m, X, ytr, epochs=cfg["tabular"]["epochs"],
                    batch=cfg["tabular"]["batch"], lr=cfg["tabular"]["lr"], seed=seed)
    return m


def score_tabular(m, X, y, cols=None):
    return roc_auc_score(y.numpy(), G.predict_tabular(m, X if cols is None else X[:, cols]))


def fit_gnn(L, graphs, inputs, cfg, seed):
    torch.manual_seed(seed)
    m = G.GNN(L, n_hidden=cfg["gnn"]["n_hidden"])
    G.train_gnn(m, graphs, inputs, epochs=cfg["gnn"]["epochs"],
                lr=cfg["gnn"]["lr"], seed=seed)
    return m


def score_gnn(m, graphs, inputs):
    y = np.concatenate([g.infected for g in graphs])
    return roc_auc_score(y, G.predict_gnn(m, inputs))


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    t = cfg["task"]

    er, er_dropped = G.load_family(GEN / "sir_erdos_renyi.npz", t["min_final_size"])
    ba, ba_dropped = G.load_family(GEN / "sir_barabasi_albert.npz", t["min_final_size"])
    print(f"major outbreaks: Erdős–Rényi {len(er)} of {len(er) + er_dropped}, "
          f"Barabási–Albert {len(ba)} of {len(ba) + ba_dropped}")

    perm = rng.permutation(len(er))
    n_tr = int(round(t["train_share"] * len(er)))
    n_val = int(round(t["val_share_of_train"] * n_tr))
    fit_ix, val_ix, test_ix = perm[:n_tr - n_val], perm[n_tr - n_val:n_tr], perm[n_tr:]
    fit_g = [er[i] for i in fit_ix]; val_g = [er[i] for i in val_ix]
    test_g = [er[i] for i in test_ix]
    print(f"split by graph: {len(fit_g)} fit, {len(val_g)} validation (GNN depth), "
          f"{len(test_g)} held out; {sum(g.n for g in test_g):,} held-out nodes, "
          f"{np.mean([g.infected.mean() for g in test_g]):.1%} infected\n")

    Xfit, yfit, mu, sd = tabular(fit_g)
    Xval, yval, _, _ = tabular(val_g, mu, sd)
    Xte, yte, _, _ = tabular(test_g, mu, sd)
    Xba, yba, _, _ = tabular(ba, mu, sd)                 # BA scored with ER stats
    fit_in = [G.gnn_inputs(g) for g in fit_g]
    val_in = [G.gnn_inputs(g) for g in val_g]
    test_in = [G.gnn_inputs(g) for g in test_g]
    ba_in = [G.gnn_inputs(g) for g in ba]
    dist_col = [G.FEATURE_NAMES.index("dist_to_seed")]
    deg_col = [G.FEATURE_NAMES.index("degree")]

    # ---- post-hoc, added after the pre-registered results were seen ------
    # The GNN's logit turned out to be rank-correlated +0.95 with degree, so
    # the ceiling for a degree-only predictor is computed empirically: the
    # infection rate at each degree on the fit graphs, applied to held-out
    # nodes. No model can beat this using degree alone. Disclosed in README.
    def degree_of(graphs):
        return np.concatenate([np.bincount(g.edges.ravel(), minlength=g.n) for g in graphs])
    k_fit, k_te, k_ba = degree_of(fit_g), degree_of(test_g), degree_of(ba)
    yf_np = yfit.numpy()
    rate = {k: float(yf_np[k_fit == k].mean()) for k in np.unique(k_fit)}
    base = float(yf_np.mean())
    ceiling = {
        "degree_only_er": float(roc_auc_score(yte.numpy(), [rate.get(k, base) for k in k_te])),
        "degree_only_ba": float(roc_auc_score(yba.numpy(), [rate.get(k, base) for k in k_ba])),
        "rate_by_degree": {int(k): v for k, v in rate.items() if (k_fit == k).sum() >= 30},
    }

    rows, depth_rows, drift_rows, emb_rows = [], [], [], []
    for s in range(cfg["seeds"]):
        seed = cfg["seed"] + s
        t0 = time.time()
        # ---- tabular models --------------------------------------------
        models = {
            "logreg_dist": (fit_tabular("logreg", Xfit, yfit, cfg, seed, dist_col), dist_col),
            # post-hoc: the feature the GNN actually learned (see README)
            "logreg_degree": (fit_tabular("logreg", Xfit, yfit, cfg, seed, deg_col), deg_col),
            "logreg_features": (fit_tabular("logreg", Xfit, yfit, cfg, seed), None),
            "mlp_features": (fit_tabular("mlp", Xfit, yfit, cfg, seed), None),
        }
        for name, (m, cols) in models.items():
            rows.append({"seed": s, "model": name,
                         "auc": score_tabular(m, Xte, yte, cols)})
            drift_rows.append({"seed": s, "model": name,
                               "auc_er": rows[-1]["auc"],
                               "auc_ba": score_tabular(m, Xba, yba, cols)})

        # ---- the GNN, at every depth, chosen on validation ---------------
        best_L, best_val, best_m = None, -1, None
        for L in cfg["gnn"]["layers"]:
            m = fit_gnn(L, fit_g, fit_in, cfg, seed)
            va, te = score_gnn(m, val_g, val_in), score_gnn(m, test_g, test_in)
            depth_rows.append({"seed": s, "layers": L, "auc_val": va, "auc_test": te})
            if va > best_val:
                best_L, best_val, best_m = L, va, m
        rows.append({"seed": s, "model": "gnn", "auc": score_gnn(best_m, test_g, test_in),
                     "layers": best_L})
        drift_rows.append({"seed": s, "model": "gnn", "auc_er": rows[-1]["auc"],
                           "auc_ba": score_gnn(best_m, ba, ba_in)})

        # ---- what the embedding encodes: rank correlation of the GNN's
        # held-out logit with each hand-computed feature ------------------
        logit = G.predict_gnn(best_m, test_in)
        feats = np.concatenate([G.node_features(g) for g in test_g])
        for j, fname in enumerate(G.FEATURE_NAMES):
            emb_rows.append({"seed": s, "feature": fname,
                             "spearman": float(pd.Series(logit).corr(
                                 pd.Series(feats[:, j]), method="spearman"))})

        r = {x["model"]: x["auc"] for x in rows if x["seed"] == s}
        print(f"  seed {s}: AUC  dist-only {r['logreg_dist']:.3f}   "
              f"logreg {r['logreg_features']:.3f}   mlp {r['mlp_features']:.3f}   "
              f"gnn {r['gnn']:.3f} (L={best_L})   [{time.time() - t0:.0f}s]", flush=True)

    A = pd.DataFrame(rows); A.to_csv(OUT / "erdos_renyi.csv", index=False)
    D = pd.DataFrame(depth_rows); D.to_csv(OUT / "depth.csv", index=False)
    C = pd.DataFrame(drift_rows); C.to_csv(OUT / "drift.csv", index=False)
    E = pd.DataFrame(emb_rows); E.to_csv(OUT / "embedding.csv", index=False)

    # ---- P1: the control ------------------------------------------------
    med = A.groupby("model")["auc"].median()
    print(f"\ncontrol: hand-computed features reach held-out AUC "
          f"{med['logreg_features']:.3f}")
    if med["logreg_features"] <= 0.6:
        sys.exit("\nFAIL - structural features cannot predict infection. The task "
                 "is not about structure and nothing below measures it.")
    print("  OK - the task is structural\n")

    # ---- summary --------------------------------------------------------
    d = D.groupby("layers").median(numeric_only=True)
    c = C.groupby("model").median(numeric_only=True)
    e = E.groupby("feature")["spearman"].median()
    summary = {
        "n_graphs": {"er_major": len(er), "er_dropped": er_dropped,
                     "ba_major": len(ba), "ba_dropped": ba_dropped},
        "auc_er": {k: float(v) for k, v in med.items()},
        "gnn_layers_chosen": [int(x) for x in A[A.model == "gnn"]["layers"]],
        "P2_gnn_minus_mlp": float(med["gnn"] - med["mlp_features"]),
        "P3_dist_share_of_gnn_gain": float((med["logreg_dist"] - 0.5) / (med["gnn"] - 0.5)),
        "P4_drift": {m: {"er": float(c.loc[m, "auc_er"]), "ba": float(c.loc[m, "auc_ba"]),
                         "loss": float(c.loc[m, "auc_er"] - c.loc[m, "auc_ba"])}
                     for m in c.index},
        "P5_depth": {int(L): float(d.loc[L, "auc_test"]) for L in d.index},
        "embedding_spearman": {k: float(v) for k, v in e.items()},
        "posthoc_ceiling": ceiling,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("medians over seeds, held-out Erdős–Rényi graphs")
    for m in ("logreg_dist", "logreg_degree", "logreg_features", "mlp_features", "gnn"):
        print(f"  {m:<18} AUC {med[m]:.4f}")
    print(f"  {'(ceiling, degree)':<18} AUC {ceiling['degree_only_er']:.4f}   "
          f"post-hoc: empirical P(infected | degree)")
    print(f"\n  P2  gnn - mlp_features = {summary['P2_gnn_minus_mlp']:+.4f}  (need >= +0.02)")
    print(f"  P3  distance alone covers {summary['P3_dist_share_of_gnn_gain']:.0%} of the "
          f"GNN's gain over chance  (need > 50%)")
    print(f"  P4  AUC loss ER -> BA:  " + "   ".join(
        f"{m} {summary['P4_drift'][m]['loss']:+.3f}" for m in ("mlp_features", "gnn")))
    print(f"  P5  GNN AUC by depth:  " + "  ".join(
        f"L={L} {v:.3f}" for L, v in summary["P5_depth"].items()))
    print(f"  embedding vs features (spearman):  " + "  ".join(
        f"{k} {v:+.2f}" for k, v in summary["embedding_spearman"].items()))
    print(f"\nWrote erdos_renyi.csv, depth.csv, drift.csv, embedding.csv, summary.json -> {OUT}")


if __name__ == "__main__":
    main()
