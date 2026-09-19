"""
make_figures.py -- Three figures.

    python make_figures.py

  fig_models.png   every model on held-out graphs, and how far one feature gets
  fig_depth.png    what more message-passing layers buy, and what the logit encodes
  fig_drift.png    trained on one degree distribution, scored on another
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

BG = "#0e0e0e"
FG = "#f5f0e8"
GREY = "#8f8f8f"
ACCENT = "#0284C7"
WARN = "#e69f00"
GREEN = "#4caf7d"
RED = "#e05555"
VIOLET = "#9a7fd1"

ORDER = ["logreg_dist", "logreg_degree", "logreg_features", "mlp_features", "gnn"]
LABEL = {"logreg_dist": "logistic regression,\ndistance to seed only",
         "logreg_degree": "logistic regression,\ndegree only (post-hoc)",
         "logreg_features": "logistic regression,\n5 hand-computed features",
         "mlp_features": "MLP,\n5 hand-computed features",
         "gnn": "GNN,\ndegree + seed flag only"}
COLOUR = {"logreg_dist": GREEN, "logreg_degree": RED, "logreg_features": VIOLET,
          "mlp_features": ACCENT, "gnn": WARN}


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=GREY, labelsize=9)
    ax.set_title(title, color=FG, fontsize=11)
    ax.set_xlabel(xlabel, color=GREY, fontsize=9)
    ax.set_ylabel(ylabel, color=GREY, fontsize=9)
    ax.grid(True, color="#2a2a2a", lw=0.7)
    ax.set_axisbelow(True)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(FG)


def models() -> None:
    a = pd.read_csv(OUT / "erdos_renyi.csv")
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(12.5, 5.6), facecolor=BG)

    for j, m in enumerate(ORDER):
        v = a[a["model"] == m]["auc"].to_numpy()
        ax.scatter(np.full(len(v), j) + np.linspace(-0.08, 0.08, len(v)), v,
                   s=90, color=COLOUR[m], zorder=3)
        ax.plot([j - 0.25, j + 0.25], [np.median(v)] * 2, color=COLOUR[m], lw=2.6)
        ax.text(j, np.median(v) + 0.012, f"{np.median(v):.3f}", ha="center",
                color=FG, fontsize=9.5)
    ax.axhline(0.5, color=GREY, lw=1.2, ls=":")
    ax.text(-0.45, 0.503, "chance", color=GREY, fontsize=8.5)

    # P3, drawn: the gap from chance to the GNN, and how much of it one
    # feature covers
    g, d = s["auc_er"]["gnn"], s["auc_er"]["logreg_dist"]
    ax.annotate("", xy=(4.45, g), xytext=(4.45, 0.5),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.2))
    ax.text(4.5, (g + 0.5) / 2, "GNN's gain\nover chance", color=GREY, fontsize=8.5, va="center")
    ax.axhline(d, color=GREEN, lw=1.0, ls="--", xmin=0.05, xmax=0.95)
    ax.text(2.0, d - 0.012, f"distance alone covers {s['P3_dist_share_of_gnn_gain']:.0%} "
                            f"of that gain", ha="center", color=GREEN, fontsize=9.5)
    # the post-hoc ceiling: no predictor that sees only degree can beat this
    ceil = s["posthoc_ceiling"]["degree_only_er"]
    ax.axhline(ceil, color=RED, lw=1.0, ls=":", xmin=0.05, xmax=0.95)
    ax.text(0.5, ceil - 0.009, f"ceiling for any degree-only predictor: {ceil:.3f}",
            ha="center", color=RED, fontsize=9)

    ax.set_xticks(range(5), [LABEL[m] for m in ORDER])
    ax.set_xlim(-0.6, 5.1)
    ax.set_ylim(0.47, max(a["auc"].max(), g) + 0.05)
    _style(ax, "Who gets infected: held-out Erdős–Rényi graphs, 3 seeds each",
           "", "ROC AUC")
    fig.tight_layout()
    fig.savefig(OUT / "fig_models.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_models.png")


def depth() -> None:
    d = pd.read_csv(OUT / "depth.csv")
    e = pd.read_csv(OUT / "embedding.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), facecolor=BG)

    ax = axes[0]
    g = d.groupby("layers").agg(val=("auc_val", "median"), test=("auc_test", "median"),
                                lo=("auc_test", "min"), hi=("auc_test", "max")).reset_index()
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    ax.fill_between(g["layers"], g["lo"], g["hi"], color=WARN, alpha=0.18, lw=0)
    ax.plot(g["layers"], g["test"], "o-", color=WARN, lw=2.2, ms=6, label="GNN, held-out AUC")
    ax.plot(g["layers"], g["val"], "s--", color=GREY, lw=1.4, ms=5, label="GNN, validation AUC (picks L)")
    # the reference lines the depth curve should be read against
    ax.axhline(s["auc_er"]["mlp_features"], color=ACCENT, lw=1.4, ls="--")
    ax.text(6.9, s["auc_er"]["mlp_features"] - 0.007, "MLP on 5 hand-computed features",
            ha="right", color=ACCENT, fontsize=8.5)
    ax.axhline(s["posthoc_ceiling"]["degree_only_er"], color=RED, lw=1.2, ls=":")
    ax.text(1.02, s["posthoc_ceiling"]["degree_only_er"] - 0.006, "degree-only ceiling",
            color=RED, fontsize=8.5)
    ax.axhline(s["auc_er"]["logreg_dist"], color=GREEN, lw=1.2, ls=":")
    ax.text(1.02, s["auc_er"]["logreg_dist"] + 0.002, "distance to seed only",
            color=GREEN, fontsize=8.5)
    # typical seed-to-node distance on these graphs: log(n)/log(<k>) ~ 3.2
    ax.axvspan(3, 4, color="#1c1c1c", zorder=0)
    ax.text(3.5, 0.585, "typical seed-to-node\ndistance", ha="center", color=GREY, fontsize=8.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 3, 4, 6, 8], ["1", "2", "3", "4", "6", "8"])
    # a y-axis wide enough that a 0.003 spread looks like what it is
    ax.set_ylim(0.55, 0.70)
    _style(ax, "More layers buy nothing: one hop is already the best",
           "message-passing layers L", "ROC AUC")
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=9, loc="lower right")

    ax = axes[1]
    m = e.groupby("feature")["spearman"].median().reindex(
        ["dist_to_seed", "degree", "log_degree", "pagerank", "clustering"])
    y = np.arange(len(m))
    for i, (f, v) in enumerate(m.items()):
        ax.plot([0, v], [i, i], color="#2a2a2a", lw=2)
        ax.scatter([v], [i], s=110, color=WARN if abs(v) == abs(m).max() else GREY, zorder=3)
        ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", color=FG, fontsize=9.5)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_yticks(y, [f.replace("_", " ") for f in m.index])
    ax.set_xlim(-1.05, 1.05)
    _style(ax, "What the GNN's output correlates with (Spearman, held out)",
           "rank correlation of the GNN logit with each hand-computed feature", "")

    fig.suptitle("The GNN was given only degree and a seed flag. This is what it built.",
                 color=FG, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_depth.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_depth.png")


def drift() -> None:
    c = pd.read_csv(OUT / "drift.csv")
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(10.5, 5.4), facecolor=BG)

    g = c.groupby("model")[["auc_er", "auc_ba"]].median()
    for m in ORDER:
        ax.plot([0, 1], [g.loc[m, "auc_er"], g.loc[m, "auc_ba"]], "o-", color=COLOUR[m],
                lw=2.4, ms=8, label=LABEL[m].replace("\n", " "))
    # right-hand loss labels, pushed apart so three models within 0.005 AUC
    # of each other do not print on top of one another
    ys = sorted(((float(g.loc[m, "auc_ba"]), m) for m in ORDER))
    placed = []
    for y, m in ys:
        if placed and y - placed[-1] < 0.0065:
            y = placed[-1] + 0.0065
        placed.append(y)
        loss = g.loc[m, "auc_er"] - g.loc[m, "auc_ba"]
        ax.annotate(f"−{loss:.3f}" if loss > 0 else f"+{-loss:.3f}",
                    xy=(1, g.loc[m, "auc_ba"]), xytext=(1.05, y), va="center",
                    color=COLOUR[m], fontsize=9.5,
                    arrowprops=dict(arrowstyle="-", color=COLOUR[m], lw=0.6, alpha=0.6))
    ax.axhline(0.5, color=GREY, lw=1.2, ls=":")
    ax.set_xticks([0, 1], ["Erdős–Rényi\n(trained here)", "Barabási–Albert\n(never seen: hubs, heavy tail)"])
    ax.set_xlim(-0.3, 1.45)
    _style(ax, "Trained on one degree distribution, scored on another",
           "", "ROC AUC on major outbreaks")
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=8.5, loc="lower left")
    gl = s["P4_drift"]["gnn"]["loss"]; ml = s["P4_drift"]["mlp_features"]["loss"]
    fig.text(0.5, 0.015, f"Right-hand labels are the AUC lost. P4 predicted the GNN would "
             f"lose less than the feature MLP; it lost {gl:.3f} against {ml:.3f}.",
             ha="center", color=GREY, fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(OUT / "fig_drift.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_drift.png")


if __name__ == "__main__":
    models()
    depth()
    drift()
    print(f"-> {OUT}")
