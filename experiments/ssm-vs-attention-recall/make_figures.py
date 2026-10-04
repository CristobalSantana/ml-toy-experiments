"""
make_figures.py -- Three figures.

    python make_figures.py

  fig_recall.png     accuracy against the number of pairs, every model
  fig_capacity.png   capacity against state size; memory against length
  fig_training.png   test accuracy during training - was the budget enough?
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
TESTS = HERE.parent.parent / "generators" / "associative_recall" / "outputs" / "recall_test.npz"

BG = "#0e0e0e"
FG = "#f5f0e8"
GREY = "#8f8f8f"
WARN = "#e69f00"        # attention
RED = "#e05555"         # LTI
SSM_COLOURS = ["#2c5f80", "#1f86c2", "#45b4f0", "#b3e0ff"]    # N = 4 .. 32, dim to bright


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


def _legend(ax, **kw):
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=8.5, **kw)


def models(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(name, label, colour) in drawing order."""
    out = [("attention", "Transformer", WARN)]
    ssm = sorted({n for n in df.name if n.startswith("ssm-")}, key=lambda n: int(n[4:]))
    for name, c in zip(ssm, SSM_COLOURS):
        out.append((name, f"selective SSM, N = {name[4:]}", c))
    for name in sorted({n for n in df.name if n.startswith("lti-")}):
        out.append((name, f"LTI SSM, N = {name[4:]} (no selection)", RED))
    return out


def guess_level() -> pd.Series:
    """Accuracy of answering every query with one of the sequence's own N
    values, chosen uniformly: what a model scores if it has learned which
    tokens are values but not which key each belongs to. Computed from the
    test sets, no model involved. A post-hoc reference, not pre-registered."""
    z = np.load(TESTS)
    out = {}
    for n in z["pair_counts"]:
        tok, tgt = z[f"tokens_{n}"].astype(int), z[f"targets_{n}"].astype(int)
        vals, ans = tok[:, 1:2 * n:2], tgt[:, 2 * n:]
        out[int(n)] = float((vals[:, None, :] == ans[:, :, None]).sum(-1).mean() / n)
    return pd.Series(out)


def recall(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4), facecolor=BG)
    g = guess_level()
    ax.plot(g.index, g.values, ":", color=FG, lw=1.4,
            label="a random value from the sequence (post-hoc reference)")
    for name, label, colour in models(df):
        d = df[df.name == name]
        med = d.groupby("pairs").accuracy.median()
        ls = "--" if name.startswith("lti") else "-"
        ax.plot(med.index, med.values, ls, color=colour, lw=2.2, marker="o", ms=4, label=label)
        ax.scatter(d.pairs, d.accuracy, s=18, color=colour, alpha=0.55, lw=0)
    ax.axhline(0.9, color=GREY, ls=":", lw=1)
    ax.text(4.1, 0.915, "90%", color=GREY, fontsize=8.5)
    ax.set_xscale("log", base=2)
    ticks = sorted(df.pairs.unique())
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_ylim(-0.02, 1.04)
    g.to_frame("accuracy").rename_axis("pairs").to_csv(OUT / "guess_level.csv")
    _style(ax, "Recall against the number of pairs to remember  (lines: median of 3 seeds; dots: seeds)",
           "key-value pairs in the sequence", "query accuracy, 1,000 test sequences")
    _legend(ax, loc="center right", framealpha=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_recall.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_recall.png")


def capacity(cap: pd.DataFrame, mem: pd.DataFrame, s: dict, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), facecolor=BG)
    ax = axes[0]
    k = s["P4"]["median_k90"]
    sizes = sorted(int(n) for n in k)
    for n, c in zip(sizes, SSM_COLOURS):
        seeds = cap[cap.name == f"ssm-{n}"]
        for r in seeds.itertuples():
            marker = {"": "o", "below": "v", "above": "^"}[r.censored if isinstance(r.censored, str) else ""]
            ax.scatter(n, r.k90, s=40, color=c, alpha=0.6, marker=marker, lw=0)
        v = k[str(n)]
        marker = {"": "o", "below": "v", "above": "^"}[v["censored"]]
        ax.scatter(n, v["value"], s=90, color=c, marker=marker, zorder=3, edgecolor=FG, lw=0.8)
    free = [(n, k[str(n)]["value"]) for n in sizes if k[str(n)]["censored"] == ""]
    if len(free) >= 2:
        x = np.log2([a for a, _ in free]); y = np.log2([b for _, b in free])
        slope, icpt = np.polyfit(x, y, 1)
        xs = np.linspace(min(sizes), max(sizes), 50)
        ax.plot(xs, 2 ** (icpt + slope * np.log2(xs)), color=FG, lw=1.2,
                label=f"fit: K90 ~ N^{slope:.2f}")
        ax.plot(xs, free[0][1] * xs / free[0][0], ":", color=GREY, lw=1.2,
                label="proportional (slope 1), through the smallest")
    lti = s["P5"]["lti_k90"]
    ax.scatter(max(sizes) * 1.12, lti["value"], s=90, color=RED,
               marker={"": "o", "below": "v", "above": "^"}[lti["censored"]], edgecolor=FG,
               lw=0.8, label="LTI SSM (no selection)", zorder=3)
    # how many of the three runs at each size learned the task at all (95%+ at 4 pairs)
    for n in sizes:
        at4 = df[(df.name == f"ssm-{n}") & (df.pairs == df.pairs.min())].accuracy
        ax.text(n, 2.35, f"{int((at4 >= 0.95).sum())}/{len(at4)} runs\nlearned", color=GREY,
                fontsize=8, ha="center", va="bottom")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(sizes, [str(n) for n in sizes])
    yt = [2, 4, 8, 16, 32, 64]
    ax.set_yticks(yt, [str(t) for t in yt])
    _style(ax, "Capacity against state size  (v: below 4,  ^: at least 64)",
           "state dimension N  (state = 128 N numbers per layer)",
           "K90: pairs recalled at 90% accuracy")
    _legend(ax, loc="upper left")

    ax = axes[1]
    for name, label, colour in models(mem.rename(columns={})):
        d = mem[mem.name == name].sort_values("length")
        ls = "--" if name.startswith("lti") else "-"
        ax.plot(d.length, d.floats_per_layer, ls, color=colour, lw=2, marker="o", ms=3.5,
                label=label)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    lens = sorted(mem.length.unique())
    ax.set_xticks(lens, [str(t) for t in lens])
    _style(ax, "What each model keeps while reading", "sequence length (tokens)",
           "numbers held per layer at inference")
    _legend(ax, loc="upper left")
    fig.suptitle("A fixed state is cheap - when the model learns to use it", color=FG, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_capacity.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_capacity.png")


def training(cur: pd.DataFrame) -> None:
    """Every seed's test accuracy during training, at three pair counts, with
    the guess level: a run that sits on the dotted line never learned which
    value goes with which key, whatever its state could have held."""
    counts = sorted(cur.pairs.unique())
    show = [c for c in (4, 16, 64) if c in counts]
    g = guess_level()
    fig, axes = plt.subplots(1, len(show), figsize=(15, 4.7), facecolor=BG, sharey=True)
    for ax, k in zip(np.atleast_1d(axes), show):
        for name, label, colour in models(cur):
            ls = "--" if name.startswith("lti") else "-"
            for i, (seed, d) in enumerate(cur[(cur.name == name) & (cur.pairs == k)]
                                          .groupby("seed")):
                ax.plot(d.step, d.accuracy, ls, color=colour, lw=1.5, alpha=0.85,
                        label=label if i == 0 else None)
        ax.axhline(g[k], color=FG, ls=":", lw=1.2)
        if k == show[0]:
            ax.text(cur.step.max(), g[k] + 0.015, "random value from the sequence",
                    color=FG, fontsize=7.5, ha="right")
        ax.set_ylim(-0.02, 1.04)
        _style(ax, f"{k} pairs", "training step",
               "accuracy, 200 test sequences" if k == show[0] else "")
    _legend(np.atleast_1d(axes)[-1], loc="center right", bbox_to_anchor=(1.0, 0.36), framealpha=1.0)
    fig.suptitle("Every seed during training: learned, still climbing, or stuck at guessing?",
                 color=FG, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_training.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_training.png")


if __name__ == "__main__":
    df = pd.read_csv(OUT / "accuracy.csv")
    cap = pd.read_csv(OUT / "capacity.csv", keep_default_na=False)
    cap["k90"] = cap["k90"].astype(float)
    mem = pd.read_csv(OUT / "memory.csv")
    cur = pd.read_csv(OUT / "curves.csv")
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    recall(df)
    capacity(cap, mem, s, df)
    training(cur)
    print(f"-> {OUT}   ({sum(s[k]['passed'] for k in ('P1', 'P2', 'P3', 'P4', 'P5'))}/5 passed)")
