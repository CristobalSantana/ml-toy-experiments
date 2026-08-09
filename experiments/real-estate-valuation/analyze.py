"""
analyze.py -- Aggregate the CV results and apply CRITERIA.md's decision rules.

Reads outputs/cv_results.csv and writes:

    outputs/comparison_table.csv    per-arm aggregate, mean ± sd over folds
    outputs/comparison_table.md     the same, formatted for the README
    outputs/decision.md             the pre-registered rules, evaluated

Two things this script is careful about:

* Every number is a mean ± spread over folds, never a single run, as
  CRITERIA.md requires.
* TabPFN could not run on 5 of the 25 regime-limited folds, because whole-
  manzana subsampling overshot its 5,000-row CPU ceiling on those. Comparing
  its 20-fold mean against the classical models' 25-fold means would compare
  different folds, and folds differ a lot in this arm. So the decision rule is
  evaluated on the *paired* subset - the folds where every model ran - with
  the unpaired figures reported alongside for transparency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_ROOT / "outputs"
RESULTS_CSV = OUTPUT_DIR / "cv_results.csv"

CLASSICAL = ["ridge", "random_forest", "lightgbm", "catboost", "mlp"]


def load() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        sys.exit(f"Missing {RESULTS_CSV} - run run_cross_sectional.py first.")
    df = pd.read_csv(RESULTS_CSV)
    failed = df["mae"].isna().sum()
    if failed:
        print(f"WARNING: {failed} recorded fits failed and are excluded")
    return df[df["mae"].notna()].copy()


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± sd per (arm, model), plus total cost = fit + predict."""
    df = df.copy()
    df["total_seconds"] = df["fit_seconds"] + df["predict_seconds"]
    g = df.groupby(["arm", "model"]).agg(
        n_folds=("mae", "size"),
        n_train=("n_train", "mean"),
        mae=("mae", "mean"), mae_sd=("mae", "std"),
        r2=("r2", "mean"), r2_sd=("r2", "std"),
        mdape=("mdape_uf_m2", "mean"), mdape_sd=("mdape_uf_m2", "std"),
        fit_s=("fit_seconds", "mean"), fit_s_sd=("fit_seconds", "std"),
        predict_s=("predict_seconds", "mean"),
        total_s=("total_seconds", "mean"),
        peak_mem_mb=("peak_memory_mb", "mean"),
    ).reset_index()
    return g.sort_values(["arm", "mae"])


def paired_subset(df: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Restrict an arm to the (seed, fold) cells where *every* model ran."""
    a = df[df["arm"] == arm]
    if a.empty:
        return a
    n_models = a["model"].nunique()
    counts = a.groupby(["seed", "fold"])["model"].nunique()
    complete = set(counts[counts == n_models].index)
    return a[a.set_index(["seed", "fold"]).index.isin(complete)]


def decide(df: pd.DataFrame, cfg: dict) -> str:
    """Evaluate CRITERIA.md's win/loss rules on the regime-limited arm.

    That is the only arm where the rule is even defined: TabPFN cannot be run
    on the full arm at all, which is itself the headline constraint.
    """
    lines: list[str] = []
    a = paired_subset(df, "regime_limited")
    if a.empty or "tabpfn" not in set(a["model"]):
        return "No paired regime-limited data including TabPFN; rule not evaluable.\n"

    a = a.copy()
    a["total_seconds"] = a["fit_seconds"] + a["predict_seconds"]
    g = a.groupby("model").agg(
        mae=("mae", "mean"), sd=("mae", "std"),
        fit_s=("fit_seconds", "mean"), total_s=("total_seconds", "mean"),
        mem=("peak_memory_mb", "mean"), n=("mae", "size"),
    )

    n_cells = a.groupby(["seed", "fold"]).ngroups
    t = g.loc["tabpfn"]
    classical = g.loc[[m for m in CLASSICAL if m in g.index]]
    best = classical["mae"].idxmin()
    c = classical.loc[best]

    E_c, E_t = float(c["mae"]), float(t["mae"])
    s = float(c["sd"]) + float(t["sd"])          # pooled spread, as pre-registered
    time_ratio = float(t["total_s"]) / float(c["total_s"])
    fit_ratio = float(t["fit_s"]) / max(float(c["fit_s"]), 1e-9)
    mem_ratio = float(t["mem"]) / max(float(c["mem"]), 1e-9)

    lines.append(f"Evaluated on the {n_cells} paired (seed, fold) cells where every "
                 f"model ran.\n")
    lines.append(f"- Best classical model: **{best}**, E_c = {E_c:.4f} ± {c['sd']:.4f}")
    lines.append(f"- TabPFN:               E_t = {E_t:.4f} ± {t['sd']:.4f}")
    lines.append(f"- Pooled spread s = {s:.4f}; |E_c - E_t| = {abs(E_c - E_t):.4f}\n")
    lines.append(f"- Cost ratios (TabPFN / {best}): total time {time_ratio:.1f}x, "
                 f"fit time {fit_ratio:.2f}x, peak memory {mem_ratio:.1f}x\n")

    cost_advantage = (time_ratio >= 5) or (mem_ratio >= 5)
    if E_c < E_t - s:
        verdict = ("**Classical methods win** (rule 1): the best classical model is more "
                   "accurate and the distributions separate.")
    elif E_c <= E_t + s and cost_advantage:
        verdict = (f"**Classical methods win** (rule 2): accuracy is statistically "
                   f"indistinguishable (|E_c - E_t| = {abs(E_c-E_t):.4f} <= s = {s:.4f}) "
                   f"and {best} reaches it at "
                   f"{'>=5x lower total time' if time_ratio >= 5 else ''}"
                   f"{' and ' if time_ratio >= 5 and mem_ratio >= 5 else ''}"
                   f"{'>=5x lower peak memory' if mem_ratio >= 5 else ''}.")
    elif E_t < E_c - s:
        verdict = ("**TabPFN wins**, regime-limited: more accurate with separating "
                   "distributions, inside its documented envelope only.")
    else:
        verdict = ("**Inconclusive**: the distributions overlap and no >=5x cost gap "
                   "exists.")
    lines.append(verdict + "\n")
    return "\n".join(lines)


def to_markdown(g: pd.DataFrame) -> str:
    out = []
    for arm in ["full", "regime_limited"]:
        a = g[g["arm"] == arm]
        if a.empty:
            continue
        title = ("Full data (five classical models; TabPFN cannot ingest this many rows)"
                 if arm == "full" else
                 "Regime-limited (all six models, inside TabPFN's CPU envelope)")
        out.append(f"### {arm} - {title}\n")
        out.append("| Model | Folds | Train rows | MAE (log10) | R² | MdAPE % | "
                   "Fit s | Predict s | Peak MB |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in a.iterrows():
            out.append(
                f"| {r['model']} | {int(r['n_folds'])} | {int(r['n_train']):,} | "
                f"{r['mae']:.4f} ± {r['mae_sd']:.4f} | {r['r2']:.3f} | {r['mdape']:.1f} | "
                f"{r['fit_s']:.2f} | {r['predict_s']:.3f} | {r['peak_mem_mb']:.0f} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    cfg = yaml.safe_load((EXPERIMENT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    df = load()
    g = aggregate(df)

    g.to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)
    md = to_markdown(g)
    (OUTPUT_DIR / "comparison_table.md").write_text(md, encoding="utf-8")

    decision = decide(df, cfg)
    (OUTPUT_DIR / "decision.md").write_text(
        "# Pre-registered decision rules, evaluated\n\n"
        "Rules fixed in CRITERIA.md before any result was seen.\n\n" + decision,
        encoding="utf-8")

    print(md)
    print("\n" + "=" * 70 + "\nDECISION\n" + "=" * 70)
    print(decision)
    print(f"Wrote comparison_table.csv/.md and decision.md -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
