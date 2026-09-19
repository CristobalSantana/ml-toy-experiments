"""
make_figures.py -- Three figures.

    python make_figures.py

  fig_rollout.png          where the two models go when left to run
  fig_learned_energy.png   the scalar the HNN learned, against the real one
  fig_when_wrong.png       the same inductive bias on a system that breaks it
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
ACCENT = "#0284C7"      # MLP
WARN = "#e69f00"        # HNN
GREEN = "#4caf7d"
RED = "#e05555"


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


def wrap(q):
    return np.mod(q + np.pi, 2 * np.pi) - np.pi


def rollout() -> None:
    """Left: three held-out orbits, the reference and both models. Right:
    energy drift over time, median over the 30 rolled-out trajectories."""
    z = np.load(OUT / "rollout_example.npz")
    t = z["t"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=BG)

    ax = axes[0]
    # pick three ICs at different energies: a small swing, a big one, a rotation
    H0 = z["ref_H"][0]
    order = np.argsort(H0)
    picks = [order[len(order) // 6], order[len(order) // 2], order[-len(order) // 5]]
    for j, c in enumerate(picks):
        lab = j == 0
        ax.plot(wrap(z["ref_q"][:, c]), z["ref_p"][:, c], ".", ms=1.2, color=GREY,
                alpha=0.8, label="reference" if lab else None, zorder=1)
        ax.plot(wrap(z["mlp_traj"][:, c, 0]), z["mlp_traj"][:, c, 1], ".", ms=1.2,
                color=ACCENT, alpha=0.7, label="MLP rollout" if lab else None, zorder=2)
        ax.plot(wrap(z["hnn_traj"][:, c, 0]), z["hnn_traj"][:, c, 1], ".", ms=1.2,
                color=WARN, alpha=0.9, label="HNN rollout" if lab else None, zorder=3)
    ax.set_xlim(-np.pi, np.pi)
    _style(ax, "100 time units from three held-out starts",
           "q  (angle, wrapped)", "p  (momentum)")
    leg = ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=9,
                    loc="upper right", markerscale=8)

    ax = axes[1]
    for name, colour, lab in (("mlp", ACCENT, "MLP"), ("hnn", WARN, "HNN")):
        H = z[f"{name}_H"]
        drift = np.abs(H - H[0])
        med = np.median(drift, axis=1)
        lo, hi = np.percentile(drift, [25, 75], axis=1)
        ax.plot(t, med, color=colour, lw=2, label=f"{lab}, median over 30 starts")
        ax.fill_between(t, lo, hi, color=colour, alpha=0.18, lw=0)
    ax.axhline(1.7e-10, color=GREEN, lw=1.2, ls=":")
    ax.text(2, 3e-10, "RK4 on the true field", color=GREEN, fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylim(5e-11, 5)
    _style(ax, "Energy drift  |H(x_t) - H(x_0)|, true H on the model's path",
           "t", "energy drift (log scale)")
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=9, loc="center right")

    fig.suptitle("Same data, same budget, same integrator. One of them is "
                 "built to conserve.", color=FG, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rollout.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_rollout.png")


def learned_energy() -> None:
    """The HNN was never shown H. Left: its H_theta against the true H on a
    grid. Right: its level sets over the true ones."""
    z = np.load(OUT / "learned_energy_grid.npz")
    e = pd.read_csv(OUT / "learned_energy.csv").median(numeric_only=True)
    q, p, Ht, H = z["q"], z["p"], z["H_theta"], z["H_true"]
    a, b = float(z["slope"]), float(z["intercept"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=BG)

    ax = axes[0]
    # the training data covers |p| up to about 2; beyond it the network is
    # extrapolating, and the figure says so rather than hiding it
    inside = np.abs(p) <= 2.0
    ax.scatter(H[~inside], Ht[~inside] - b, s=3, color=GREY, alpha=0.35,
               label="outside the training range of p")
    ax.scatter(H[inside], Ht[inside] - b, s=3, color=WARN, alpha=0.6,
               label="inside")
    lim = [H.min() - 0.2, H.max() + 0.2]
    ax.plot(lim, lim, color=FG, lw=1.2, ls="--", label="H_theta - c = H")
    ax.text(0.03, 0.96, f"affine fit over the training region:\n"
                        f"H_theta = {e['slope']:.4f} H + c,   R² = {e['r2']:.5f}",
            transform=ax.transAxes, va="top", color=FG, fontsize=9.5)
    _style(ax, "The scalar it learned, against the one it was never shown",
           "true H(q, p)", "learned H_theta(q, p), shifted by its constant c")
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=8.5,
              loc="lower right", markerscale=4)

    ax = axes[1]
    levels = np.linspace(-0.9, 2.5, 12)
    ax.contour(q, p, H, levels=levels, colors=GREY, linewidths=1.0, linestyles="--")
    ax.contour(q, p, (Ht - b) / a, levels=levels, colors=WARN, linewidths=1.2)
    ax.contour(q, p, H, levels=[1.0], colors=FG, linewidths=1.6)
    ax.axhline(2.0, color=GREY, lw=0.8, ls=":"); ax.axhline(-2.0, color=GREY, lw=0.8, ls=":")
    ax.text(-3.05, 2.08, "training data ends", color=GREY, fontsize=8)
    ax.plot([], [], color=GREY, ls="--", label="true H, level sets")
    ax.plot([], [], color=WARN, label="learned H_theta, same levels")
    ax.plot([], [], color=FG, label="separatrix, H = 1")
    _style(ax, "Level sets", "q  (angle)", "p  (momentum)")
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=8.5, loc="upper right")

    fig.suptitle("Trained only on dq/dt and dp/dt, the network rediscovered the energy",
                 color=FG, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_learned_energy.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_learned_energy.png")


def when_wrong() -> None:
    """Left: the damped system, where no scalar generates the field. Right:
    extrapolation to energies above anything in training."""
    d = pd.read_csv(OUT / "damped.csv")
    c = pd.read_csv(OUT / "extrapolation.csv")
    i = pd.read_csv(OUT / "ideal.csv")
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), facecolor=BG)

    def column(ax, x, v, colour):
        ax.scatter(np.full(len(v), x) + np.linspace(-0.08, 0.08, len(v)), v,
                   s=90, color=colour, zorder=3)
        ax.plot([x - 0.22, x + 0.22], [np.median(v)] * 2, color=colour, lw=2.4)

    # left: the damped system. On the ideal one the two are close; this is
    # the gap the wrong inductive bias opens
    ax = axes[0]
    for j, (name, colour) in enumerate((("mlp", ACCENT), ("hnn", WARN))):
        column(ax, j, d[d["model"] == name]["field_rmse"].to_numpy(), colour)
    ideal_ratio = (i[i.model == "hnn"].field_rmse.median()
                   / i[i.model == "mlp"].field_rmse.median())
    ax.set_xticks([0, 1], ["MLP", "HNN"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_yscale("log")
    lo, hi = d["field_rmse"].min(), d["field_rmse"].max()
    ax.set_ylim(lo / 1.6, hi * 3.2)              # headroom for the annotation
    ax.text(0.5, 0.96, f"HNN / MLP = {s['P4_ratio']:.1f}x here,\n"
                       f"against {ideal_ratio:.1f}x on the ideal system",
            transform=ax.transAxes, ha="center", va="top", color=FG, fontsize=10.5)
    _style(ax, "Damped pendulum: field error on held-out states", "",
           "field RMSE (log scale)")

    # right: extrapolation. Two sets - the pre-registered held-out
    # trajectories in the band, and the extension to every trajectory in
    # the band (none of which contributed a training state, since energy is
    # conserved and training was restricted to H < 0.5)
    ax = axes[1]
    sets = [("pre-registered", "pre-registered\n(held out, n={n})"),
            ("extended", "extended, post-hoc\n(all in band, n={n})")]
    if "set" not in c.columns:                   # outputs from before the extension
        c = c.assign(set="pre-registered", n_trajectories=2)
        sets = sets[:1]
    ticks, labels = [], []
    for k, (set_name, label) in enumerate(sets):
        sub = c[c["set"] == set_name]
        for j, (name, colour) in enumerate((("mlp", ACCENT), ("hnn", WARN))):
            column(ax, 2.6 * k + j, sub[sub["model"] == name]["state_error_final"].to_numpy(),
                   colour)
            ticks.append(2.6 * k + j); labels.append(name.upper())
        n = int(sub["n_trajectories"].iloc[0])
        ax.text(2.6 * k + 0.5, 0.03, label.format(n=n), transform=ax.get_xaxis_transform(),
                ha="center", color=GREY, fontsize=8.5)
    ax.set_xticks(ticks, labels)
    ax.set_xlim(-0.6, 2.6 * (len(sets) - 1) + 1.6)
    ax.set_yscale("log")
    lo, hi = c["state_error_final"].min(), c["state_error_final"].max()
    ax.set_ylim(lo / 2.5, hi * 3.2)
    p5 = s["P5_extrapolation_state_error"]
    ax.text(0.5, 0.96, f"pre-registered medians: MLP {p5['mlp']:.3f}, HNN {p5['hnn']:.3f}",
            transform=ax.transAxes, ha="center", va="top", color=FG, fontsize=10.5)
    _style(ax, "Trained on H < 0.5, rolled out at 0.5 < H < 0.9", "",
           "state error at t = 100 (log scale)")

    fig.suptitle("A conservation law built in: a floor when it is false, "
                 "a reach when it is true", color=FG, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_when_wrong.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_when_wrong.png")


if __name__ == "__main__":
    rollout()
    learned_energy()
    when_wrong()
    print(f"-> {OUT}")
