"""
make_figures.py -- Three figures.

    python make_figures.py

  fig_resolution.png   error against grid size, for every model and mode
  fig_fronts.png       what each answer looks like at one sharp front
  fig_spectrum.png     the detail each answer has, wavenumber by wavenumber
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
ACCENT = "#0284C7"      # FNO, trained at 64
WARN = "#e69f00"        # interpolation
GREEN = "#4caf7d"       # FNO, trained at 256
RED = "#e05555"         # CNN


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


def interp_linear(u, n):
    m = u.shape[-1]
    pos = np.arange(n) * m / n
    i0 = np.floor(pos).astype(int)
    w = pos - i0
    return u[..., i0 % m] * (1 - w) + u[..., (i0 + 1) % m] * w


def resolution(df: pd.DataFrame, fl: pd.DataFrame) -> None:
    """One panel per viscosity: error on the evaluation grid against its
    size, median over seeds, every seed as a faint dot."""
    nus = sorted(df.nu.unique(), reverse=True)
    fig, axes = plt.subplots(1, len(nus), figsize=(15, 4.9), facecolor=BG, sharey=True)
    lines = [
        ("FNO trained at 64, run on the finer grid", dict(model="fno", train_res=64, mode="zero_shot"), ACCENT, "-"),
        ("FNO trained at 64, its 64-pt output interpolated", dict(model="fno", train_res=64, mode="interp_linear"), WARN, "-"),
        ("CNN trained at 64, run on the finer grid", dict(model="cnn", train_res=64, mode="zero_shot"), RED, "-"),
        ("FNO trained at 256", dict(model="fno", train_res=256, mode="zero_shot"), GREEN, "--"),
    ]
    for ax, nu in zip(axes, nus):
        d = df[df.nu == nu]
        for label, where, colour, ls in lines:
            sel = d
            for k, v in where.items():
                sel = sel[sel[k] == v]
            if sel.empty:
                continue
            med = sel.groupby("eval_res").error.median()
            ax.plot(med.index, med.values, ls, color=colour, lw=2, marker="o", ms=4, label=label)
            ax.scatter(sel.eval_res, sel.error, s=10, color=colour, alpha=0.35, lw=0)
        f = fl[(fl.nu == nu) & (fl["mode"] == "interp_linear") & (fl.eval_res > 64)]
        ax.plot(f.eval_res, f.error, ":", color=FG, lw=1.4,
                label="the truth sampled at 64, interpolated")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([64, 128, 256, 512], ["64", "128", "256", "512"])
        _style(ax, f"nu = {nu:g}", "evaluation grid (points)",
               "relative L2 error, 200 held-out starts" if ax is axes[0] else "")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, facecolor=BG, edgecolor=GREY,
               labelcolor=FG, fontsize=9)
    fig.suptitle("Trained on 64 points, asked on finer grids", color=FG, fontsize=12.5)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(OUT / "fig_resolution.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_resolution.png")


def fronts(ex: dict, nu: float) -> None:
    """The steepest front among the saved examples, at nu, zoomed: the
    truth, the 64 points the model was trained on, and every answer at 512."""
    x = ex["x512"]
    truth = ex[f"truth_nu{nu:g}"]
    grad = np.abs(np.diff(truth, axis=1, append=truth[:, :1]))
    j, i = np.unravel_index(np.argmax(grad), grad.shape)
    sl = np.arange(i - 24, i + 25) % 512
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=BG)
    # left: the whole sample, nothing zoomed
    ax = axes[0]
    ax.plot(x, truth[j], color=FG, lw=1.6, label="truth")
    ax.plot(x, ex[f"nu{nu:g}_fno64_zero_shot_512"][j], color=ACCENT, lw=1.3,
            label="FNO trained at 64, run at 512")
    ax.axvspan(x[sl[0]], x[sl[-1]], color=GREY, alpha=0.15, lw=0)
    _style(ax, f"one held-out start, nu = {nu:g}", "x", "u at t = 1")
    _legend(ax, loc="best")
    # right: the front
    ax = axes[1]
    xs = np.unwrap(x[sl], period=2 * np.pi)
    own = ex[f"nu{nu:g}_fno64_zero_shot_64"][j]
    ax.plot(xs, truth[j][sl], color=FG, lw=2.2, label="truth (512 points)")
    ax.plot(xs, ex[f"nu{nu:g}_fno64_zero_shot_512"][j][sl], color=ACCENT, lw=1.8,
            label="FNO trained at 64, run at 512")
    ax.plot(xs, interp_linear(own, 512)[sl], color=WARN, lw=1.5,
            label="FNO at 64, interpolated")
    key = f"nu{nu:g}_fno256_zero_shot_512"
    if key in ex:
        ax.plot(xs, ex[key][j][sl], "--", color=GREEN, lw=1.5, label="FNO trained at 256, run at 512")
    on64 = sl[sl % 8 == 0]
    ax.plot(np.unwrap(x[on64], period=2 * np.pi), own[on64 // 8], "o", ms=6, color=ACCENT,
            mfc=BG, mew=1.6, label="FNO at its 64 training points")
    _style(ax, "the steepest front, zoomed", "x", "")
    _legend(ax, loc="best")
    fig.suptitle("Between the coarse points, what does the operator put?", color=FG, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fronts.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_fronts.png")


def spectrum(ex: dict, nus) -> None:
    """Mean energy per wavenumber at 512 points, over the saved examples."""
    fig, axes = plt.subplots(1, len(nus), figsize=(15, 4.6), facecolor=BG, sharey=True)

    def e(u):
        return (np.abs(np.fft.rfft(u, axis=1)) ** 2).mean(0)[1:]

    for ax, nu in zip(axes, nus):
        k = np.arange(1, 257)
        ax.semilogy(k, e(ex[f"truth_nu{nu:g}"]), color=FG, lw=2, label="truth")
        ax.semilogy(k, e(ex[f"nu{nu:g}_fno64_zero_shot_512"]), color=ACCENT, lw=1.4,
                    label="FNO trained at 64, run at 512")
        ax.semilogy(k, e(interp_linear(ex[f"nu{nu:g}_fno64_zero_shot_64"], 512)), color=WARN,
                    lw=1.2, label="FNO at 64, interpolated")
        key = f"nu{nu:g}_fno256_zero_shot_512"
        if key in ex:
            ax.semilogy(k, e(ex[key]), "--", color=GREEN, lw=1.3, label="FNO trained at 256")
        ax.axvline(32, color=GREY, ls=":", lw=1)
        ax.text(34, 1e-9, "highest mode\na 64-pt grid holds", color=GREY, fontsize=8)
        ax.set_ylim(1e-12, None)
        _style(ax, f"nu = {nu:g}", "wavenumber k", "mean energy in mode k" if ax is axes[0] else "")
    _legend(axes[0], loc="upper right")
    fig.suptitle("The detail each answer contains, at 512 points", color=FG, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_spectrum.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  fig_spectrum.png")


if __name__ == "__main__":
    df = pd.read_csv(OUT / "errors.csv")
    fl = pd.read_csv(OUT / "floors.csv")
    z = np.load(OUT / "examples.npz")
    ex = {k: z[k] for k in z.files}
    nus = sorted(df.nu.unique(), reverse=True)
    resolution(df, fl)
    fronts(ex, min(nus))
    spectrum(ex, nus)
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    print(f"-> {OUT}   ({sum(s[k]['passed'] for k in ('P1', 'P2', 'P3', 'P4', 'P5'))}/5 passed)")
