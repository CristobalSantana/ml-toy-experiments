"""
run.py -- End-to-end entry point for the KAN vs MLP battery-diffusion
experiment.

    python run.py

reproduces the whole experiment from config.yaml:
  1. solves the reference diffusion equation (generators/diffusion_1d),
  2. trains both architectures -- a standard MLP and a from-scratch KAN
     (models/mlp, models/kan) -- as physics-informed neural networks,
     across every seed in config.yaml,
  3. evaluates each trained model against the reference solution,
  4. writes the comparison table + article figures to outputs/ and the
     trained weights to checkpoints/.

Every physical constant, architecture choice, and training hyperparameter
lives in config.yaml -- nothing reproducible is hardcoded below. Expected
runtime: ~45-60 minutes on CPU for the default config (5 seeds x 2
architectures x 3000 epochs).
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
import matplotlib.pyplot as plt

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from generators.diffusion_1d.generate import (  # noqa: E402
    DiffusionParams, solve_diffusion_1d, save_solution, plot_solution, check_mass_balance,
)
from models.kan.kan import KAN  # noqa: E402
from models.mlp.mlp import MLP  # noqa: E402
from experiments.kan_vs_mlp_battery_diffusion.pinn import (  # noqa: E402
    DimensionlessProblem, sample_collocation_points, physics_informed_loss,
)

CONFIG_PATH = EXPERIMENT_ROOT / "config.yaml"
GENERATOR_OUTPUT_DIR = REPO_ROOT / "generators" / "diffusion_1d" / "outputs"
OUTPUT_DIR = EXPERIMENT_ROOT / "outputs"
CHECKPOINT_DIR = EXPERIMENT_ROOT / "checkpoints"

ARCH_COLORS = {"mlp": "tab:blue", "kan": "tab:orange"}
ARCH_LABELS = {"mlp": "MLP", "kan": "KAN"}


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def count_mlp_params(layers: list[int]) -> int:
    return sum(layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1))


def count_kan_params(layers: list[int], grid_size: int, spline_order: int) -> int:
    per_edge = 1 + grid_size + spline_order
    return sum(layers[i] * layers[i + 1] * per_edge for i in range(len(layers) - 1))


# ============================================================ step 1 =====
def generate_reference(config: dict, outdir: Path = GENERATOR_OUTPUT_DIR):
    """Solve the reference diffusion equation with the configured physical
    parameters and save it into the generator's own outputs/ folder. This
    is what makes the experiment runnable end to end from a clean clone --
    no pre-existing reference file is assumed."""
    params = DiffusionParams(**config["physics"])
    sol = solve_diffusion_1d(params)

    rel_err = check_mass_balance(sol)
    print(f"Reference solution: mass-balance relative error = {rel_err:.2e}", flush=True)

    save_solution(sol, outdir)
    plot_solution(sol, outdir / "diffusion_1d_solution.png")
    return sol.x, sol.t, sol.C, params


# ============================================================ step 2 =====
def build_reference_grid(x, t, C_ref, problem, params, device):
    X_hat, T_hat = np.meshgrid(x / params.L, t / params.t_end, indexing="xy")  # (nt+1, nx)
    xt_grid = torch.tensor(np.stack([X_hat.ravel(), T_hat.ravel()], axis=1),
                            dtype=torch.float32, device=device)
    c_hat_ref = (C_ref - params.c0) / problem.c_scale
    c_hat_ref_flat = torch.tensor(c_hat_ref.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
    return xt_grid, c_hat_ref_flat


def train_and_evaluate(model, points, problem, xt_grid, c_hat_ref_flat, lr, n_epochs, log_every, name):
    """Physics-informed training loop. At every `log_every` checkpoint,
    also logs the relative-L2 error against ground truth (a cheap
    forward-pass-only diagnostic under torch.no_grad -- it does not
    influence training)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    t0 = time.perf_counter()
    for epoch in range(n_epochs + 1):
        optimizer.zero_grad()
        loss, parts = physics_informed_loss(model, points, problem)
        loss.backward()
        optimizer.step()
        if epoch % log_every == 0:
            with torch.no_grad():
                pred = model(xt_grid)
                rel_l2 = (torch.linalg.norm(pred - c_hat_ref_flat)
                          / torch.linalg.norm(c_hat_ref_flat)).item()
            history.append(dict(epoch=epoch, total=loss.item(), rel_l2=rel_l2, **parts))
    elapsed = time.perf_counter() - t0
    print(f"  [{name}] done in {elapsed:.1f}s -- final loss={history[-1]['total']:.3e}, "
          f"rel_l2={history[-1]['rel_l2']:.3e}", flush=True)
    return history, elapsed


def physical_errors(model, xt_grid, C_ref, params, c_scale):
    """Final absolute-error statistics in physical units [mol/m^3]."""
    with torch.no_grad():
        c_hat_pred = model(xt_grid).cpu().numpy().reshape(C_ref.shape)
    C_pred = params.c0 + c_scale * c_hat_pred
    abs_err = np.abs(C_pred - C_ref)
    return dict(mean_abs_error=float(abs_err.mean()), max_abs_error=float(abs_err.max()))


def run_sweep(config: dict, x, t, C_ref, params,
              output_dir: Path = OUTPUT_DIR, checkpoint_dir: Path = CHECKPOINT_DIR) -> dict:
    """Train both architectures across every configured seed, using SHARED
    collocation points per seed (apples-to-apples) but independently
    seeded initialization per architecture. Returns the full raw payload
    (also saved as outputs/phase3_results.json)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = config["training"]["device"]
    problem = DimensionlessProblem.from_params(params)
    xt_grid, c_hat_ref_flat = build_reference_grid(x, t, C_ref, problem, params, device)

    mlp_cfg, kan_cfg = config["architectures"]["mlp"], config["architectures"]["kan"]
    n_mlp = count_mlp_params(mlp_cfg["layers"])
    n_kan = count_kan_params(kan_cfg["layers"], kan_cfg["grid_size"], kan_cfg["spline_order"])

    tr = config["training"]
    seeds, n_epochs, lr, log_every = tr["seeds"], tr["n_epochs"], tr["learning_rate"], tr["log_every"]
    cp = tr["collocation_points"]
    n_pde, n_bc, n_ic = cp["pde"], cp["bc"], cp["ic"]

    print(f"MLP {mlp_cfg['layers']} -> {n_mlp} params | "
          f"KAN {kan_cfg['layers']} grid={kan_cfg['grid_size']} order={kan_cfg['spline_order']} "
          f"-> {n_kan} params", flush=True)
    print(f"seeds={seeds}  epochs={n_epochs}  pde/bc/ic points={n_pde}/{n_bc}/{n_ic}\n", flush=True)

    archs = {
        "mlp": (lambda: MLP(mlp_cfg["layers"]).to(device), n_mlp),
        "kan": (lambda: KAN(kan_cfg["layers"], grid_size=kan_cfg["grid_size"],
                            spline_order=kan_cfg["spline_order"]).to(device), n_kan),
    }

    results = {"mlp": [], "kan": []}
    for seed in seeds:
        print(f"=== seed {seed} ===", flush=True)
        # Shared collocation points for this seed, used by BOTH architectures
        # below -- what makes the comparison apples-to-apples rather than
        # each architecture also seeing different "training locations".
        torch.manual_seed(seed)
        points = sample_collocation_points(n_pde, n_bc, n_ic, device=device)

        for arch_name, (build_model, n_params) in archs.items():
            torch.manual_seed(seed)  # each architecture still gets its own seeded init
            model = build_model()
            history, elapsed = train_and_evaluate(
                model, points, problem, xt_grid, c_hat_ref_flat, lr, n_epochs, log_every,
                f"{arch_name} seed={seed}")
            errs = physical_errors(model, xt_grid, C_ref, params, problem.c_scale)

            torch.save(model.state_dict(), checkpoint_dir / f"{arch_name}_seed{seed}.pt")
            results[arch_name].append(dict(
                seed=seed, n_params=n_params, wall_time=elapsed,
                time_per_epoch_ms=1000 * elapsed / n_epochs,
                final_loss=history[-1]["total"], final_rel_l2=history[-1]["rel_l2"],
                **errs, history=history,
            ))

    payload = dict(
        seeds=seeds, n_epochs=n_epochs, log_every=log_every, n_pde=n_pde, n_bc=n_bc, n_ic=n_ic, lr=lr,
        mlp_layers=mlp_cfg["layers"], kan_layers=kan_cfg["layers"],
        kan_grid_size=kan_cfg["grid_size"], kan_spline_order=kan_cfg["spline_order"],
        fourier=problem.fourier, bc0_target=problem.bc0_target, c_scale=problem.c_scale,
        results=results,
    )
    (output_dir / "phase3_results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved raw results -> {output_dir / 'phase3_results.json'}", flush=True)
    return payload


# ============================================================ step 3 =====
def write_comparison_table(results: dict, output_dir: Path = OUTPUT_DIR) -> None:
    rows = []
    for arch, runs in results.items():
        for r in runs:
            rows.append(dict(architecture=arch.upper(), seed=r["seed"], n_params=r["n_params"],
                              final_loss=r["final_loss"], final_rel_l2=r["final_rel_l2"],
                              mean_abs_error=r["mean_abs_error"], max_abs_error=r["max_abs_error"],
                              wall_time_s=r["wall_time"], time_per_epoch_ms=r["time_per_epoch_ms"]))
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(output_dir / "results_raw.csv", index=False)

    agg = raw_df.groupby("architecture").agg(
        n_params=("n_params", "first"),
        final_loss_mean=("final_loss", "mean"), final_loss_std=("final_loss", "std"),
        rel_l2_mean=("final_rel_l2", "mean"), rel_l2_std=("final_rel_l2", "std"),
        mean_abs_error_mean=("mean_abs_error", "mean"), mean_abs_error_std=("mean_abs_error", "std"),
        wall_time_mean=("wall_time_s", "mean"), wall_time_std=("wall_time_s", "std"),
        time_per_epoch_ms=("time_per_epoch_ms", "mean"),
    ).reset_index()
    agg.to_csv(output_dir / "comparison_table.csv", index=False)

    lines = [
        "| Architecture | Params | Final loss | Rel. L2 error vs FD | Mean abs error [mol/m³] | Wall time [s] | Time/epoch [ms] |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in agg.iterrows():
        lines.append(
            f"| {row['architecture']} | {int(row['n_params'])} | "
            f"{row['final_loss_mean']:.2e} ± {row['final_loss_std']:.2e} | "
            f"{row['rel_l2_mean']:.2e} ± {row['rel_l2_std']:.2e} | "
            f"{row['mean_abs_error_mean']:.1f} ± {row['mean_abs_error_std']:.1f} | "
            f"{row['wall_time_mean']:.1f} ± {row['wall_time_std']:.1f} | "
            f"{row['time_per_epoch_ms']:.2f} |"
        )
    (output_dir / "comparison_table.md").write_text("\n".join(lines))
    print(f"Saved comparison_table.csv / .md -> {output_dir}", flush=True)
    print("\n" + "\n".join(lines))


# ============================================================ step 4 =====
def plot_convergence_curves(payload, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for arch in ["mlp", "kan"]:
        runs = payload["results"][arch]
        epochs = np.array([h["epoch"] for h in runs[0]["history"]])
        loss = np.array([[h["total"] for h in r["history"]] for r in runs])
        rel_l2 = np.array([[h["rel_l2"] for h in r["history"]] for r in runs])
        label = f"{ARCH_LABELS[arch]} ({runs[0]['n_params']} params)"
        color = ARCH_COLORS[arch]

        m, s = loss.mean(0), loss.std(0)
        axes[0].plot(epochs, m, label=label, color=color)
        axes[0].fill_between(epochs, np.clip(m - s, 1e-12, None), m + s, alpha=0.2, color=color)

        m, s = rel_l2.mean(0), rel_l2.std(0)
        axes[1].plot(epochs, m, label=label, color=color)
        axes[1].fill_between(epochs, np.clip(m - s, 1e-12, None), m + s, alpha=0.2, color=color)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("physics-informed loss")
    axes[0].set_title("Training signal (PDE + BC + IC residual)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("relative L2 error vs. FD reference")
    axes[1].set_title("True solution error (held-out ground truth)")
    for ax in axes:
        ax.legend()
    fig.suptitle(f"Convergence across {len(payload['seeds'])} seeds (shaded band = ±1 std)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_error_distribution(payload, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    data_by_arch = [[r["final_rel_l2"] for r in payload["results"][arch]] for arch in ["mlp", "kan"]]
    labels = [f"{ARCH_LABELS[a]}\n({payload['results'][a][0]['n_params']} params)" for a in ["mlp", "kan"]]

    ax.boxplot(data_by_arch, showmeans=True, widths=0.5)
    ax.set_xticks([1, 2], labels)

    rng = np.random.default_rng(0)
    for i, (arch, vals) in enumerate(zip(["mlp", "kan"], data_by_arch), start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=ARCH_COLORS[arch], zorder=3)

    ax.set_yscale("log")
    ax.set_ylabel("final relative L2 error vs. FD reference")
    ax.set_title(f"Sensitivity to initialization across {len(payload['seeds'])} seeds")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cost_accuracy(payload, save_path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for arch in ["mlp", "kan"]:
        runs = payload["results"][arch]
        times = [r["wall_time"] for r in runs]
        errs = [r["final_rel_l2"] for r in runs]
        ax.scatter(times, errs, color=ARCH_COLORS[arch],
                   label=f"{ARCH_LABELS[arch]} ({runs[0]['n_params']} params)", s=60, alpha=0.85)
        ax.scatter(np.mean(times), np.exp(np.mean(np.log(errs))), color=ARCH_COLORS[arch],
                   marker="*", s=300, edgecolor="black", linewidth=0.8, zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("training wall-clock time [s]  (fixed epoch budget)")
    ax.set_ylabel("final relative L2 error vs. FD reference")
    ax.set_title("Cost-accuracy trade-off  (★ = geometric-mean seed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_solution_fit(payload, x, t, C_ref, params, problem, config, checkpoint_dir, save_path):
    mlp_cfg, kan_cfg = config["architectures"]["mlp"], config["architectures"]["kan"]
    device = config["training"]["device"]

    def load_model(arch, seed):
        if arch == "mlp":
            model = MLP(mlp_cfg["layers"])
        else:
            model = KAN(kan_cfg["layers"], grid_size=kan_cfg["grid_size"], spline_order=kan_cfg["spline_order"])
        model.load_state_dict(torch.load(checkpoint_dir / f"{arch}_seed{seed}.pt", map_location=device))
        model.eval()
        return model

    best = {arch: min(payload["results"][arch], key=lambda r: r["final_rel_l2"])["seed"]
            for arch in ["mlp", "kan"]}
    models = {arch: load_model(arch, seed) for arch, seed in best.items()}

    t_end, L, c0, c_scale = params.t_end, params.L, params.c0, problem.c_scale
    snapshot_fracs = [0.0, 1 / 3, 2 / 3, 1.0]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)

    x_hat = torch.tensor(x / L, dtype=torch.float32).unsqueeze(1)
    for ax, frac in zip(axes, snapshot_fracs):
        t_phys = frac * t_end
        n_idx = int(round(frac * (len(t) - 1)))
        ref_slice = C_ref[n_idx]

        t_hat_col = torch.full_like(x_hat, frac)
        xt = torch.cat([x_hat, t_hat_col], dim=1)
        with torch.no_grad():
            preds = {arch: (c0 + c_scale * models[arch](xt).numpy().ravel()) for arch in ["mlp", "kan"]}

        ax.plot(x * 1e6, ref_slice, "k-", linewidth=2, label="FD reference")
        ax.plot(x * 1e6, preds["mlp"], "--", color=ARCH_COLORS["mlp"], label=f"MLP (seed {best['mlp']})")
        ax.plot(x * 1e6, preds["kan"], ":", color=ARCH_COLORS["kan"], linewidth=2.5,
                label=f"KAN (seed {best['kan']})")
        ax.set_title(f"t = {t_phys / 60:.1f} min")
        ax.set_xlabel("x [µm]")
    axes[0].set_ylabel("c [mol/m³]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Best-of-seeds fit vs. finite-difference reference")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def make_figures(payload, x, t, C_ref, params, config,
                  output_dir: Path = OUTPUT_DIR, checkpoint_dir: Path = CHECKPOINT_DIR) -> None:
    problem = DimensionlessProblem.from_params(params)
    plot_convergence_curves(payload, output_dir / "convergence_curves.png")
    plot_error_distribution(payload, output_dir / "error_distribution.png")
    plot_cost_accuracy(payload, output_dir / "cost_accuracy_tradeoff.png")
    plot_solution_fit(payload, x, t, C_ref, params, problem, config, checkpoint_dir,
                      output_dir / "solution_fit_comparison.png")
    print(f"Saved 4 figures -> {output_dir}")


# ============================================================== main =====
def main() -> None:
    config = load_config()
    x, t, C_ref, params = generate_reference(config)
    payload = run_sweep(config, x, t, C_ref, params)
    write_comparison_table(payload["results"])
    make_figures(payload, x, t, C_ref, params, config)


if __name__ == "__main__":
    main()
