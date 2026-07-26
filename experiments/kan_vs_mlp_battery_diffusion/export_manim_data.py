"""
export_manim_data.py -- Precompute the dense, plain-numpy data the Manim
animations need, so the Manim script itself never has to import torch
(it runs in a different virtual environment that only has numpy/manim).

Optional / supplementary: not part of run.py's core pipeline. Requires
run.py to have already been run (reads outputs/phase3_results.json and
checkpoints/*.pt).

Produces outputs/manim_data/solution_fit_dense.npz with:
  x_um          (nx,)               position across the electrode [um]
  t_min         (n_frames,)         charging time [min]
  charge_ref    (n_frames, nx)      reference "charge level" in [0,100]
  charge_mlp    (n_frames, nx)      best-of-seeds MLP prediction, same scale
  charge_kan    (n_frames, nx)      best-of-seeds KAN prediction, same scale
  best_mlp_seed, best_kan_seed      scalars, for labeling

"Charge level" is a 0-100 affine rescaling of the physical concentration
c(x,t) using the reference solution's own global min/max -- a shape-
preserving transform chosen so the animation can show relative fit
quality without putting raw mol/m^3 numbers in front of a non-technical
audience.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from generators.diffusion_1d.generate import DiffusionParams  # noqa: E402
from models.kan.kan import KAN  # noqa: E402
from models.mlp.mlp import MLP  # noqa: E402

OUTPUT_DIR = EXPERIMENT_ROOT / "outputs"
CHECKPOINT_DIR = EXPERIMENT_ROOT / "checkpoints"
MANIM_DATA_DIR = OUTPUT_DIR / "manim_data"
REFERENCE_NPZ = REPO_ROOT / "generators" / "diffusion_1d" / "outputs" / "diffusion_1d_solution.npz"

N_FRAMES = 90


def load_reference():
    data = np.load(REFERENCE_NPZ)
    params = DiffusionParams(
        D=float(data["D"]), L=float(data["L"]), c0=float(data["c0"]),
        surface_flux=float(data["surface_flux"]), t_end=float(data["t_end"]),
        nx=int(data["nx"]), nt=int(data["nt"]),
    )
    return data["x"], data["t"], data["C"], params


def load_model(config: dict, arch: str, seed: int):
    mlp_cfg, kan_cfg = config["architectures"]["mlp"], config["architectures"]["kan"]
    device = config["training"]["device"]
    if arch == "mlp":
        model = MLP(mlp_cfg["layers"])
    else:
        model = KAN(kan_cfg["layers"], grid_size=kan_cfg["grid_size"], spline_order=kan_cfg["spline_order"])
    model.load_state_dict(torch.load(CHECKPOINT_DIR / f"{arch}_seed{seed}.pt", map_location=device))
    model.eval()
    return model


def main() -> None:
    config = yaml.safe_load((EXPERIMENT_ROOT / "config.yaml").read_text())
    payload = json.loads((OUTPUT_DIR / "phase3_results.json").read_text())
    best = {arch: min(payload["results"][arch], key=lambda r: r["final_rel_l2"])["seed"]
            for arch in ["mlp", "kan"]}

    x, t, C_ref, params = load_reference()
    c_scale = payload["c_scale"]

    models = {arch: load_model(config, arch, seed) for arch, seed in best.items()}

    frame_idx = np.linspace(0, len(t) - 1, N_FRAMES).round().astype(int)
    t_frames = t[frame_idx]
    C_ref_frames = C_ref[frame_idx]  # (N_FRAMES, nx)

    x_hat = torch.tensor(x / params.L, dtype=torch.float32).unsqueeze(1)
    C_pred = {"mlp": np.empty_like(C_ref_frames), "kan": np.empty_like(C_ref_frames)}
    for i, t_phys in enumerate(t_frames):
        t_hat_col = torch.full_like(x_hat, t_phys / params.t_end)
        xt = torch.cat([x_hat, t_hat_col], dim=1)
        with torch.no_grad():
            for arch in ["mlp", "kan"]:
                c_hat_pred = models[arch](xt).numpy().ravel()
                C_pred[arch][i] = params.c0 + c_scale * c_hat_pred

    # Shape-preserving 0-100 rescaling, using the REFERENCE's own global
    # range so relative motion across frames stays physically meaningful.
    c_min, c_max = C_ref.min(), C_ref.max()

    def to_charge_pct(C):
        return 100.0 * (C - c_min) / (c_max - c_min)

    MANIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MANIM_DATA_DIR / "solution_fit_dense.npz",
        x_um=x * 1e6,
        t_min=t_frames / 60.0,
        charge_ref=to_charge_pct(C_ref_frames),
        charge_mlp=to_charge_pct(C_pred["mlp"]),
        charge_kan=to_charge_pct(C_pred["kan"]),
        best_mlp_seed=best["mlp"],
        best_kan_seed=best["kan"],
    )
    print(f"Saved -> {MANIM_DATA_DIR / 'solution_fit_dense.npz'}")
    print(f"best_mlp_seed={best['mlp']}  best_kan_seed={best['kan']}")


if __name__ == "__main__":
    main()
