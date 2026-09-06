"""
data.py -- The field, and the number the method has to reproduce.

The generator in `generators/diffusion_1d/` solves

    dc/dt = D d2c/dx2

on a physical grid where D = 4e-14 m^2/s and L = 6e-6 m. Those magnitudes
are hopeless for a least-squares problem: the `u` column would be O(1e4) and
the `u_xxx` column O(1e26), and a threshold applied to that would be
selecting on units. Rescaling to x/L and t/T fixes it and, because the
equation is linear, changes the answer in a way that is known exactly:

    du/dt~ = (D T / L^2) d2u/dx~2,      D T / L^2 = 4/3

Rescaling `u` itself changes nothing at all, which `test_sindy.py` checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GEN = HERE.parent.parent / "generators" / "diffusion_1d" / "outputs"


@dataclass(frozen=True)
class Field:
    U: np.ndarray        # (nt+1, nx), scaled to span exactly [0, 1]
    dx: float            # in units of L
    dt: float            # in units of t_end
    true_coef: float     # D * t_end / L^2
    params: dict

    @property
    def shape(self) -> tuple[int, int]:
        return self.U.shape


def load_field() -> Field:
    npz = GEN / "diffusion_1d_solution.npz"
    if not npz.exists():
        raise SystemExit(
            f"missing {npz}\nrun:  python ../../generators/diffusion_1d/generate.py")
    z = np.load(npz)
    C, x, t = z["C"], z["x"], z["t"]
    p = json.loads((GEN / "diffusion_1d_params.json").read_text())

    # span exactly 1, so a noise sigma quoted as a fraction of the range is
    # numerically equal to the standard deviation added to U
    U = (C - C.min()) / (C.max() - C.min())

    L, T = p["L"], p["t_end"]
    return Field(U=U,
                 dx=float((x[1] - x[0]) / L),
                 dt=float((t[1] - t[0]) / T),
                 true_coef=float(p["D"] * T / L**2),
                 params=p)


def add_noise(U: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Gaussian noise on the sampled field, before any derivative is taken.

    U spans [0, 1] by construction, so `sigma` is both the standard deviation
    and the fraction of the concentration range it represents.
    """
    if sigma == 0.0:
        return U.copy()
    return U + np.random.default_rng(seed).normal(0.0, sigma, size=U.shape)
