"""
test_models.py -- Are the operator, the control and the data what the
pre-registration says they are?

    python test_models.py

The check that matters most is the first. An FNO whose spectral layer
depended on the grid - an FFT normalised the wrong way, a mode count tied
to the number of points - trains exactly as well at 64 points and fails at
256 for a reason that has nothing to do with the claim being tested.

Run before the experiment; `run_all.py` stops if any check fails.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

import models as M
from run_experiment import interpolate_linear, load_data

HERE = Path(__file__).resolve().parent
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def band_limited(n: int, k_max: int, batch: int = 4, seed: int = 0) -> torch.Tensor:
    """Random fields with modes 1..k_max, sampled exactly on an n-point grid."""
    g = torch.Generator().manual_seed(seed)
    x = torch.arange(n, dtype=torch.float64) * 2 * math.pi / n
    k = torch.arange(1, k_max + 1, dtype=torch.float64)
    a = torch.randn(batch, k_max, 1, generator=g, dtype=torch.float64)
    b = torch.randn(batch, k_max, 1, generator=g, dtype=torch.float64)
    return (a * torch.cos(k[:, None] * x) + b * torch.sin(k[:, None] * x)).sum(1)


# 1 -------------------------------------------------------------------------
def test_spectral_layer_is_grid_independent() -> None:
    torch.manual_seed(0)
    layer = M.SpectralConv1d(width=3, modes=16).to(torch.complex128)
    layer.weight.data = layer.weight.data.to(torch.complex128)
    v64 = torch.stack([band_limited(64, 16, seed=s) for s in range(3)], 1)
    v256 = torch.stack([band_limited(256, 16, seed=s) for s in range(3)], 1)
    out64, out256 = layer(v64), layer(v256)
    err = float((out256[..., ::4] - out64).abs().max() / out64.abs().max())
    check("spectral layer gives the same values on 64 and 256 points",
          err < 1e-10, f"max relative difference at shared points {err:.1e}")


# 2 -------------------------------------------------------------------------
def test_spectral_layer_is_translation_equivariant() -> None:
    torch.manual_seed(1)
    layer = M.SpectralConv1d(width=4, modes=16)
    v = torch.randn(2, 4, 64)
    shifted = layer(torch.roll(v, 5, dims=-1))
    err = float((shifted - torch.roll(layer(v), 5, dims=-1)).abs().max())
    check("spectral layer commutes with a shift of the grid", err < 1e-5,
          f"max difference {err:.1e}")


# 3 -------------------------------------------------------------------------
def test_parameter_counts() -> None:
    cfg = {"fno": {"width": 32, "modes": 16, "layers": 4},
           "cnn": {"channels": 63, "kernel": 5, "dilations": [1, 2, 4, 8, 16, 4, 1]}}
    nf, nc = M.n_params(M.build("fno", cfg)), M.n_params(M.build("cnn", cfg))
    check("parameter counts as pre-registered", nf == 139_777 and nc == 140_428,
          f"FNO {nf:,}, CNN {nc:,} ({100 * (nc / nf - 1):+.1f}%)")


# 4 -------------------------------------------------------------------------
def test_cnn_sees_whole_domain_at_64() -> None:
    torch.manual_seed(2)
    cnn = M.CNN()
    u = torch.randn(1, 64, requires_grad=True)
    cnn(u)[0, 0].backward()
    reached = int((u.grad[0].abs() > 0).sum())
    check("CNN output depends on every input point at 64", reached == 64,
          f"{reached}/64 points reach output 0 (receptive field {cnn.receptive_field})")
    u = torch.randn(1, 256, requires_grad=True)
    cnn(u)[0, 0].backward()
    reached = int((u.grad[0].abs() > 0).sum())
    check("... and the same weights reach only part of it at 256", reached < 256,
          f"{reached}/256 points - the same physical reach shrinks by 4")


# 5 -------------------------------------------------------------------------
def test_data() -> None:
    d = load_data()
    u0 = d["u0"].astype(np.float64)
    e = np.abs(np.fft.rfft(u0, axis=1)) ** 2
    above = float((e[:, 17:].sum(1) / e.sum(1)).max())
    check("starts are band-limited to mode 16, so 64 points hold them",
          above < 1e-12, f"largest energy share above mode 16: {above:.1e}")
    sub = d["x"][::8]
    exact = np.arange(64) * 2 * np.pi / 64
    check("the 64-point grid is an exact subsample of the stored one",
          np.abs(sub - exact).max() < 1e-6, f"max |x - 2 pi i / 64| {np.abs(sub - exact).max():.1e}")
    check("1,200 samples at every viscosity",
          all(d[f"uT_nu{nu:g}"].shape == (1200, 512) for nu in d["viscosities"]),
          ", ".join(f"nu {nu:g}: {d[f'uT_nu{nu:g}'].shape}" for nu in d["viscosities"]))


# 6 -------------------------------------------------------------------------
def test_interpolation() -> None:
    x64 = np.arange(64) * 2 * np.pi / 64
    x256 = np.arange(256) * 2 * np.pi / 256
    f64 = torch.tensor(np.sin(3 * x64))[None]
    got = interpolate_linear(f64, 256).numpy()[0]
    keeps = float(np.abs(got[::4] - f64.numpy()[0]).max())
    # linear interpolation error is at most h^2/8 max|f''| = (2 pi/64)^2/8 * 9
    bound = (2 * np.pi / 64) ** 2 / 8 * 9
    err = float(np.abs(got - np.sin(3 * x256)).max())
    check("periodic linear interpolation keeps the coarse values",
          keeps < 1e-12, f"max change at coarse points {keeps:.1e}")
    check("... and is within its error bound between them, including the wrap",
          err <= bound * 1.0001, f"max error {err:.2e}, bound {bound:.2e}")


def main() -> None:
    torch.set_num_threads(2)
    for t in (test_spectral_layer_is_grid_independent,
              test_spectral_layer_is_translation_equivariant,
              test_parameter_counts, test_cnn_sees_whole_domain_at_64,
              test_data, test_interpolation):
        t()
    if FAILURES:
        sys.exit(f"\n{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
    print("\nall implementation checks passed")


if __name__ == "__main__":
    main()
