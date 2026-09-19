"""
test_models.py -- Is the Hamiltonian wiring right, and does the integrator
reproduce the generator?

    python test_models.py

Five checks. The first two are the ones that matter: an HNN with the two
partial derivatives swapped, or the minus sign dropped, is still
divergence-free and still conserves *something*, and it trains just as well
on a loss over the field - it simply describes a pendulum that swings the
wrong way. Nothing in a loss curve would show it.

Run before the experiment; `run_all.py` stops if any check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import models as M

HERE = Path(__file__).resolve().parent
GEN = HERE.parent.parent / "generators" / "hamiltonian_pendulum" / "outputs"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


class TrueH(nn.Module):
    """The pendulum's own energy, in the HNN's clothing. If the symplectic
    gradient is wired correctly, plugging this in must give the exact field.

    It receives what the HNN's body receives - the embedding (cos q, sin q, p)
    - so H = p^2/2 - cos q reads as z[:, 2]^2/2 - z[:, 0]. The gradient is
    still taken with respect to (q, p), through the embedding, which is
    exactly the path the trained model uses.
    """

    def forward(self, z):
        return (0.5 * z[:, 2] ** 2 - z[:, 0]).unsqueeze(-1)


# 1 -------------------------------------------------------------------------
def test_true_h_gives_true_field() -> None:
    h = M.HNN()
    h.net = TrueH()
    x = torch.tensor([[0.3, 0.7], [-1.2, 0.0], [2.5, -1.4], [0.0, 2.0]])
    got = h.field(x.clone().requires_grad_(True)).detach()
    want = M.true_field(x)
    err = float((got - want).abs().max())
    check("true H in, true field out (signs and order)", err < 1e-6,
          f"max |error| {err:.1e} over 4 test states")


# 2 -------------------------------------------------------------------------
def test_hnn_field_is_divergence_free() -> None:
    """d(dq/dt)/dq + d(dp/dt)/dp must vanish for ANY weights, because it is
    d2H/dqdp - d2H/dpdq. The MLP has no such property, and the check confirms
    that too, or it would be testing nothing."""
    torch.manual_seed(3)
    x = torch.randn(256, 2, dtype=torch.float64)

    def divergence(model):
        xx = x.clone().requires_grad_(True)
        f = model.field(xx)
        d0 = torch.autograd.grad(f[:, 0].sum(), xx, create_graph=True)[0][:, 0]
        d1 = torch.autograd.grad(f[:, 1].sum(), xx, create_graph=True)[0][:, 1]
        return float((d0 + d1).detach().abs().max())

    hnn, mlp = M.HNN().double(), M.MLP().double()
    dh, dm = divergence(hnn), divergence(mlp)
    check("HNN field is divergence-free by construction, MLP is not",
          dh < 1e-10 and dm > 1e-3,
          f"HNN max |div| {dh:.1e} with random weights; MLP {dm:.2e}")


# 3 -------------------------------------------------------------------------
def test_rollout_reproduces_generator() -> None:
    """Integrate the TRUE field with the experiment's RK4 and compare with the
    generator's stored trajectories. If these disagree, every rollout error
    in the experiment would include an integrator mismatch."""
    z = np.load(GEN / "pendulum_ideal.npz")
    dt, k = float(z["dt"]), int(z["save_every"])
    q, p = z["q"][:, :8], z["p"][:, :8]                # 8 trajectories
    x0 = torch.tensor(np.stack([q[0], p[0]], -1), dtype=torch.float64)

    class Exact(nn.Module):
        def field(self, x):
            return M.true_field(x)

    n_saved = q.shape[0] - 1
    traj = M.rollout(Exact(), x0, n_steps=n_saved * k, dt=dt, save_every=k)
    ref = torch.tensor(np.stack([q, p], -1), dtype=torch.float64)
    err = float((traj - ref).abs().max())
    # the stored reference is float32, so agreement is bounded by that
    check("RK4 rollout of the true field reproduces the generator",
          err < 1e-4, f"max |state error| {err:.1e} over {n_saved} stored "
          f"states x 8 trajectories (reference stored in float32)")


# 4 -------------------------------------------------------------------------
def test_energy_drift_floor() -> None:
    """The integrator's own drift on the true field, so the HNN's drift has a
    floor to be read against rather than being compared with zero."""
    class Exact(nn.Module):
        def field(self, x):
            return M.true_field(x)
    x0 = torch.tensor([[0.5, 0.0], [2.0, 0.5], [1.0, 1.8]], dtype=torch.float64)
    traj = M.rollout(Exact(), x0, n_steps=10000, dt=0.01, save_every=100)
    H = M.true_energy(traj)
    drift = float((H - H[0]).abs().max())
    check("RK4 energy drift on the true field is tiny", drift < 1e-7,
          f"max |H(t) - H(0)| = {drift:.1e} over 100 time units")


# 5 -------------------------------------------------------------------------
def test_parameter_counts() -> None:
    """Same body, so the counts differ only by the output layer."""
    nm, nh = M.n_parameters(M.MLP()), M.n_parameters(M.HNN())
    check("MLP and HNN differ only by the output layer",
          nm - nh == 65 and nm == 4546 and nh == 4481,
          f"MLP {nm:,}  HNN {nh:,}  difference {nm - nh} "
          f"(= one 64->1 head fewer: 64 weights + 1 bias); "
          f"CRITERIA froze 4,546 and 4,481")


if __name__ == "__main__":
    print("implementation checks")
    if not (GEN / "pendulum_ideal.npz").exists():
        sys.exit(f"missing generator output\nrun: python "
                 f"../../generators/hamiltonian_pendulum/generate.py")
    test_true_h_gives_true_field()
    test_hnn_field_is_divergence_free()
    test_rollout_reproduces_generator()
    test_energy_drift_floor()
    test_parameter_counts()
    print()
    if FAILURES:
        sys.exit(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
    print("all checks passed")
