"""
models.py -- A network that predicts a vector field, and one that predicts a
scalar and derives the field from it.

The difference is one line. The MLP outputs (dq/dt, dp/dt). The HNN outputs
a single number H_theta and the field is

    dq/dt =  dH_theta/dp
    dp/dt = -dH_theta/dq

by autograd. Any field of that form conserves H_theta exactly and has zero
divergence, whatever the weights are. That is the whole architecture: the
guarantee comes from the wiring, not from the training.

The wiring is also the place to get wrong. Swapping the two partials, or
dropping the minus sign, gives a field that is still divergence-free and
still conserves something - just not the right something, and the pendulum
rotates the wrong way. `test_models.py` plugs the true H in as H_theta and
checks that the exact pendulum field comes out, sign and all.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def embed(x: torch.Tensor) -> torch.Tensor:
    """(q, p) -> (cos q, sin q, p). Fixed, with no parameters.

    The pendulum's angle lives on a circle, and a rotating pendulum's q grows
    without bound - the generator's rotating trajectories reach +/-234 rad.
    Fed raw q, a tanh network has to represent sin(q) over that whole range
    with 64 units, and cannot: the first version of this experiment failed
    its own control at 28% field error after 300 epochs. Both models get the
    same embedding, and the HNN still differentiates with respect to the
    original (q, p), so its field is still the symplectic gradient of a
    scalar - the chain rule goes through the embedding.
    """
    q, p = x[..., 0:1], x[..., 1:2]
    return torch.cat([torch.cos(q), torch.sin(q), p], dim=-1)


def body(n_hidden: int, n_out: int) -> nn.Sequential:
    """Two tanh layers. Tanh rather than ReLU because the HNN's field is a
    derivative of the output, and a piecewise-linear H_theta would give a
    piecewise-constant field."""
    return nn.Sequential(nn.Linear(3, n_hidden), nn.Tanh(),
                         nn.Linear(n_hidden, n_hidden), nn.Tanh(),
                         nn.Linear(n_hidden, n_out))


class MLP(nn.Module):
    """(q, p) -> (dq/dt, dp/dt), directly."""

    def __init__(self, n_hidden: int = 64):
        super().__init__()
        self.net = body(n_hidden, 2)

    def field(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(embed(x))

    forward = field


class HNN(nn.Module):
    """(q, p) -> H_theta, and the field is its symplectic gradient."""

    def __init__(self, n_hidden: int = 64):
        super().__init__()
        self.net = body(n_hidden, 1)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(embed(x)).squeeze(-1)

    def field(self, x: torch.Tensor) -> torch.Tensor:
        # create_graph so the field itself can be differentiated during
        # training - the loss is on the field, so gradients flow through this
        # gradient
        needs = x.requires_grad
        if not needs:
            x = x.requires_grad_(True)
        H = self.energy(x)
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        dH_dq, dH_dp = dH[:, 0], dH[:, 1]
        return torch.stack([dH_dp, -dH_dq], dim=-1)

    forward = field


def n_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------
# integration of a learned field
# --------------------------------------------------------------------------

@torch.no_grad()
def _no_grad_field(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The HNN needs autograd even at inference, so this cannot simply be
    wrapped in no_grad; it turns grad on locally and detaches the result."""
    with torch.enable_grad():
        return model.field(x.detach().requires_grad_(True)).detach()


def rollout(model: nn.Module, x0: torch.Tensor, n_steps: int, dt: float,
            save_every: int = 1) -> torch.Tensor:
    """Fixed-step RK4, batched over initial conditions. Same scheme and same
    step as the generator, so drift is a property of the field.

    Returns (n_saved, N, 2), including the initial state.
    """
    x = x0.clone()
    out = [x.clone()]
    for i in range(1, n_steps + 1):
        k1 = _no_grad_field(model, x)
        k2 = _no_grad_field(model, x + 0.5 * dt * k1)
        k3 = _no_grad_field(model, x + 0.5 * dt * k2)
        k4 = _no_grad_field(model, x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if i % save_every == 0:
            out.append(x.clone())
    return torch.stack(out)


def true_energy(x: torch.Tensor) -> torch.Tensor:
    """The pendulum's H, for scoring rollouts of *either* model."""
    return 0.5 * x[..., 1] ** 2 - torch.cos(x[..., 0])


def true_field(x: torch.Tensor, gamma: float = 0.0) -> torch.Tensor:
    q, p = x[..., 0], x[..., 1]
    return torch.stack([p, -torch.sin(q) - gamma * p], dim=-1)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train(model: nn.Module, X: torch.Tensor, Y: torch.Tensor, *, epochs: int,
          batch: int, lr: float, seed: int, verbose: bool = False) -> list[float]:
    """Adam on the field MSE. No early stopping and no validation split: the
    two models get exactly the same budget, and the comparison is between
    architectures rather than between stopping rules."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    history = []
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = ((model.field(X[idx]) - Y[idx]) ** 2).mean()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        history.append(total / n)
        if verbose and ep % 50 == 0:
            print(f"    epoch {ep:>4}  train mse {history[-1]:.3e}")
    return history
