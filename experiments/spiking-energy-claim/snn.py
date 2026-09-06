"""
snn.py -- A leaky integrate-and-fire network, a dense one, and an honest
operation counter for both.

The neuron
----------
Discrete-time LIF, the standard formulation:

    V[t] = beta * V[t-1] * (1 - S[t-1]) + I[t]
    S[t] = 1 if V[t] > threshold else 0

`beta` is the membrane leak per step and the `(1 - S[t-1])` factor is the
reset: a neuron that fired starts from zero. The membrane is the only memory
in the network - there are no recurrent weights - so anything the SNN does
with `T` timesteps it does through the neuron's own dynamics.

The gradient
------------
The forward spike is a true Heaviside step, which has zero derivative
everywhere it is defined and no derivative where it matters. Training uses a
surrogate on the backward pass only (Neftci, Mostafa & Zenke 2019): the
derivative of a fast sigmoid, peaked at the threshold. `test_snn.py` checks
that the forward pass is genuinely binary and that the backward pass is
genuinely non-zero, because getting one of those wrong produces a network
that trains beautifully and is not a spiking network.

The counting
------------
`OpCount` is the point of the experiment, so it is measured from the actual
spike tensors produced during evaluation rather than estimated from a
firing-rate assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# the spike function
# --------------------------------------------------------------------------

class SurrogateSpike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid derivative backward."""

    slope = 25.0

    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v > 0).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (v,) = ctx.saved_tensors
        sg = 1.0 / (SurrogateSpike.slope * v.abs() + 1.0) ** 2
        return grad_out * sg


spike_fn = SurrogateSpike.apply


# --------------------------------------------------------------------------
# operation counting
# --------------------------------------------------------------------------

@dataclass
class OpCount:
    """Operations per prediction, split by where they are spent.

    Kept separate because the whole finding is that the split decides the
    answer: `hidden` is the sparse, event-driven part everybody quotes, and
    `input_layer` is the part that is dense `T` times over and usually is
    not quoted.
    """
    input_layer: float
    hidden: float
    spike_rate: float = 0.0
    input_spike_rate: float = 0.0

    @property
    def total(self) -> float:
        return self.input_layer + self.hidden

    def as_dict(self, prefix: str = "") -> dict:
        return {f"{prefix}ops_input": self.input_layer,
                f"{prefix}ops_hidden": self.hidden,
                f"{prefix}ops_total": self.total,
                f"{prefix}spike_rate": self.spike_rate,
                f"{prefix}input_spike_rate": self.input_spike_rate}


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

class MLP(nn.Module):
    """The dense reference. One hidden layer, so that `n_hidden` means the
    same thing it means in the SNN."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.act = nn.ReLU()
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x))).squeeze(-1)

    def count_ops(self, x) -> OpCount:
        """Dense, and the same for every input: that is what dense means."""
        return OpCount(input_layer=float(self.n_in * self.n_hidden),
                       hidden=float(self.n_hidden * self.n_out))


class SNN(nn.Module):
    """LIF hidden layer, leaky non-spiking readout.

    The readout integrates without firing, which is the usual choice for
    regression: a spike count would quantise the output to `T + 1` levels and
    the comparison would be measuring that instead of the architecture.
    """

    def __init__(self, n_in: int, n_hidden: int, n_steps: int,
                 beta: float = 0.9, threshold: float = 1.0,
                 encoding: str = "direct", n_out: int = 1):
        super().__init__()
        if encoding not in ("direct", "rate"):
            raise ValueError(f"unknown encoding {encoding!r}")
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.n_steps, self.beta, self.threshold = n_steps, beta, threshold
        self.encoding = encoding

    def encode(self, x, generator=None):
        """(batch, n_in) -> (T, batch, n_in).

        `direct` repeats the analogue value: the first layer then does a dense
        matrix multiply on every one of the T steps.

        `rate` emits Bernoulli spikes whose probability is the min-max scaled
        feature, so the first layer sees a sparse binary vector. The scaling
        is applied by the caller; anything outside [0, 1] is clamped, which is
        what a real encoder would have to do with an out-of-range input.
        """
        T = self.n_steps
        if self.encoding == "direct":
            return x.unsqueeze(0).expand(T, *x.shape)
        p = x.clamp(0.0, 1.0).unsqueeze(0).expand(T, *x.shape)
        return torch.bernoulli(p, generator=generator)

    def forward(self, x, generator=None, collect: bool = False):
        inp = self.encode(x, generator)
        v = torch.zeros(x.shape[0], self.n_hidden, dtype=x.dtype,
                        device=x.device)
        s = torch.zeros_like(v)
        out = torch.zeros(x.shape[0], self.n_out, dtype=x.dtype,
                          device=x.device)
        n_hidden_spikes = 0.0
        for t in range(self.n_steps):
            cur = self.fc1(inp[t])
            v = self.beta * v * (1.0 - s) + cur
            s = spike_fn(v - self.threshold)
            out = self.beta * out + self.fc2(s)
            if collect:
                n_hidden_spikes += float(s.detach().sum())
        y = out.squeeze(-1)
        if not collect:
            return y
        return y, n_hidden_spikes, float(inp.detach().sum())

    @torch.no_grad()
    def count_ops(self, x, generator=None) -> OpCount:
        """Measured from the spikes this input actually produced.

        Not from a rate assumption: the whole question is how many operations
        really happen, and a firing rate quoted from another paper is exactly
        the thing being checked.
        """
        _, hidden_spikes, input_sum = self.forward(x, generator, collect=True)
        n = x.shape[0]
        T = self.n_steps

        if self.encoding == "direct":
            # dense matrix multiply, every timestep, regardless of activity
            input_ops = float(T * self.n_in * self.n_hidden)
            in_rate = 1.0
        else:
            # one accumulate per input spike per hidden neuron
            input_ops = float(input_sum / n * self.n_hidden)
            in_rate = float(input_sum / (n * T * self.n_in))

        hidden_ops = float(hidden_spikes / n * self.n_out)
        return OpCount(input_layer=input_ops, hidden=hidden_ops,
                       spike_rate=float(hidden_spikes / (n * T * self.n_hidden)),
                       input_spike_rate=in_rate)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train(model: nn.Module, Xtr, ytr, Xva, yva, *, epochs: int, batch: int,
          lr: float, seed: int, patience: int, verbose: bool = False) -> nn.Module:
    """Adam, MSE, early stopping on the validation split.

    The best validation state is restored at the end, so `epochs` is a budget
    rather than a tuned quantity - the SNN needs more of them than the MLP and
    fixing the count would confound depth of training with architecture.
    """
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    n = Xtr.shape[0]
    best, best_state, bad = float("inf"), None, 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va = float(lossf(model(Xva), yva))
        if va < best - 1e-6:
            best, bad = va, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 10 == 0:
            print(f"    epoch {ep:>3}  val mse {va:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))
