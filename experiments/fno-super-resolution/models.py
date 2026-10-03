"""
models.py -- A Fourier Neural Operator and a convolutional network with the
same number of parameters, both in plain PyTorch.

Both take the start u0 sampled on any grid of n points, as three channels
(u0, sin x, cos x), and return u at t = 1 on the same n points. The position
channels are functions of x, not of the grid index, so they mean the same
thing at every resolution.

FNO (Li et al., 2021)
---------------------
    lift:      pointwise linear, 3 -> width
    4 layers:  v <- GELU( K v + W v )         (no GELU after the last)
    project:   pointwise, width -> 128 -> 1

K is the spectral convolution: take the FFT of each channel, keep the
lowest `modes` wavenumbers, mix channels there with a learned complex matrix
per wavenumber, inverse FFT back onto the same grid. W is a pointwise
linear map. Nothing in the layer refers to the grid spacing - the weights
live on wavenumbers, which are the same on every grid - and that is where
the claim of resolution invariance comes from.

With the default FFT normalisation the forward transform scales with n and
the inverse divides by n, so for a band-limited input the layer's output
values do not depend on n. `test_models.py` checks exactly that.

CNN
---
Fully convolutional, circular padding, kernel 5, dilations growing to 16 so
that at 64 points every output sees the whole periodic domain - the same
global reach the FNO has at the training resolution. Its weights live on
grid offsets, so on a finer grid the same weights reach a shorter physical
distance. It is here as the control that shows resolution invariance is a
property worth testing, not one every network has.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def features(u0: torch.Tensor) -> torch.Tensor:
    """(batch, n) start -> (batch, 3, n): u0, sin x, cos x on the grid
    x_i = 2 pi i / n."""
    n = u0.shape[-1]
    x = torch.arange(n, dtype=u0.dtype, device=u0.device) * (2 * math.pi / n)
    pos = torch.stack([torch.sin(x), torch.cos(x)]).expand(u0.shape[0], 2, n)
    return torch.cat([u0.unsqueeze(1), pos], dim=1)


class SpectralConv1d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (width * width)
        self.weight = nn.Parameter(scale * torch.randn(width, width, modes, dtype=torch.cfloat))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        n = v.shape[-1]
        vh = torch.fft.rfft(v)                                  # (batch, width, n//2+1)
        m = min(self.modes, vh.shape[-1])
        out = torch.zeros_like(vh)
        out[..., :m] = torch.einsum("bix,iox->box", vh[..., :m], self.weight[..., :m])
        return torch.fft.irfft(out, n=n)


class FNO(nn.Module):
    def __init__(self, width: int = 32, modes: int = 16, layers: int = 4):
        super().__init__()
        self.lift = nn.Conv1d(3, width, 1)
        self.spectral = nn.ModuleList(SpectralConv1d(width, modes) for _ in range(layers))
        self.pointwise = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(layers))
        self.proj1 = nn.Conv1d(width, 128, 1)
        self.proj2 = nn.Conv1d(128, 1, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        v = self.lift(features(u0))
        last = len(self.spectral) - 1
        for i, (k, w) in enumerate(zip(self.spectral, self.pointwise)):
            v = k(v) + w(v)
            if i < last:
                v = F.gelu(v)
        return self.proj2(F.gelu(self.proj1(v))).squeeze(1)


class CNN(nn.Module):
    def __init__(self, channels: int = 63, kernel: int = 5,
                 dilations: tuple = (1, 2, 4, 8, 16, 4, 1)):
        super().__init__()
        def conv(cin, cout, d):
            return nn.Conv1d(cin, cout, kernel, dilation=d, padding=d * (kernel // 2),
                             padding_mode="circular")
        self.inp = conv(3, channels, 1)
        self.hidden = nn.ModuleList(conv(channels, channels, d) for d in dilations)
        self.out = nn.Conv1d(channels, 1, 1)
        self.receptive_field = 1 + (kernel - 1) * (1 + sum(dilations))

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        v = F.gelu(self.inp(features(u0)))
        for c in self.hidden:
            v = F.gelu(c(v))
        return self.out(v).squeeze(1)


def n_params(model: nn.Module) -> int:
    """Real parameters: a complex weight counts twice."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Per-sample ||pred - true|| / ||true|| on the grid both live on."""
    return torch.linalg.vector_norm(pred - true, dim=-1) / torch.linalg.vector_norm(true, dim=-1)


def build(name: str, cfg: dict) -> nn.Module:
    if name == "fno":
        return FNO(**cfg["fno"])
    if name == "cnn":
        c = dict(cfg["cnn"])
        c["dilations"] = tuple(c["dilations"])
        return CNN(**c)
    raise ValueError(name)
