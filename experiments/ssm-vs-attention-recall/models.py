"""
models.py -- A Transformer and a selective state-space model, built from
the same parts, in plain PyTorch.

Both are two-layer language models: token embedding, two pre-norm residual
blocks of (mixer, MLP), a final norm and a linear head over the vocabulary.
Everything is identical except the mixer - the one part that moves
information between positions.

Attention
---------
Causal softmax attention, 4 heads of 16 dimensions, rotary position
embedding. To answer a query it looks back over every earlier token and
weighs them by similarity: the whole sequence is kept, so its memory - the
key-value cache - grows with the length.

Selective SSM (Mamba-2 style; Dao & Gu, 2024)
---------------------------------------------
Projects the input to x (8 heads of 16), a gate z, and per-position B_t,
C_t in R^N and step size dt_t; a short causal convolution runs over x, B,
C first. Then, per head, a recurrence over a state h of size 16 x N:

    h_t = exp(dt_t A) h_{t-1} + dt_t x_t B_t^T          (write)
    y_t = h_t C_t + D x_t                               (read)

Because B, C and dt are functions of the input, the model chooses at each
token what to write, what to read and how fast to forget: that is the
"selection". The state has 128 x N numbers per layer whatever the length of
the sequence - that is the claim to efficiency, and the constraint this
experiment measures.

The recurrence is computed here in its equivalent quadratic form (Mamba-2's
state-space duality): y = (L o C B^T) x with L the matrix of cumulative
decays. Same function, faster on a CPU for short sequences; `test_models.py`
checks the two agree. The model is still a fixed-state recurrence - the
form it is trained in does not change what it can represent.

LTI SSM (the ablation)
----------------------
The same block with selection switched off: B, C, dt and A are learned
constants, not functions of the input, with one decay per state dimension
(S4D-style) so the N state dimensions still carry N different timescales.
The recurrence becomes a fixed long convolution. Same state size.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# attention
# --------------------------------------------------------------------------

def rotary(x: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding on (batch, heads, length, dim)."""
    b, h, l, d = x.shape
    half = d // 2
    freq = 10000 ** (-torch.arange(half, dtype=x.dtype, device=x.device) / half)
    ang = torch.arange(l, dtype=x.dtype, device=x.device)[:, None] * freq[None, :]
    cos, sin = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        b, l, d = u.shape
        q, k, v = self.qkv(u).view(b, l, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(rotary(q), rotary(k), v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, l, d))

    def memory_floats(self, length: int, d_model: int) -> int:
        """Key-value cache per layer at inference, in numbers."""
        return 2 * length * d_model


# --------------------------------------------------------------------------
# state-space mixers
# --------------------------------------------------------------------------

def ssm_quadratic(x, dt, A, B, C):
    """Selective SSM, quadratic form.
    x (b, l, h, p), dt (b, l, h), A (h,) negative, B and C (b, l, n)."""
    l = x.shape[1]
    S = torch.cumsum(dt * A, dim=1)                                  # (b, l, h)
    diff = S[:, :, None, :] - S[:, None, :, :]                       # (b, t, s, h)
    causal = torch.ones(l, l, dtype=torch.bool, device=x.device).tril()
    decay = torch.exp(diff.masked_fill(~causal[None, :, :, None], float("-inf")))
    G = torch.einsum("btn,bsn->bts", C, B)
    M = decay * (G[..., None] * dt[:, None, :, :])                   # dt at the source
    return torch.einsum("btsh,bshp->bthp", M, x)


def ssm_chunked(x, dt, A, B, C, chunk: int = 16):
    """Selective SSM, chunked (Mamba-2's SSD algorithm): the quadratic form
    inside chunks of `chunk` positions, the recurrence between them through
    the explicit state. Same function as the other two forms; the cost grows
    with length x chunk instead of length^2."""
    b, l, h, p = x.shape
    n = B.shape[-1]
    pad = (-l) % chunk
    if pad:
        x, dt = F.pad(x, (0, 0, 0, 0, 0, pad)), F.pad(dt, (0, 0, 0, pad))
        B, C = F.pad(B, (0, 0, 0, pad)), F.pad(C, (0, 0, 0, pad))
    c = (l + pad) // chunk
    x, dt = x.view(b, c, chunk, h, p), dt.view(b, c, chunk, h)
    B, C = B.view(b, c, chunk, n), C.view(b, c, chunk, n)
    S = torch.cumsum(dt * A, dim=2)                                   # (b, c, q, h)
    # within each chunk
    diff = S[:, :, :, None, :] - S[:, :, None, :, :]                  # (b, c, t, s, h)
    causal = torch.ones(chunk, chunk, dtype=torch.bool, device=x.device).tril()
    decay = torch.exp(diff.masked_fill(~causal[None, None, :, :, None], float("-inf")))
    G = torch.einsum("bctn,bcsn->bcts", C, B)
    y = torch.einsum("bctsh,bcshp->bcthp", decay * (G[..., None] * dt[:, :, None]), x)
    # what each chunk adds to the state, decayed to the chunk's end
    to_end = torch.exp(S[:, :, -1:, :] - S) * dt                      # (b, c, q, h)
    added = torch.einsum("bcshp,bcsn->bchpn", x * to_end[..., None], B)
    # carry the state across chunk boundaries
    state = x.new_zeros(b, h, p, n)
    entering = []
    for k in range(c):
        entering.append(state)
        state = torch.exp(S[:, k, -1])[:, :, None, None] * state + added[:, k]
    entering = torch.stack(entering, dim=1)                           # (b, c, h, p, n)
    y = y + torch.einsum("bctn,bchpn->bcthp", C, entering) * torch.exp(S)[..., None]
    return y.reshape(b, c * chunk, h, p)[:, :l]


def ssm_recurrent(x, dt, A, B, C):
    """The same function as a recurrence over an explicit state of size
    (h, p, n). Slow; used to check the quadratic form and to measure the
    state."""
    b, l, h, p = x.shape
    n = B.shape[-1]
    state = x.new_zeros(b, h, p, n)
    ys = []
    for t in range(l):
        a = torch.exp(dt[:, t] * A)[:, :, None, None]                # (b, h, 1, 1)
        state = a * state + (dt[:, t, :, None, None] * x[:, t, :, :, None]
                             * B[:, t, None, None, :])
        ys.append((state * C[:, t, None, None, :]).sum(-1))
    return torch.stack(ys, dim=1)


def lti_kernel(dt, A, B, C, length):
    """Convolution kernel of the LTI SSM, per head: K_h(tau) =
    sum_n C_hn B_hn exp(tau dt_h A_hn) dt_h. Shapes: dt (h,), A, B, C (h, n)."""
    tau = torch.arange(length, dtype=dt.dtype, device=dt.device)
    expo = torch.exp(tau[None, :, None] * (dt[:, None] * A)[:, None, :])  # (h, l, n)
    return torch.einsum("hln,hn->hl", expo, C * B) * dt[:, None]


def lti_apply(x, K):
    """y_t = sum_{s <= t} K(t - s) x_s, per head. x (b, l, h, p)."""
    l = x.shape[1]
    idx = torch.arange(l, device=x.device)
    lag = idx[:, None] - idx[None, :]
    T = K[:, lag.clamp(min=0)] * (lag >= 0)                          # (h, t, s)
    return torch.einsum("hts,bshp->bthp", T, x)


def lti_recurrent(x, dt, A, B, C):
    b, l, h, p = x.shape
    n = A.shape[-1]
    a = torch.exp(dt[:, None] * A)[None, :, None, :]                 # (1, h, 1, n)
    state = x.new_zeros(b, h, p, n)
    ys = []
    for t in range(l):
        state = a * state + (dt[None, :, None, None] * x[:, t, :, :, None]
                             * B[None, :, None, :])
        ys.append((state * C[None, :, None, :]).sum(-1))
    return torch.stack(ys, dim=1)


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    return y + torch.log(-torch.expm1(-y))


class SSM(nn.Module):
    def __init__(self, d_model: int = 64, d_state: int = 16, expand: int = 2,
                 headdim: int = 16, conv: int = 4, selective: bool = True):
        super().__init__()
        self.d_inner = expand * d_model
        self.h, self.p, self.n = self.d_inner // headdim, headdim, d_state
        self.selective = selective
        h, n = self.h, self.n
        extra = 2 * n + h if selective else 0
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + extra, bias=False)
        conv_ch = self.d_inner + (2 * n if selective else 0)
        self.conv = nn.Conv1d(conv_ch, conv_ch, conv, groups=conv_ch, padding=conv - 1)
        dt0 = torch.exp(torch.empty(h).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter(inv_softplus(dt0))
        if selective:
            self.A_log = nn.Parameter(torch.log(torch.empty(h).uniform_(1, 16)))
        else:
            # S4D-real: one decay per state dimension, A_n = n + 1
            self.A_log = nn.Parameter(torch.log(torch.arange(1, n + 1, dtype=torch.float)
                                                ).repeat(h, 1))
            self.B = nn.Parameter(torch.randn(h, n) / math.sqrt(n))
            self.C = nn.Parameter(torch.randn(h, n) / math.sqrt(n))
        self.D = nn.Parameter(torch.ones(h))
        self.norm = nn.RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _project(self, u):
        b, l, _ = u.shape
        zxbcdt = self.in_proj(u)
        z, rest = zxbcdt[..., :self.d_inner], zxbcdt[..., self.d_inner:]
        if self.selective:
            xbc, dt = rest[..., :-self.h], rest[..., -self.h:]
        else:
            xbc, dt = rest, None
        xbc = F.silu(self.conv(xbc.transpose(1, 2))[..., :l].transpose(1, 2))
        x = xbc[..., :self.d_inner].reshape(b, l, self.h, self.p)
        return z, x, xbc, dt

    def core(self, u, recurrent: bool = False):
        """The mixer's state-space part. Returns (z, y) before gating."""
        z, x, xbc, dt = self._project(u)
        if self.selective:
            Bm = xbc[..., self.d_inner:self.d_inner + self.n]
            Cm = xbc[..., self.d_inner + self.n:]
            dt = F.softplus(dt + self.dt_bias)
            A = -torch.exp(self.A_log)
            f = ssm_recurrent if recurrent else ssm_chunked
            y = f(x, dt, A, Bm, Cm)
        else:
            dt = F.softplus(self.dt_bias)
            A = -torch.exp(self.A_log)
            if recurrent:
                y = lti_recurrent(x, dt, A, self.B, self.C)
            else:
                y = lti_apply(x, lti_kernel(dt, A, self.B, self.C, u.shape[1]))
        y = y + x * self.D[None, None, :, None]
        return z, y

    def forward(self, u: torch.Tensor, recurrent: bool = False) -> torch.Tensor:
        b, l, _ = u.shape
        z, y = self.core(u, recurrent)
        y = self.norm(y.reshape(b, l, self.d_inner) * F.silu(z))
        return self.out_proj(y)

    def memory_floats(self, length: int, d_model: int) -> int:
        """Recurrent state per layer at inference, in numbers: the SSM state
        plus the convolution's window. Independent of `length`."""
        conv_ch = self.conv.in_channels
        return self.d_inner * self.n + conv_ch * (self.conv.kernel_size[0] - 1)


# --------------------------------------------------------------------------
# the language model around either mixer
# --------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, mixer: nn.Module, d_model: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.mixer = mixer
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, u):
        u = u + self.mixer(self.n1(u))
        return u + self.mlp(self.n2(u))


class LM(nn.Module):
    def __init__(self, vocab: int, mixer: str, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab, d_model)

        def make():
            if mixer == "attention":
                return Attention(d_model, n_heads)
            return SSM(d_model, d_state, selective=(mixer == "ssm"))
        self.blocks = nn.ModuleList(Block(make(), d_model) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, tokens: torch.Tensor, start: int = 0) -> torch.Tensor:
        """Logits from position `start` on. Every position is still computed
        and still feeds the later ones; the head is only applied where an
        answer is scored."""
        u = self.emb(tokens)
        for blk in self.blocks:
            u = blk(u)
        return self.head(self.norm(u[:, start:]))

    def memory_floats(self, length: int) -> int:
        return sum(b.mixer.memory_floats(length, self.d_model) for b in self.blocks)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build(spec: dict, vocab: int, cfg: dict) -> LM:
    """spec: {"mixer": "attention" | "ssm" | "lti", "d_state": N}."""
    m = cfg["model"]
    return LM(vocab, spec["mixer"], d_model=m["d_model"], n_layers=m["n_layers"],
              n_heads=m["n_heads"], d_state=spec.get("d_state", 16))
