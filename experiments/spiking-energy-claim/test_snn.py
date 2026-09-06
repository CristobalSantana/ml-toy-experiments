"""
test_snn.py -- Is it actually a spiking network, and is the counter honest?

    python test_snn.py

Six checks. The two that matter most are the first and the last: a network
whose "spikes" are not binary, or whose operation counter is estimated from
an assumed firing rate, would produce every number in this experiment and
none of them would mean anything.

Run before the experiment; `run_all.py` stops if any check fails.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from snn import SNN, MLP, spike_fn

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def tiny(n_in=1, n_hidden=1, T=50, beta=0.9, thr=1.0, encoding="direct",
         w=1.0, b=0.0) -> SNN:
    m = SNN(n_in, n_hidden, T, beta=beta, threshold=thr, encoding=encoding)
    with torch.no_grad():
        m.fc1.weight.fill_(w)
        m.fc1.bias.fill_(b)
        m.fc2.weight.fill_(1.0)
        m.fc2.bias.fill_(0.0)
    return m.eval()


# 1 -------------------------------------------------------------------------
def test_matches_the_stated_recurrence() -> None:
    """The module must reproduce, exactly, the recurrence in its own docstring.

    Written out independently in numpy. An off-by-one in the reset - using
    this step's spike instead of the previous one - changes the dynamics and
    is invisible in a loss curve.
    """
    T, beta, thr, I = 50, 0.9, 1.0, 0.15
    m = tiny(T=T, beta=beta, thr=thr)

    v = s = 0.0
    ref = []
    for _ in range(T):
        v = beta * v * (1.0 - s) + I
        s = 1.0 if v > thr else 0.0
        ref.append(s)
    want = sum(ref)

    x = torch.full((1, 1), I)
    _, got, _ = m(x, collect=True)
    check("matches the stated LIF recurrence", got == want,
          f"{int(got)} spikes in {T} steps, independent reference says {int(want)}")


# 2 -------------------------------------------------------------------------
def test_subthreshold_never_fires() -> None:
    """Constant input whose steady state I/(1-beta) is below threshold must
    produce no spikes at all - not a few, none."""
    beta, thr = 0.9, 1.0
    I = 0.5 * thr * (1.0 - beta)          # steady state = half the threshold
    m = tiny(T=200, beta=beta, thr=thr)
    _, spikes, _ = m(torch.full((1, 1), I), collect=True)
    check("sub-threshold input never fires", spikes == 0.0,
          f"steady state {I/(1-beta):.2f} against threshold {thr}, "
          f"{int(spikes)} spikes in 200 steps")


# 3 -------------------------------------------------------------------------
def test_forward_is_binary_backward_is_not() -> None:
    """The forward pass must be a true step function and the backward pass
    must not be, which is the whole trick. If the forward pass leaked the
    surrogate, this would be a slightly odd dense network."""
    v = torch.linspace(-2, 2, 101, requires_grad=True)
    s = spike_fn(v)
    binary = bool(((s == 0) | (s == 1)).all())
    s.sum().backward()
    grad = v.grad
    flows = bool((grad.abs() > 0).any()) and bool(torch.isfinite(grad).all())
    peak = float(v.detach()[grad.argmax()])
    check("forward binary, backward differentiable", binary and flows,
          f"outputs in {{0,1}}: {binary}; gradient non-zero: {flows}; "
          f"steepest at v={peak:+.2f} (threshold is 0)")


# 4 -------------------------------------------------------------------------
def test_reset_actually_resets() -> None:
    """After a spike the membrane must drop. Without the reset term the
    neuron saturates and fires every step, which would look like a very
    energy-hungry SNN rather than a bug."""
    T, beta, thr, I = 60, 0.9, 1.0, 0.6
    m = tiny(T=T, beta=beta, thr=thr)
    _, spikes, _ = m(torch.full((1, 1), I), collect=True)
    # without reset, steady state is I/(1-beta) = 6.0, far above threshold,
    # so every step after the first few would fire
    check("the membrane resets after a spike", 0 < spikes < T * 0.9,
          f"{int(spikes)}/{T} steps fired; without a reset this would be "
          f"{T - 2} or more")


# 5 -------------------------------------------------------------------------
def test_rate_coding_is_sparse_and_binary() -> None:
    """Bernoulli-encoded input must be binary and must have the requested
    mean, or the input-layer operation count is measuring nothing."""
    m = tiny(n_in=8, n_hidden=4, T=200, encoding="rate")
    g = torch.Generator().manual_seed(0)
    x = torch.full((64, 8), 0.25)
    enc = m.encode(x, g)
    binary = bool(((enc == 0) | (enc == 1)).all())
    rate = float(enc.mean())
    check("rate coding is binary with the requested rate",
          binary and abs(rate - 0.25) < 0.01,
          f"binary: {binary}; mean spike rate {rate:.4f}, asked for 0.2500")


# 6 -------------------------------------------------------------------------
def test_counter_is_exact_at_both_extremes() -> None:
    """The counter is measured, so drive it to two cases whose answer is
    known exactly: every neuron firing every step, and none ever firing.

    An estimator based on an assumed firing rate would pass neither.
    """
    T, H, n_in = 20, 16, 4
    always = tiny(n_in=n_in, n_hidden=H, T=T, b=10.0)      # huge bias: fires always
    never = tiny(n_in=n_in, n_hidden=H, T=T, w=0.0, b=-10.0)  # can never reach thr
    x = torch.zeros(32, n_in)

    a, n = always.count_ops(x), never.count_ops(x)
    want_input = float(T * n_in * H)
    ok = (a.hidden == float(T * H) and a.spike_rate == 1.0
          and n.hidden == 0.0 and n.spike_rate == 0.0
          and a.input_layer == want_input == n.input_layer)
    check("operation counter exact at both extremes", ok,
          f"always: {a.hidden:.0f} hidden ops (want {T*H}), rate {a.spike_rate:.2f}; "
          f"never: {n.hidden:.0f} (want 0), rate {n.spike_rate:.2f}; "
          f"input {a.input_layer:.0f} (want {want_input:.0f})")

    # and the dense reference, whose count does not depend on the input at all
    mlp = MLP(n_in, H)
    o1, o2 = mlp.count_ops(x), mlp.count_ops(torch.randn(32, n_in) * 100)
    check("dense counter is input-independent", o1.total == o2.total,
          f"{o1.total:.0f} operations either way "
          f"({n_in}*{H} + {H}*1 = {n_in*H + H})")


if __name__ == "__main__":
    print("implementation checks")
    torch.manual_seed(0)
    test_matches_the_stated_recurrence()
    test_subthreshold_never_fires()
    test_forward_is_binary_backward_is_not()
    test_reset_actually_resets()
    test_rate_coding_is_sparse_and_binary()
    test_counter_is_exact_at_both_extremes()
    print()
    if FAILURES:
        sys.exit(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
    print("all checks passed")
