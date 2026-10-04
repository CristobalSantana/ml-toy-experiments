"""
test_models.py -- Is the SSM the recurrence it claims to be, and can no
model see the answer before it is asked?

    python test_models.py

The first check matters most. The SSM is trained through a chunked
algorithm that never materialises the state; if that algorithm computed
something other than the fixed-state recurrence - a cross-chunk term with
the wrong decay, say - the model could quietly be more than a recurrence,
and its capacity would say nothing about fixed states.

Run before the experiment; `run_all.py` stops if any check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

import models as M
from run_experiment import GEN, load_cfg, load_tests, query_loss

sys.path.insert(0, str(GEN))
from generate import sample, test_stream  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


# 1 -------------------------------------------------------------------------
def test_chunked_equals_recurrence() -> None:
    torch.manual_seed(0)
    worst = 0.0
    for l in (7, 16, 45, 192):                      # inside, on, across chunk edges
        b, h, p, n = 2, 8, 16, 32
        x = torch.randn(b, l, h, p, dtype=torch.float64)
        dt = torch.rand(b, l, h, dtype=torch.float64) * 0.5
        A = -torch.rand(h, dtype=torch.float64) * 4
        B = torch.randn(b, l, n, dtype=torch.float64)
        C = torch.randn(b, l, n, dtype=torch.float64)
        ref = M.ssm_recurrent(x, dt, A, B, C)
        got = M.ssm_chunked(x, dt, A, B, C)
        worst = max(worst, float((got - ref).abs().max() / ref.abs().max()))
    check("chunked selective SSM equals the explicit fixed-state recurrence",
          worst < 1e-10, f"max relative difference {worst:.1e} over lengths 7-192")


# 2 -------------------------------------------------------------------------
def test_lti_equals_recurrence() -> None:
    torch.manual_seed(1)
    h, n, l = 8, 32, 50
    x = torch.randn(2, l, h, 16, dtype=torch.float64)
    dt = torch.rand(h, dtype=torch.float64) * 0.3
    A = -torch.arange(1, n + 1, dtype=torch.float64).repeat(h, 1)
    B = torch.randn(h, n, dtype=torch.float64)
    C = torch.randn(h, n, dtype=torch.float64)
    ref = M.lti_recurrent(x, dt, A, B, C)
    got = M.lti_apply(x, M.lti_kernel(dt, A, B, C, l))
    err = float((got - ref).abs().max() / ref.abs().max())
    check("LTI convolution equals its recurrence", err < 1e-10, f"max relative difference {err:.1e}")


# 3 -------------------------------------------------------------------------
def test_causal() -> None:
    cfg = load_cfg()
    for spec in ({"mixer": "attention"}, {"mixer": "ssm", "d_state": 8},
                 {"mixer": "lti", "d_state": 8}):
        torch.manual_seed(2)
        lm = M.build(spec, 512, cfg).eval()
        tok = torch.randint(0, 512, (2, 48))
        changed = tok.clone()
        changed[:, 30:] = torch.randint(0, 512, (2, 18))
        with torch.no_grad():
            d = float((lm(tok)[:, :30] - lm(changed)[:, :30]).abs().max())
        check(f"{spec['mixer']}: changing tokens 30+ leaves outputs 0-29 unchanged",
              d < 1e-5, f"max change {d:.1e}")


# 4 -------------------------------------------------------------------------
def test_state_is_fixed() -> None:
    cfg = load_cfg()
    for n in cfg["ssm"]["d_states"]:
        lm = M.build({"mixer": "ssm", "d_state": n}, 512, cfg)
        mem = {L: lm.memory_floats(L) for L in (12, 192, 10_000)}
        mix = lm.blocks[0].mixer
        want = 2 * (128 * n + (128 + 2 * n) * 3)
        ok = len(set(mem.values())) == 1 and mem[12] == want and mix.d_inner * mix.n == 128 * n
        check(f"SSM N={n}: state independent of length, 128 x {n} per layer",
              ok, f"{mem[12]:,} numbers over 2 layers at any length")
    att = M.build({"mixer": "attention"}, 512, cfg)
    check("attention: key-value cache grows with length",
          att.memory_floats(192) == 4 * att.memory_floats(48),
          f"{att.memory_floats(48):,} at 48 tokens, {att.memory_floats(192):,} at 192")


# 5 -------------------------------------------------------------------------
def test_data_and_loss() -> None:
    cfg = load_cfg()
    tests = load_tests()
    check("test sets for every pair count",
          sorted(tests) == cfg["data"]["pair_counts"], f"{sorted(tests)}")
    tok, _ = sample(np.random.default_rng([cfg["seed"], 1, 0]), 1000, 4)
    test_tok = tests[4][0].numpy()
    shared = len({r.tobytes() for r in tok} & {r.tobytes() for r in test_tok})
    check("the training stream is not the test stream", shared == 0,
          f"{shared} of the first 1,000 training sequences appear in the test set")
    # a model that is right at every query and wrong everywhere else
    tok, tgt = tests[8]
    logits = torch.full((*tok.shape, 512), -20.0)
    logits.scatter_(-1, tok.unsqueeze(-1), 20.0)            # wrong: echoes the input
    q = tgt != -1
    logits[q] = -20.0
    logits[q] = logits[q].scatter(-1, tgt[q].unsqueeze(-1), 20.0)
    loss = float(query_loss(logits, tgt))
    check("the loss sees the query positions and nothing else", loss < 1e-6,
          f"loss {loss:.1e} for a model right only at the queries")


def main() -> None:
    torch.set_num_threads(2)
    for t in (test_chunked_equals_recurrence, test_lti_equals_recurrence, test_causal,
              test_state_is_fixed, test_data_and_loss):
        t()
    if FAILURES:
        sys.exit(f"\n{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
    print("\nall implementation checks passed")


if __name__ == "__main__":
    main()
