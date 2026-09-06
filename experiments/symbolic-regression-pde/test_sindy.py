"""
test_sindy.py -- Does the implementation work, on problems whose answer is
already known?

    python test_sindy.py

Six checks. Each one builds a case where the correct answer is known by
construction, so a failure points at the code rather than at the physics.
Run before the experiment; `run_all.py` stops if any check fails.
"""

from __future__ import annotations

import sys

import numpy as np

import sindy as S
from data import load_field, add_noise

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def travelling_wave(c: float = 2.0, nx: int = 201, nt: int = 201):
    """u(x,t) = sin(2pi s) + 0.5 sin(6pi s) with s = x + c t.

    Then u_t = c u_x exactly. Two wavenumbers rather than one on purpose: for
    a single sinusoid u_xx is proportional to u, the library is degenerate,
    and the test would be measuring nothing.
    """
    x = np.linspace(0.0, 1.0, nx)
    t = np.linspace(0.0, 1.0, nt)
    s = x[None, :] + c * t[:, None]
    U = np.sin(2 * np.pi * s) + 0.5 * np.sin(6 * np.pi * s)
    return U, float(x[1] - x[0]), float(t[1] - t[0])


# 1 -------------------------------------------------------------------------
def test_recovers_known_advection() -> None:
    """A field built from u_t = 2 u_x must come back as u_t = 2 u_x."""
    U, dx, dt = travelling_wave(c=2.0)
    smp = S.assemble(U, dx, dt, n_burn=3, n_edge=3)
    xi = S.stlsq(smp.Theta, smp.u_t, threshold=0.01)
    terms = S.active_terms(xi, smp.names)
    coef = xi[smp.names.index("u_x")]
    ok = terms == ["u_x"] and abs(coef - 2.0) / 2.0 < 0.02
    check("known advection recovered", ok,
          f"{S.describe(xi, smp.names)}  (truth: u_t = +2.0000 u_x)")


# 2 -------------------------------------------------------------------------
def test_derivative_is_second_order() -> None:
    """Halving dx must cut the second-derivative error by about four.

    If it does not, the stencils are wrong, and every noise threshold this
    experiment reports would be a property of the bug.
    """
    def err(nx: int) -> float:
        x = np.linspace(0.0, 1.0, nx)
        U = np.sin(2 * np.pi * x)[None, :].repeat(3, axis=0)
        got = S.d2_dx2(U, float(x[1] - x[0]))[1, 1:-1]
        want = -(2 * np.pi) ** 2 * np.sin(2 * np.pi * x)[1:-1]
        return float(np.abs(got - want).max())

    e1, e2 = err(101), err(201)
    ratio = e1 / e2
    check("central differences are 2nd order", 3.5 < ratio < 4.5,
          f"error {e1:.3e} -> {e2:.3e} on halving dx, ratio {ratio:.2f} (want ~4)")


# 3 -------------------------------------------------------------------------
def test_threshold_can_empty_the_model() -> None:
    """A threshold above every coefficient returns nothing, not the least bad
    survivor. A method that always returns a term cannot report a failure."""
    U, dx, dt = travelling_wave()
    smp = S.assemble(U, dx, dt, n_burn=3, n_edge=3)
    xi = S.stlsq(smp.Theta, smp.u_t, threshold=10.0)
    check("threshold above everything empties the model",
          S.active_terms(xi, smp.names) == [],
          f"{S.describe(xi, smp.names)}")


# 4 -------------------------------------------------------------------------
def test_linear_rescaling_of_u_is_invisible() -> None:
    """u -> a u + b must not move the u_xx coefficient.

    This is what licenses scaling the concentration to [0, 1] in `data.py`.
    It is a property of the equation being linear, and it is the assumption
    most likely to be silently wrong if the scaling code is edited later.
    """
    f = load_field()
    def coef(U):
        smp = S.assemble(U, f.dx, f.dt, n_burn=20, n_edge=3)
        xi = S.stlsq(smp.Theta, smp.u_t, threshold=0.01)
        return xi[smp.names.index("u_xx")]

    a, b = coef(f.U), coef(3.0 * f.U + 7.0)
    rel = abs(a - b) / abs(a)
    check("u_xx coefficient survives rescaling u", rel < 1e-6,
          f"{a:.6f} on U, {b:.6f} on 3U+7, relative difference {rel:.2e}")


# 5 -------------------------------------------------------------------------
def test_noise_amplifies_with_derivative_order() -> None:
    """Each derivative multiplies the noise by roughly 1/dx.

    The whole experiment rests on this being the binding constraint, so it is
    measured rather than asserted in prose.
    """
    f = load_field()
    Un = add_noise(f.U, 1e-5, seed=0)
    out = []
    for name, fn in (("u_x", S.d_dx), ("u_xx", S.d2_dx2), ("u_xxx", S.d3_dx3)):
        clean, dirty = fn(f.U, f.dx), fn(Un, f.dx)
        m = np.isfinite(clean) & np.isfinite(dirty)
        out.append((name, float(np.abs(dirty[m] - clean[m]).std()
                                / np.abs(clean[m]).std())))
    ok = out[0][1] < out[1][1] < out[2][1]
    check("noise grows with derivative order", ok,
          "relative error at sigma=1e-5: "
          + ", ".join(f"{n} {v:.3g}" for n, v in out))


# 6 -------------------------------------------------------------------------
def test_recovers_two_terms() -> None:
    """Given a target built from two library columns, return both.

    Sparsity has to be a finding, not a habit: a solver that always returns
    one term would 'confirm' the diffusion equation for the wrong reason.
    The target here is synthesised from the design matrix itself, so this
    isolates the regression from the differentiation.
    """
    U, dx, dt = travelling_wave()
    smp = S.assemble(U, dx, dt, n_burn=3, n_edge=3)
    truth = {"u_x": 2.0, "u_xx": -0.5}
    xi_true = np.array([truth.get(n, 0.0) for n in smp.names])
    y = smp.Theta @ xi_true

    xi = S.stlsq(smp.Theta, y, threshold=0.01)
    got = dict(zip(smp.names, xi))
    ok = (S.active_terms(xi, smp.names) == ["u_x", "u_xx"]
          and all(abs(got[k] - v) / abs(v) < 1e-6 for k, v in truth.items()))
    check("two-term target recovered as two terms", ok,
          f"{S.describe(xi, smp.names)}  (truth: u_t = +2.0000 u_x -0.5000 u_xx)")


if __name__ == "__main__":
    print("implementation checks")
    test_recovers_known_advection()
    test_derivative_is_second_order()
    test_threshold_can_empty_the_model()
    test_linear_rescaling_of_u_is_invisible()
    test_noise_amplifies_with_derivative_order()
    test_recovers_two_terms()
    print()
    if FAILURES:
        sys.exit(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
    print("all checks passed")
