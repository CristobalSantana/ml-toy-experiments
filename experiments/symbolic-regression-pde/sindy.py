"""
sindy.py -- Sparse identification of a PDE from a grid of numbers.

The method (Brunton, Proctor & Kutz 2016; Rudy et al. 2017) is short enough
to state completely:

  1. estimate du/dt and a library of candidate spatial terms by finite
     differences on the sampled field,
  2. solve  u_t = Theta @ xi  by least squares,
  3. zero every coefficient below a threshold, refit on what survives,
     repeat.

Step 1 is where it lives or dies, and it is the step that gets the least
attention. Differentiating sampled data amplifies noise by roughly 1/dx per
derivative order, and this library goes to third order. The rest of this
module is arithmetic; `derivatives` is the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter

# The library, fixed in CRITERIA.md. Order matters only for readability.
LIBRARY_NAMES = ["1", "u", "u^2", "u^3", "u_x", "u_xx", "u_xxx",
                 "u*u_x", "u*u_xx", "u^2*u_x", "(u_x)^2"]
TRUE_TERM = "u_xx"


# --------------------------------------------------------------------------
# derivatives
#
# Every routine writes NaN where the stencil does not fit, rather than
# one-siding the edge. A one-sided stencil at a Neumann boundary quietly
# encodes the boundary condition into the data, which is a way of telling the
# method part of the answer. The caller trims and then asserts no NaN
# survives.
# --------------------------------------------------------------------------

def d_dx(U: np.ndarray, dx: float) -> np.ndarray:
    """Second-order central first derivative along x (axis 1)."""
    d = np.full_like(U, np.nan)
    d[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / (2.0 * dx)
    return d


def d2_dx2(U: np.ndarray, dx: float) -> np.ndarray:
    """Second-order central second derivative along x."""
    d = np.full_like(U, np.nan)
    d[:, 1:-1] = (U[:, 2:] - 2.0 * U[:, 1:-1] + U[:, :-2]) / dx**2
    return d


def d3_dx3(U: np.ndarray, dx: float) -> np.ndarray:
    """Second-order central third derivative along x (5-point stencil)."""
    d = np.full_like(U, np.nan)
    d[:, 2:-2] = (U[:, 4:] - 2.0 * U[:, 3:-1] + 2.0 * U[:, 1:-3]
                  - U[:, :-4]) / (2.0 * dx**3)
    return d


def d_dt(U: np.ndarray, dt: float) -> np.ndarray:
    """Second-order central first derivative along t (axis 0)."""
    d = np.full_like(U, np.nan)
    d[1:-1, :] = (U[2:, :] - U[:-2, :]) / (2.0 * dt)
    return d


def smooth(U: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay along x then along t.

    A local polynomial fit, which is what one reaches for before
    differentiating noisy samples. `window` must be odd and greater than
    `poly`; both are swept in the experiment rather than tuned here.
    """
    if window <= poly:
        raise ValueError(f"savgol window {window} must exceed polyorder {poly}")
    if window % 2 == 0:
        raise ValueError(f"savgol window {window} must be odd")
    S = savgol_filter(U, window, poly, axis=1, mode="nearest")
    return savgol_filter(S, window, poly, axis=0, mode="nearest")


# --------------------------------------------------------------------------
# library
# --------------------------------------------------------------------------

def build_library(U: np.ndarray, dx: float) -> tuple[np.ndarray, list[str]]:
    """The 11 candidate terms, each the same shape as U, NaN where invalid."""
    ux, uxx, uxxx = d_dx(U, dx), d2_dx2(U, dx), d3_dx3(U, dx)
    cols = {
        "1": np.ones_like(U),
        "u": U,
        "u^2": U**2,
        "u^3": U**3,
        "u_x": ux,
        "u_xx": uxx,
        "u_xxx": uxxx,
        "u*u_x": U * ux,
        "u*u_xx": U * uxx,
        "u^2*u_x": U**2 * ux,
        "(u_x)^2": ux**2,
    }
    assert list(cols) == LIBRARY_NAMES, "library drifted from CRITERIA.md"
    return np.stack([cols[n] for n in LIBRARY_NAMES], axis=-1), LIBRARY_NAMES


@dataclass
class Sample:
    """Flattened, trimmed design matrix and target, plus where it came from."""
    Theta: np.ndarray          # (n, 11)
    u_t: np.ndarray            # (n,)
    names: list[str]
    t_index: np.ndarray        # (n,) row of the original grid, for time splits


def assemble(U: np.ndarray, dx: float, dt: float, n_burn: int, n_edge: int,
             U_clean: np.ndarray | None = None) -> Sample:
    """Build the regression problem from a field.

    `U_clean`, when given, supplies the target `u_t`. The experiment uses it
    to score a model against the physics rather than against the noise it was
    asked to see through; the design matrix always comes from the noisy `U`.
    """
    Theta, names = build_library(U, dx)
    ut = d_dt(U if U_clean is None else U_clean, dt)

    nt = U.shape[0]
    ts = slice(n_burn, nt - 1)
    xs = slice(n_edge, U.shape[1] - n_edge)
    Theta, ut = Theta[ts, xs, :], ut[ts, xs]

    if not np.isfinite(Theta).all() or not np.isfinite(ut).all():
        raise ValueError(
            f"NaN survived trimming: n_edge={n_edge} does not cover the widest "
            f"stencil, or n_burn={n_burn} starts before the first valid row.")

    rows = np.repeat(np.arange(ts.start, ts.stop)[:, None], ut.shape[1], axis=1)
    return Sample(Theta=Theta.reshape(-1, Theta.shape[-1]),
                  u_t=ut.reshape(-1), names=list(names),
                  t_index=rows.reshape(-1))


# --------------------------------------------------------------------------
# regression
# --------------------------------------------------------------------------

def stlsq(Theta: np.ndarray, y: np.ndarray, threshold: float,
          max_iter: int = 20) -> np.ndarray:
    """Sequentially thresholded least squares.

    Columns and target are scaled to unit norm before thresholding, so the
    threshold is dimensionless: 0.01 means "this term moves the target by
    less than 1% of its norm". Without that, the threshold would be compared
    against coefficients whose units differ by ten orders of magnitude
    between `u` and `u_xxx`, and would select on units rather than on
    relevance. Coefficients are returned in the original scale.
    """
    cs = np.linalg.norm(Theta, axis=0)
    cs[cs == 0] = 1.0
    ys = np.linalg.norm(y)
    if ys == 0:
        return np.zeros(Theta.shape[1])
    Tn, yn = Theta / cs, y / ys

    xi = np.linalg.lstsq(Tn, yn, rcond=None)[0]
    keep = np.ones(len(xi), dtype=bool)
    for _ in range(max_iter):
        small = np.abs(xi) < threshold
        new_keep = ~small
        if np.array_equal(new_keep, keep) and not small.any():
            break
        keep = new_keep
        xi = np.zeros(len(xi))
        if not keep.any():
            break
        xi[keep] = np.linalg.lstsq(Tn[:, keep], yn, rcond=None)[0]
        if (np.abs(xi[keep]) >= threshold).all():
            break
    return xi * ys / cs


def active_terms(xi: np.ndarray, names: list[str]) -> list[str]:
    return [n for n, c in zip(names, xi) if c != 0.0]


def describe(xi: np.ndarray, names: list[str]) -> str:
    """The discovered equation, as a line a person can read."""
    parts = [f"{c:+.4f} {n}" for n, c in zip(names, xi) if c != 0.0]
    return "u_t = " + (" ".join(parts) if parts else "0   (empty model)")
