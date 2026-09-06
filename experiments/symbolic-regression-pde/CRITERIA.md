# Pre-registration

Frozen 2026-09-05, before any candidate library was built and before any
regression was run. Nothing below is edited after the fact. Deviations,
if any, are recorded in `README.md`.

## The question

Sparse symbolic regression (SINDy / PDE-FIND) claims to recover a governing
equation from data alone. This repository contains a field produced by an
equation that is known exactly:

    dc/dt = D * d2c/dx2

with `D = 4.0e-14 m^2/s`, on `x in [0, 6e-6] m` and `t in [0, 1200] s`,
sampled on an 801 x 161 grid by Crank-Nicolson.

So the answer is known before the method runs. **The question is not whether
the method can be made to work. It is where it stops working, and what it
does at the moment it stops.**

## Non-dimensionalisation, and the number that has to come out

Working in `x~ = x/L`, `t~ = t/T`, the equation becomes

    du/dt~ = (D*T/L^2) * d2u/dx~2

and `D*T/L^2 = 4.0e-14 * 1200 / (6.0e-6)^2 = 4/3` exactly.

Because the equation is **linear in u**, any linear rescaling of `u` leaves
that coefficient unchanged. So the target is a single number, `1.333333...`,
on a single library term, and every other candidate coefficient is exactly
zero. This is asserted in `test_sindy.py`, not assumed.

## Method, fixed in advance

- **Library** (11 terms): `1, u, u^2, u^3, u_x, u_xx, u_xxx, u*u_x, u*u_xx,
  u^2*u_x, (u_x)^2`.
- **Derivatives**: second-order central finite differences on the grid.
- **Regression**: sequentially thresholded least squares (STLSQ), threshold
  applied to coefficients normalised by column scale, 20 iterations max.
- **Domain trimming**: the first `n_burn = 20` time steps are dropped (the
  surface flux switches on abruptly at t=0 and the numerical solution has a
  Rannacher-damped start-up transient there), and `n_edge = 3` grid points
  are dropped from each end in x (a central stencil cannot be evaluated at a
  Neumann boundary without ghost nodes, and inventing them would be feeding
  the method the answer).
- **Noise**: additive Gaussian on `c`, with sigma given as a fraction of the
  concentration *range* `max(c) - min(c)`, applied before any derivative is
  taken. Levels: `0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2`.
- **Seeds**: 5 per noise level. Reported as medians, not best cells.
- **Threshold**: swept over `1e-4 ... 1e-1`; the reported model at each noise
  level is the one at the *fixed* threshold `0.01`. The sweep is reported
  separately as a sensitivity check, not used to pick a winner per cell.

## Predictions

Written before running anything.

- **P1** On noiseless data the method recovers exactly one term, `u_xx`, with
  a coefficient within 1% of `4/3`.
- **P2** Recovery fails as a cliff, not a slope: there is a noise level below
  which the correct single term is recovered with under 5% coefficient error
  and above which the recovered term set is wrong, and the transition spans
  less than one decade of sigma.
- **P3** At the cliff the failure mode is **adding spurious terms**, not
  dropping `u_xx`.
- **P4** Savitzky-Golay smoothing before differentiation moves the cliff by
  at least one order of magnitude in sigma.
- **P5** At a noise level where the equation is still recovered, a gradient
  booster predicting `u_t` from the same library columns achieves **lower**
  error than the recovered equation on the time range it was trained on, and
  **higher** error on the held-out later half of the time domain.

P1 is the control. If it fails, the implementation is wrong and every other
number is measuring the bug; `run_all.py` aborts.

## What would make this uninteresting

If every noise level either works or fails identically, there is no cliff to
locate and P2/P3/P4 are vacuous. If that happens it gets reported as such.
