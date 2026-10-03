# Pre-registration

Frozen 2026-10-03, after the data was generated and before any network was
trained. Not edited afterwards. Deviations are recorded in `README.md`.

## The question

A Fourier Neural Operator (Li et al., 2021) learns a map between functions
rather than between arrays. Its weights live on Fourier wavenumbers, which
are the same on every grid, so the same trained network can be run on a
grid of any size. The paper's claim is **zero-shot super-resolution**: train
on a coarse grid, evaluate on a fine one, and the error stays about the
same.

There are two readings of that claim, and they are not the same thing:

1. **Consistency**: the network gives the same answer, whatever grid it is
   asked on. Where the true solution has no detail finer than the coarse
   grid can hold, that is all super-resolution needs to mean.
2. **Detail**: on the fine grid, the network produces structure the coarse
   training data could not show.

**This measures both, on a solution whose fine-scale detail is turned up by
a dial, and asks whether the second reading beats interpolating the
network's own coarse answer.**

## Data

[`generators/burgers_1d`](../../generators/burgers_1d/): viscous Burgers'
equation `u_t + u u_x = nu u_xx` on a periodic line, 1,200 random starts,
solved to `t = 1` at three viscosities, checked against the exact Cole-Hopf
solution. The starts use Fourier modes 1 to 16 only, so **a grid of 64
points holds the whole start**: everything a model needs to know is in its
64-point input, at every resolution.

What the 64-point grid cannot hold is the *answer's* detail - the share of
`u(t = 1)`'s energy above wavenumber 32, measured by the generator:

| nu | energy a 64-point grid cannot hold |
|---|---|
| 0.1 | 1.8e-8 (smooth) |
| 0.03 | 5.4e-4 |
| 0.01 | 5.6e-3 (fronts much narrower than one coarse cell) |

- Samples 0-999 train, 1000-1199 are held out. The same split at every
  viscosity.
- Resolutions are nested subsamplings of the stored 512-point grid: 64, 128,
  256, 512.

## Models

- **FNO**: width 32, 16 Fourier modes, 4 layers - 139,777 parameters.
- **CNN**: kernel 5, 63 channels, dilations to 16, circular padding -
  140,428 parameters. At 64 points its receptive field covers the whole
  domain.

Both take `(u0, sin x, cos x)` and return `u(t = 1)` on the same grid.
Loss: mean relative L2 error, the FNO paper's. Adam, learning rate 1e-3
halved every 60 epochs, weight decay 1e-4, batch 20, 300 epochs.

Trained per viscosity, per seed, at **64 points**. The FNO is also trained
at **256 points** (P5 only), same architecture, same budget.

## What is measured

Mean relative L2 error over the 200 held-out samples, on the grid the
evaluation is at, for each model evaluated at 64, 128, 256 and 512 points:

- **zero-shot**: the network trained at 64 points, run directly on the
  finer grid's input.
- **interpolated**: the network trained at 64 points, run at 64 points, its
  output linearly interpolated (periodically) onto the finer grid. The
  cheapest thing anyone would do instead.

Medians over 3 seeds, never best cells.

## Predictions

Written before any model was trained.

- **P1** Control. At nu = 0.1, the FNO trained and evaluated at 64 points
  reaches a mean relative L2 error below 1%. The solution is smooth and the
  input complete; a 140,000-parameter operator that cannot fit it means the
  setup is broken, and `run_all.py` aborts.
- **P2** At the training resolution, the FNO's error is **at most half**
  the CNN's, at every viscosity.
- **P3** Consistency, where the answer is smooth (nu = 0.1). Evaluated
  zero-shot at 256 points, the FNO's error is **within 1.25 times** its
  error at 64 points. The CNN's, by contrast, is **at least 3 times** its
  error at 64.
- **P4** Detail, where the answer is not smooth (nu = 0.01). At 256 points
  the FNO's zero-shot error is **not more than 10% lower** than the error of
  its own 64-point output, interpolated - that is,
  `zero-shot >= 0.9 x interpolated`. Zero-shot super-resolution adds no
  detail that interpolation would not.
- **P5** The gap is the data, not the architecture. At nu = 0.01 and 256
  points, the same FNO **trained at 256 points** has at most half the error
  of the one trained at 64 and run zero-shot.

## What would overturn the story

**P4 failing** - the zero-shot FNO clearly beating interpolation at nu =
0.01 - would mean it learned sub-grid structure from data that never showed
it, from the physics implied by the coarse samples alone. That is the strong
reading of the paper's claim, and this would be evidence for it. **P3
failing** for the FNO would mean even the weak reading does not hold:
running the network on a different grid changes its answer where the truth
does not change.

## Known in advance

**The input is complete at every resolution.** The starts are band-limited
to mode 16, so a finer grid shows the network nothing about the start that
the coarse grid did not. Any difference between zero-shot outputs at 64 and
at 256 points comes from the network, not from new information. This is the
fairest possible setting for super-resolution: everything needed to predict
the fine answer is in the coarse input.

**Aliasing.** The FFT of a network's hidden layer on a 64-point grid folds
any content above wavenumber 32 back onto lower wavenumbers; on 256 points
it does not. The pointwise nonlinearities make that content. So the FNO's
spectral layers do not see the same coefficients on the two grids, and
exact grid-independence is not expected even in principle. P3 allows 25%.

**One equation, one dimension.** Burgers is the standard first test for
neural operators and the FNO paper's own. Nothing here speaks to 2D flows,
where the paper's super-resolution figure was made.

**Errors are measured on the grid the evaluation is at.** A relative L2
error on 256 points weights the fronts by how many of the 256 points fall
on them, which is the honest measure of what a finer grid adds.
