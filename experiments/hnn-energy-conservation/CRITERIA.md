# Pre-registration

Frozen 2026-09-19, before any network was built and before any number was
measured. Not edited afterwards. Deviations are recorded in `README.md`.

## The question

A Hamiltonian Neural Network (Greydanus, Dzamba & Yosinski, 2019) does not
learn a vector field. It learns a scalar `H_theta(q, p)` and *derives* the
vector field from it:

    dq/dt =  dH_theta/dp        dp/dt = -dH_theta/dq

Whatever `H_theta` turns out to be, that field conserves it exactly - the
conservation is a property of the architecture, not something the training
has to discover. The claim is that this buys long-horizon stability that an
ordinary network predicting `(dq/dt, dp/dt)` directly cannot have.

**This measures what building a conservation law into the architecture buys,
and what it costs when the law is false.**

## Data

[`generators/hamiltonian_pendulum`](../../generators/hamiltonian_pendulum/):
200 trajectories of the pendulum `H = p^2/2 - cos q`, 100 time units each,
with the exact vector field at every stored state. Two versions: ideal
(energy conserved to 1e-9) and damped (`dp/dt = -sin q - 0.1 p`, energy lost
at the known rate `-0.1 p^2`).

Both models are trained on the **exact derivatives** at sampled states - the
standard HNN setup - and never see a trajectory during training. Trajectories
are used only to evaluate rollouts.

- **Split by trajectory**: 140 initial conditions for training, 60 held out.
  A held-out trajectory shares no state with any training trajectory.
- **Split by energy** (P5 only): training states restricted to `H < 0.5`,
  evaluation on held-out trajectories with `0.5 < H < 0.9` - still
  librating, but larger swings than anything seen in training.

## Models

Identical bodies: two hidden layers of 64 units, `tanh` activation, so that
`H_theta` is smooth enough to differentiate twice.

- **MLP**: input `(q, p)`, output `(dq/dt, dp/dt)` - 4,546 parameters.
- **HNN**: input `(q, p)`, output the scalar `H_theta`, field by autograd -
  4,481 parameters.

Same optimiser, same learning rate, same number of epochs, same batches,
same seeds. Loss is mean squared error on the vector field.

## Rollouts

Both learned fields are integrated with the same fixed-step RK4 the generator
used, `dt = 0.01`, for the full 100 time units, from held-out initial
conditions. Energy drift is measured with the **true** `H`, evaluated on the
model's trajectory: `|H(x_t) - H(x_0)|`. State error is against the
generator's reference trajectory.

## Predictions

Written before any model was trained.

- **P1** Control. On the ideal system, both models reach a held-out
  derivative RMSE below 1% of the field's RMS. A 2D smooth field that a
  4,500-parameter network cannot fit means the setup is broken, and
  `run_all.py` aborts.
- **P2** Over the 100-unit rollout from held-out initial conditions, the
  MLP's median energy drift is **at least 10 times** the HNN's.
- **P3** The HNN's learned `H_theta` matches the true `H` up to an additive
  constant: an affine fit `H_theta = a H + b` over the training region has
  R² above 0.99 and slope `a` within 5% of 1. The network was never shown
  `H`; if this holds it discovered the energy from the field alone.
- **P4** On the damped system, the HNN's held-out derivative RMSE is **at
  least 3 times** the MLP's. The inductive bias is now wrong - no
  `H_theta` can generate a field with non-zero divergence - and no amount of
  training can remove that floor.
- **P5** Trained only on `H < 0.5` and evaluated on `0.5 < H < 0.9`, the
  HNN's rollout state error is lower than the MLP's. A learned scalar with
  the right structure extrapolates further than a learned 2D field.

Medians over 3 seeds, never best cells.

## What would overturn the story

**P2 failing** would mean that on this system the architectural guarantee
does not translate into rollout stability in practice - possible if the HNN's
`H_theta` is accurate enough on the field but has spurious structure off the
data manifold. **P4 failing** would mean the HNN can fake dissipation well
enough that the wrong inductive bias does not matter, which would undercut
the case for choosing architectures by their conservation laws at all.

## Known in advance

**One system, two dimensions.** The pendulum is the smallest interesting
Hamiltonian system. Nothing here speaks to high-dimensional or chaotic
dynamics, where both the difficulty and the value of conservation are larger.

**The models are trained on exact derivatives.** Real data comes as
trajectories, and derivatives have to be estimated from them - which is where
[`symbolic-regression-pde`](../symbolic-regression-pde/) found the whole
difficulty lives. This experiment deliberately removes that step to isolate
the architecture.

**RK4 is not symplectic**, so even a perfect `H_theta` would show a small
energy drift over 10,000 steps. The generator measured it at about 1e-9 for
the true field; that is the floor, and the HNN's drift is compared against
the MLP's, not against zero.

**`H_theta` is identified only up to a constant**, since the field depends
on gradients. P3 is phrased to allow that and nothing more.
