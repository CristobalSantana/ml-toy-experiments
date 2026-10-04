# Pre-registration

Frozen 2026-10-03, after the test sets were generated and before any
network was trained. Not edited afterwards. Deviations are recorded in
`README.md`.

## The question

State-space models of the Mamba family (Gu & Dao, 2023; Dao & Gu, 2024)
are sold as Transformers at linear cost: they read a sequence with a
recurrence over a state of fixed size, so memory does not grow with
length. The counter-argument (Arora et al., 2023; Jelassi et al., 2024) is
that this is also exactly their limit. Attention keeps every token and can
look any of them up; a fixed state has to compress, and recalling `N`
arbitrary facts needs a state that grows with `N`.

The usual demonstration is a single curve where the SSM falls away and the
Transformer does not. That shows *a* gap. It does not show that the gap is
the state, rather than training, width, or the particular model.

**This turns the state size by itself - the dimension `N` of `B_t` and
`C_t`, every other part held fixed - and measures how many key-value
pairs each size can recall.** If the limit is the state, capacity should
grow with `N`, roughly in proportion. It also switches selection off at
the largest `N`, which is the other half of Mamba's claim: that making the
recurrence depend on the input is what lets it recall at all.

## Data

[`generators/associative_recall`](../../generators/associative_recall/):
multi-query associative recall. `N` key-value pairs, then every key again
in a new order; at each query the model must produce that key's value. 256
keys, 256 values, the pairing random in every sequence. The generator
checks that a lookup in the sequence answers every query, and that no fixed
key-to-value or position-to-value table does better than chance (1/256).

Pair counts 4, 8, 16, 32, 64 - sequences of 12 to 192 tokens. Test: the
generator's 1,000 fixed sequences per count. Training: fresh sequences
every step from the generator's sampler, on a seed stream the test sets
never use.

## Models

Two-layer language models, `d_model = 64`, identical except the mixer
(see `models.py`):

- **Transformer**: causal softmax attention, 4 heads of 16, rotary
  positions.
- **Selective SSM**, Mamba-2 style: 8 heads of 16, state dimension
  `N` in **4, 8, 16, 32** - a recurrent state of `128 N` numbers per layer.
  Nothing else changes with `N` except the two projections that make
  `B_t` and `C_t`.
- **LTI SSM**: the same block at `N = 32` with `B`, `C`, the step size and
  the decays learned as constants rather than computed from the input.

One model per (architecture, `N`, seed), trained on all five pair counts in
rotation: batch 64, one pair count per step, cycling 4, 8, 16, 32, 64.
AdamW, learning rate 1e-3, 300 warm-up steps then cosine decay to zero,
weight decay 0.1, gradient clipping at 1.0, 7,500 steps. Cross-entropy on
the query positions only. 3 seeds.

## What is measured

**Accuracy**: the share of query positions where the most likely token is
the right value, on the 1,000 test sequences of each pair count.

**Capacity, `K90`**: the pair count at which accuracy falls to 90%,
interpolated linearly in `log2(pairs)` between the last count at or above
90% and the first below it. If accuracy is below 90% already at 4 pairs,
`K90` is recorded as *below 4*; if it never falls below 90%, as *at least
64*. Medians over seeds, with censored values placed at their bound.

Also reported, not predicted: each model's inference memory per layer -
the SSM's state, the Transformer's key-value cache at each length.

## Predictions

Written before any model was trained.

- **P1** Control. The Transformer and the selective SSM at `N = 32` both
  reach 95% accuracy at 4 pairs. Four pairs is 32 bits; a model that cannot
  recall them is not trained, and `run_all.py` aborts.
- **P2** The Transformer reaches **at least 95%** at every pair count up to
  64 - with 16-dimensional keys per head, fewer dimensions than pairs.
- **P3** The state runs out. At 64 pairs **every** selective SSM, `N = 4`
  to `32`, is below 90%.
- **P4** Capacity grows with the state, roughly in proportion: the slope of
  `log2 K90` against `log2 N`, fitted over the selective SSMs whose median
  `K90` is not censored, lies **between 0.5 and 1.5**. Fewer than three
  uncensored sizes counts as a failure - the scaling was not measured.
- **P5** Selection is what makes recall possible. The LTI SSM's `K90` is
  **at most a quarter** of the selective SSM's at the same `N = 32`.

## What would overturn the story

**P4 failing with a slope near zero** - capacity that does not move when
the state is multiplied by eight - would mean that at this scale the state
is not what limits recall, and the fixed-state argument, however sound in
the limit, is not what decides the comparison here. **P3 failing** would
mean a recurrent state of a few thousand numbers holds 64 arbitrary
bindings, 512 bits, well enough that the gap to attention does not open in
this range at all.

## Known in advance

**The quadratic form.** The SSM is trained through Mamba-2's chunked
algorithm, which computes the recurrence exactly but not step by step.
`test_models.py` checks it against an explicit recurrence over the state.
The function, and so the capacity, is the recurrence's.

**A finite vocabulary changes the arithmetic.** With 256 values, a state
does not have to store each value exactly, only well enough to pick it out
of 256 at the end. Capacity measured this way is therefore an upper bound
on what the same state would hold of open-ended content.

**One learning rate.** Published comparisons sweep the learning rate per
model. This uses one, the same for all, with P1 as the guard against a
model that simply failed to train. A model that trains at 4 pairs and fails
at 64 has not failed to train.

**Small models.** `d_model = 64`, two layers. The question is how capacity
moves with the state, not what a production model recalls.
