# Pre-registration

Frozen 2026-09-19, before any network was built and before any number was
measured. Not edited afterwards. Deviations are recorded in `README.md`.

## The question

A graph neural network passes messages along edges, so what it computes for
a node depends on that node's neighbourhood. That is the whole pitch: **it
learns from structure that a per-node model cannot see.**

The honest question is how much of what it learns is structure that could
not have been written down by hand. If a graph neural network beats a plain
model on graph data, and a plain model given one well-chosen structural
feature closes most of that gap, the architecture's contribution is the
convenience of not having to choose the feature - which is worth something,
and is not the same thing as learning structure.

## Data

[`generators/sir_on_graph`](../../generators/sir_on_graph/): SIR epidemics
on 300-node random graphs with mean degree 6, one random seed node each.
Erdős–Rényi (homogeneous degrees) and Barabási–Albert (heavy-tailed), 300
graphs of each, `beta = 0.06`, `gamma = 0.2`. The generator checks the
epidemic threshold against Newman's formula on every run.

**Task.** Node-level, binary: given the graph and which node was the seed,
did this node ever get infected? Restricted to **major outbreaks** - runs
that reached at least 10% of nodes - because in a run that fizzled after
three nodes there is nothing structural to predict. The share of runs that
qualify is reported.

**Split by graph**, never by node: 80% of qualifying graphs train, 20% are
held out. A held-out node's graph was never seen.

## Models

- **`majority`** - predicts the base rate. AUC 0.5 by construction.
- **`logreg_dist`** - logistic regression on **one** feature: BFS distance
  from the seed node. The feature a person would write down first.
- **`logreg_features`** and **`mlp_features`** - logistic regression and a
  two-layer MLP (64 hidden) on five hand-computed structural features:
  degree, log degree, BFS distance from the seed, local clustering
  coefficient, PageRank. Standardised on the training graphs.
- **`gnn`** - message passing in plain PyTorch, no graph library. Node input
  is **only** normalised degree and a seed indicator - no distances, no
  centralities. `L` layers of `h' = ReLU(W1 h + W2 mean(h over neighbours))`
  with 64 hidden units, then a linear readout per node. `L` swept over
  {1, 2, 3, 4, 6, 8}; the reported GNN is the best `L` on a validation split
  of the training graphs, chosen before the held-out graphs are scored.

All models: Adam, binary cross-entropy, same epochs, 3 seeds, medians.

## Predictions

- **P1** Control. The generator's threshold check passes, and
  `logreg_features` reaches held-out AUC above 0.6 on Erdős–Rényi graphs.
  If hand-computed structure cannot predict infection at all, the task is
  not about structure and `run_all.py` aborts.
- **P2** On Erdős–Rényi graphs, the GNN beats `mlp_features` by at least
  0.02 AUC on held-out graphs.
- **P3** `logreg_dist` - one feature, distance from the seed - gets **more
  than halfway** from chance to the GNN: `AUC(dist) - 0.5 > 0.5 *
  (AUC(gnn) - 0.5)`. Most of the gain is one feature a person would have
  written down first.
- **P4** Structural drift. Trained on Erdős–Rényi and evaluated on
  Barabási–Albert, every model loses AUC, and **the GNN loses less** than
  `mlp_features`, in AUC points. The hand-computed features are calibrated
  to one degree distribution; message passing is not.
- **P5** GNN accuracy saturates with depth: AUC at `L = 8` is no more than
  0.01 above AUC at `L = 4`. Beyond the typical seed-to-node distance, which
  is about 3 to 4 hops on these graphs, extra layers have nothing to fetch.

Medians over 3 seeds, never best cells.

## What would overturn the story

**P3 failing** - the GNN well ahead of what distance alone explains - would
mean the architecture is learning structure that one obvious feature does
not capture, which is the strongest case for it. **P4 failing** in the
other direction - the GNN degrading *more* under structural drift - would
mean its learned aggregation is tuned to the training degree distribution
in a way the explicit features are not.

## Known in advance

**One process, one task.** Infection under SIR from a single seed. A
different dynamic on the same graphs - say, opinion spread with thresholds -
could reward structure very differently.

**The hand-computed features were chosen with the answer in mind.** Distance
from the seed is an obvious feature *for this task*. Part of the argument
for graph networks is that in real problems nobody knows which feature is
obvious in advance, and this experiment cannot measure that.

**Graphs of 300 nodes.** Small enough that BFS distance is cheap and
diameter is 3 to 4. Message passing's depth-versus-diameter tradeoff looks
different on graphs a thousand times larger.

**Stochastic labels.** Two identical graphs with the same seed can produce
different infection sets. There is an irreducible error no model can beat,
and it is not measured here; AUC is compared between models, not against a
ceiling.
