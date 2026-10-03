# models

Two architectures live in this folder: the Kolmogorov-Arnold Network and
the MLP it was compared against, both written for the first experiment,
[`kan_vs_mlp_battery_diffusion`](../experiments/kan_vs_mlp_battery_diffusion/).

Every architecture since then was written for one experiment and lives in
that experiment's folder, next to the pre-registration that fixed its size
before it was trained. A model's parameter count is part of what an
experiment predicts, so the model and the prediction are kept together.

This is the index of all of them.

| Model | What it is | Where |
|---|---|---|
| Kolmogorov-Arnold Network | learnable spline functions on every edge instead of fixed activations | [`kan/`](kan/) |
| MLP | `tanh` feedforward network, the parameter-matched baseline for the KAN | [`mlp/`](mlp/) |
| Echo state network | reservoir computing: fixed random recurrent weights, only the linear readout is trained | [`solar-forecast-skill/models.py`](../experiments/solar-forecast-skill/models.py) |
| Hyperdimensional classifier | records as products of random 10,000-dimensional vectors, classes as their sums; training is one pass of addition | [`hdc-vs-boosting/hdc.py`](../experiments/hdc-vs-boosting/hdc.py) |
| Spiking network | leaky integrate-and-fire neurons trained with a surrogate gradient, with a measured operation counter | [`spiking-energy-claim/snn.py`](../experiments/spiking-energy-claim/snn.py) |
| Hamiltonian Neural Network | learns a scalar `H_θ` and derives the vector field as its symplectic gradient, so the field conserves `H_θ` by construction | [`hnn-energy-conservation/models.py`](../experiments/hnn-energy-conservation/models.py) |
| Message-passing GNN | `h' = ReLU(W₁h + W₂·mean of h over neighbours)`, stacked `L` times, in plain PyTorch | [`gnn-epidemic-structure/graph.py`](../experiments/gnn-epidemic-structure/graph.py) |
| Fourier Neural Operator | spectral convolution: weights on the lowest Fourier modes, so one trained network runs on any grid | [`fno-super-resolution/models.py`](../experiments/fno-super-resolution/models.py) |

Dense controls that an experiment sizes to its own pre-registration - the
MLPs in the HNN, spiking and GNN experiments, the CNN in the FNO one - are
defined beside the model they control and are not listed separately.
