# MLP

A standard fully-connected feedforward network (`mlp.py`): the
parameter-matched baseline the KAN is compared against in
[`kan_vs_mlp_battery_diffusion`](../../experiments/kan_vs_mlp_battery_diffusion/).
Later experiments define their own dense controls beside the model they
control, sized to their own pre-registration.

Uses `tanh` activations - not a stylistic choice: physics-informed losses
need a second derivative of the network output taken via autograd, and
ReLU-family activations have zero second derivative almost everywhere,
which starves a PDE residual of gradient information. `tanh` is the
standard choice in the physics-informed neural network literature for
exactly this reason (Raissi, Perdikaris & Karniadakis, 2019).
