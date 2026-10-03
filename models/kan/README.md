# KAN - Kolmogorov-Arnold Network

A minimal, self-contained implementation (~100 lines, `kan.py`) of a
Kolmogorov-Arnold Network. No `pykan` / `efficient-kan` dependency, so the
parameter count is exactly computable by hand (needed to match KAN against
an MLP baseline fairly) and every line is explainable in a technical
article.

A KAN layer replaces "linear map + fixed pointwise nonlinearity" with a
**learnable 1D function on every input-output edge**:

```
phi(x) = w_base * SiLU(x) + spline(x)
```

`spline(x)` is a linear combination of B-spline basis functions on a fixed,
uniform knot grid (no adaptive grid updates, for training determinism). The
residual `w_base * SiLU(x)` term is what makes the layer trainable by plain
backprop from a generic initialization (Liu et al. 2024, eq. 2.10).

A `KANLinear(in_features, out_features, grid_size=G, spline_order=k)` layer
has exactly `in_features * out_features * (1 + G + k)` parameters - one
base weight plus `G + k` spline coefficients per edge. No separate
activation function sits between KAN layers; the nonlinearity already lives
on every edge.

Reference: Liu et al., *"KAN: Kolmogorov-Arnold Networks"* (2024).

Used by [`kan_vs_mlp_battery_diffusion`](../../experiments/kan_vs_mlp_battery_diffusion/).
