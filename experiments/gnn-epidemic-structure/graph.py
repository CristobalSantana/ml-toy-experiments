"""
graph.py -- Graphs from the generator, structural features by hand, and a
message-passing network in plain PyTorch.

No graph library. The message-passing layer is eleven lines and the point of
this experiment is to see exactly what it does, so it is written out:

    h' = ReLU( W1 h  +  W2 * mean of h over neighbours )

Mean aggregation, GraphSAGE-style. Stacking L of these lets a node see L hops
out. The node's own input is only its degree and whether it is the seed;
anything about distance, centrality or clustering has to come from the
message passing or not at all.

The hand-computed features are the honest competitor. They are what a person
who understood the task would write down, and the experiment measures how
much of the GNN's advantage survives against them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import torch
import torch.nn as nn

# the operator is built once per graph from a valid COO triple; the invariant
# check costs time on every construction and would only re-verify scipy
torch.sparse.check_sparse_tensor_invariants.disable()

FEATURE_NAMES = ["degree", "log_degree", "dist_to_seed", "clustering", "pagerank"]


@dataclass
class Graph:
    n: int
    edges: np.ndarray          # (E, 2)
    seed: int
    infected: np.ndarray       # (n,) bool
    final_size: float

    def adjacency(self) -> sp.csr_matrix:
        A = sp.coo_matrix((np.ones(len(self.edges)),
                           (self.edges[:, 0], self.edges[:, 1])), shape=(self.n, self.n))
        return (A + A.T).tocsr()


def load_family(path, min_final_size: float) -> tuple[list[Graph], int]:
    """Every graph of a family, keeping only major outbreaks. Returns the
    graphs and how many were dropped for fizzling."""
    z = np.load(path)
    n = int(z["n_nodes"])
    n_graphs = len(z["graph_final_size"])
    graphs, dropped = [], 0
    for g in range(n_graphs):
        fs = float(z["graph_final_size"][g])
        if fs < min_final_size:
            dropped += 1
            continue
        graphs.append(Graph(
            n=n, edges=z["edges"][z["edge_graph"] == g].astype(np.int64),
            seed=int(z["graph_seed_node"][g]),
            infected=z["node_infected"][z["node_graph"] == g].astype(bool),
            final_size=fs))
    return graphs, dropped


# --------------------------------------------------------------------------
# hand-computed structure
# --------------------------------------------------------------------------

def bfs_distance(A: sp.csr_matrix, source: int) -> np.ndarray:
    d = csg.shortest_path(A, unweighted=True, indices=source)
    # unreachable (isolated nodes exist in Erdős–Rényi): one hop past the
    # farthest reachable node, so the feature stays finite and ordered
    far = np.isinf(d)
    d[far] = np.nanmax(d[~far]) + 1 if (~far).any() else 0
    return d


def clustering(A: sp.csr_matrix) -> np.ndarray:
    k = np.asarray(A.sum(1)).ravel()
    tri2 = (A @ A @ A).diagonal()            # 2 x triangles through each node
    denom = k * (k - 1)
    return np.where(denom > 0, tri2 / np.maximum(denom, 1), 0.0)


def pagerank(A: sp.csr_matrix, damping: float = 0.85, iters: int = 100) -> np.ndarray:
    n = A.shape[0]
    k = np.asarray(A.sum(1)).ravel()
    P = sp.diags(np.where(k > 0, 1.0 / np.maximum(k, 1), 0.0)) @ A   # row-stochastic
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        r = (1 - damping) / n + damping * (P.T @ r + r[k == 0].sum() / n)
    return r


def node_features(g: Graph) -> np.ndarray:
    """(n, 5): the five hand-computed features, in FEATURE_NAMES order."""
    A = g.adjacency()
    k = np.asarray(A.sum(1)).ravel()
    return np.stack([k, np.log1p(k), bfs_distance(A, g.seed),
                     clustering(A), pagerank(A)], 1)


def gnn_inputs(g: Graph) -> tuple[torch.Tensor, torch.Tensor]:
    """What the GNN is allowed to see: (n, 2) of normalised degree and a
    seed indicator, plus the mean-aggregation operator D^-1 A as a sparse
    tensor."""
    A = g.adjacency()
    k = np.asarray(A.sum(1)).ravel()
    x = np.stack([k / 10.0, (np.arange(g.n) == g.seed).astype(float)], 1)
    P = sp.diags(np.where(k > 0, 1.0 / np.maximum(k, 1), 0.0)) @ A
    P = P.tocoo()
    op = torch.sparse_coo_tensor(
        np.vstack([P.row, P.col]), P.data, (g.n, g.n), dtype=torch.float32
    ).coalesce()
    return torch.tensor(x, dtype=torch.float32), op


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

class MeanAggLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.self_w = nn.Linear(n_in, n_out)
        self.nbr_w = nn.Linear(n_in, n_out, bias=False)

    def forward(self, h: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.self_w(h) + self.nbr_w(torch.sparse.mm(op, h)))


class GNN(nn.Module):
    """L rounds of mean aggregation, then one logit per node."""

    def __init__(self, n_layers: int, n_in: int = 2, n_hidden: int = 64):
        super().__init__()
        dims = [n_in] + [n_hidden] * n_layers
        self.layers = nn.ModuleList(MeanAggLayer(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(dims[-1], 1)
        self.n_layers = n_layers

    def embed(self, x: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, op)
        return h

    def forward(self, x: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
        return self.out(self.embed(x, op)).squeeze(-1)


class FeatureMLP(nn.Module):
    def __init__(self, n_in: int, n_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU(),
                                 nn.Linear(n_hidden, n_hidden), nn.ReLU(),
                                 nn.Linear(n_hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FeatureLogReg(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.lin = nn.Linear(n_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).squeeze(-1)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train_gnn(model: GNN, graphs: list, inputs: list, *, epochs: int, lr: float,
              seed: int) -> None:
    """One graph per step, shuffled each epoch. Graphs are small enough that
    a full-graph forward is one sparse matmul per layer."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    ys = [torch.tensor(gr.infected, dtype=torch.float32) for gr in graphs]
    for _ in range(epochs):
        for i in torch.randperm(len(graphs), generator=g).tolist():
            x, op = inputs[i]
            opt.zero_grad()
            lossf(model(x, op), ys[i]).backward()
            opt.step()


def train_tabular(model: nn.Module, X: torch.Tensor, y: torch.Tensor, *,
                  epochs: int, batch: int, lr: float, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            lossf(model(X[idx]), y[idx]).backward()
            opt.step()


@torch.no_grad()
def predict_gnn(model: GNN, inputs: list) -> np.ndarray:
    model.eval()
    return np.concatenate([model(x, op).numpy() for x, op in inputs])


@torch.no_grad()
def predict_tabular(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    return model(X).numpy()
