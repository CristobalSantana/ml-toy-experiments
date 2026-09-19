"""
test_graph.py -- Are the hand-computed features right, and does the
message-passing layer pass messages?

    python test_graph.py

Five checks on graphs whose answers are known by construction. The one that
matters most is the last: a GNN with L layers must be able to tell a node
L hops from the seed apart from one L+1 hops away, and must NOT be able to
tell L+1 from L+2. If it could, something other than message passing is
carrying information; if it could not, the layer is not aggregating.

Run before the experiment; `run_all.py` stops if any check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

import graph as G

HERE = Path(__file__).resolve().parent
GEN = HERE.parent.parent / "generators" / "sir_on_graph" / "outputs"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def path_graph(n: int) -> G.Graph:
    """0-1-2-...-(n-1). Distances from node 0 are the node index."""
    edges = np.array([(i, i + 1) for i in range(n - 1)])
    return G.Graph(n=n, edges=edges, seed=0, infected=np.zeros(n, bool), final_size=0.0)


def triangle_plus_tail() -> G.Graph:
    """Nodes 0,1,2 form a triangle; 3 hangs off 2. Clustering: 0 and 1 are
    1.0 (their two neighbours are connected), 2 is 1/3, 3 is 0."""
    edges = np.array([(0, 1), (1, 2), (0, 2), (2, 3)])
    return G.Graph(n=4, edges=edges, seed=0, infected=np.zeros(4, bool), final_size=0.0)


# 1 -------------------------------------------------------------------------
def test_bfs_distance() -> None:
    g = path_graph(7)
    d = G.bfs_distance(g.adjacency(), 0)
    check("BFS distance on a path", np.array_equal(d, np.arange(7)),
          f"got {d.astype(int).tolist()}, want [0..6]")


# 2 -------------------------------------------------------------------------
def test_clustering() -> None:
    c = G.clustering(triangle_plus_tail().adjacency())
    want = np.array([1.0, 1.0, 1 / 3, 0.0])
    check("clustering coefficient on a triangle with a tail",
          np.allclose(c, want), f"got {np.round(c, 3).tolist()}, want {np.round(want, 3).tolist()}")


# 3 -------------------------------------------------------------------------
def test_pagerank() -> None:
    """On a complete graph every node is equivalent, so PageRank is uniform
    and sums to one."""
    n = 6
    edges = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    g = G.Graph(n=n, edges=edges, seed=0, infected=np.zeros(n, bool), final_size=0.0)
    r = G.pagerank(g.adjacency())
    check("PageRank is uniform on a complete graph and sums to 1",
          np.allclose(r, 1 / n, atol=1e-6) and abs(r.sum() - 1) < 1e-6,
          f"values {np.round(r, 4).tolist()}, sum {r.sum():.6f}")


# 4 -------------------------------------------------------------------------
def test_generator_threshold_recorded() -> None:
    """The generator ran its own physics check; make sure it passed."""
    import json
    s = json.loads((GEN / "sir_params.json").read_text())
    tc = s["threshold_check"]
    low = [r for r in tc if r["R0"] < 0.6]
    high = [r for r in tc if r["R0"] > 2.0]
    ok = (all(r["major_outbreak_share"] < 0.05 for r in low)
          and all(r["major_outbreak_share"] > 0.30 for r in high))
    check("generator's epidemic threshold matches Newman's formula", ok,
          f"major outbreaks {max(r['major_outbreak_share'] for r in low):.0%} at most "
          f"below R0=0.6, {min(r['major_outbreak_share'] for r in high):.0%} at least "
          f"above R0=2")


# 5 -------------------------------------------------------------------------
def test_receptive_field_is_exactly_L_hops() -> None:
    """On a path with the seed at one end, an L-layer GNN's embedding of node
    i can depend on the seed only if i <= L. So nodes L and L+1 must differ
    in embedding (if the layer aggregates), and nodes L+1 and L+2 must be
    identical (if nothing leaks past L hops). Both directions are tested,
    with random weights, so this is a property of the wiring."""
    torch.manual_seed(0)
    g = path_graph(12)
    x, op = G.gnn_inputs(g)
    # make degree uniform so the only asymmetry is the seed indicator
    x[:, 0] = 0.2
    results = []
    for L in (1, 2, 3):
        m = G.GNN(n_layers=L)
        with torch.no_grad():
            h = m.embed(x, op)
        sees = float((h[L] - h[L + 1]).abs().max())
        blind = float((h[L + 1] - h[L + 2]).abs().max())
        results.append((L, sees, blind))
    ok = all(s > 1e-6 and b < 1e-9 for _, s, b in results)
    check("an L-layer GNN sees exactly L hops from the seed", ok,
          "; ".join(f"L={L}: hop L vs L+1 differ by {s:.1e}, hops L+1 vs L+2 by {b:.1e}"
                    for L, s, b in results))


if __name__ == "__main__":
    print("implementation checks")
    if not (GEN / "sir_erdos_renyi.npz").exists():
        sys.exit(f"missing generator output\nrun: python "
                 f"../../generators/sir_on_graph/generate.py")
    test_bfs_distance()
    test_clustering()
    test_pagerank()
    test_generator_threshold_recorded()
    test_receptive_field_is_exactly_L_hops()
    print()
    if FAILURES:
        sys.exit(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
    print("all checks passed")
