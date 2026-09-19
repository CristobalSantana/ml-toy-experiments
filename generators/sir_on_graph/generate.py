"""
generate.py -- Epidemics on random graphs, with a threshold the theory predicts.

The process
-----------
Discrete-time SIR on a graph. Every node is susceptible, infected or
recovered. Each step, every infected node passes the infection to each of its
susceptible neighbours independently with probability beta, and then recovers
with probability gamma. Recovered nodes are immune. The run ends when nobody
is infected. One node is infected at the start, chosen at random.

Two graph families with the same mean degree and very different shapes:

    Erdos-Renyi        every pair connected with probability p; degrees
                       cluster tightly around the mean
    Barabasi-Albert    grown by preferential attachment; a few hubs with
                       very many connections and a heavy tail

The known result
----------------
On a random graph, SIR is bond percolation with transmissibility T - the
probability that an infected node passes the disease across a given edge
before it recovers. For a geometric infectious period,

    T = 1 - gamma (1 - beta) / (1 - (1 - gamma)(1 - beta))

and the epidemic threshold is (Newman 2002)

    R0 = T * (<k^2> - <k>) / <k>       outbreak possible iff R0 > 1

The second moment <k^2> is what makes the two families behave differently at
the same beta and gamma: hubs raise <k^2>, and an epidemic that fizzles on an
Erdos-Renyi graph can take off on a Barabasi-Albert one. That is the
structural fact the consuming experiment is about.

`check_threshold` sweeps beta on Erdos-Renyi graphs and confirms that major
outbreaks appear where the formula says they should. A simulator with a bug
in the transmission step would pass a smoke test and fail this.

Deterministic given the seed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import scipy.sparse as sp         # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "outputs"


@dataclass(frozen=True)
class SIRParams:
    n_nodes: int = 300
    mean_degree: float = 6.0
    n_graphs_per_family: int = 300
    beta: float = 0.06            # per-edge, per-step transmission probability
    gamma: float = 0.2            # per-step recovery probability
    seed: int = 20260919

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# graphs
# --------------------------------------------------------------------------

def erdos_renyi(n: int, mean_degree: float, rng) -> np.ndarray:
    """Edge list (E, 2) of G(n, p) with p = <k> / (n - 1)."""
    p = mean_degree / (n - 1)
    iu = np.triu_indices(n, k=1)
    keep = rng.random(len(iu[0])) < p
    return np.stack([iu[0][keep], iu[1][keep]], 1)


def barabasi_albert(n: int, m: int, rng) -> np.ndarray:
    """Edge list of a BA graph: each new node attaches to m existing nodes
    with probability proportional to their degree. Mean degree -> 2m."""
    edges = []
    targets = list(range(m))                     # a small seed clique
    repeated = []                                # nodes, repeated by degree
    for i in range(m):
        for j in range(i + 1, m):
            edges.append((i, j)); repeated += [i, j]
    for new in range(m, n):
        chosen = set()
        while len(chosen) < m:
            chosen.add(repeated[rng.integers(len(repeated))]
                       if repeated else targets[rng.integers(len(targets))])
        for t in chosen:
            edges.append((t, new)); repeated += [t, new]
    return np.array(edges, dtype=np.int64)


def adjacency(edges: np.ndarray, n: int) -> sp.csr_matrix:
    A = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                      shape=(n, n))
    return (A + A.T).tocsr()


# --------------------------------------------------------------------------
# the epidemic
# --------------------------------------------------------------------------

def simulate(A: sp.csr_matrix, seed_node: int, beta: float, gamma: float,
             rng) -> tuple[np.ndarray, np.ndarray]:
    """Run to extinction. Returns (ever_infected: bool (n,), t_infected: int (n,)
    with -1 for never)."""
    n = A.shape[0]
    S = np.ones(n, bool); I = np.zeros(n, bool)
    S[seed_node] = False; I[seed_node] = True
    t_inf = np.full(n, -1); t_inf[seed_node] = 0
    t = 0
    while I.any():
        t += 1
        # number of infected neighbours of each node; the chance of escaping
        # all of them is (1 - beta)^count
        count = A @ I.astype(float)
        p_inf = 1.0 - (1.0 - beta) ** count
        new = S & (rng.random(n) < p_inf)
        recover = I & (rng.random(n) < gamma)
        I = (I & ~recover) | new
        S &= ~new
        t_inf[new] = t
    return t_inf >= 0, t_inf


# --------------------------------------------------------------------------
# the theory
# --------------------------------------------------------------------------

def transmissibility(beta: float, gamma: float) -> float:
    return 1.0 - gamma * (1 - beta) / (1.0 - (1 - gamma) * (1 - beta))


def r0(edges: np.ndarray, n: int, beta: float, gamma: float) -> float:
    k = np.bincount(edges.ravel(), minlength=n).astype(float)
    return transmissibility(beta, gamma) * ((k**2).mean() - k.mean()) / k.mean()


def check_threshold(params: SIRParams, rng, n_graphs: int = 60) -> list[dict]:
    """Sweep beta on Erdos-Renyi graphs and watch major outbreaks appear.

    A 'major' outbreak reaches at least 10% of the nodes. Below R0 = 1 these
    should be rare, above it common. The transition is smeared on a graph of
    300 nodes, so the check asserts the two ends and the ordering rather than
    a sharp crossing.
    """
    rows = []
    for beta in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.14, 0.20]:
        sizes, r0s = [], []
        for _ in range(n_graphs):
            e = erdos_renyi(params.n_nodes, params.mean_degree, rng)
            A = adjacency(e, params.n_nodes)
            inf, _ = simulate(A, rng.integers(params.n_nodes), beta, params.gamma, rng)
            sizes.append(inf.mean()); r0s.append(r0(e, params.n_nodes, beta, params.gamma))
        sizes = np.array(sizes)
        rows.append({"beta": beta, "R0": float(np.mean(r0s)),
                     "major_outbreak_share": float((sizes >= 0.10).mean()),
                     "mean_final_size": float(sizes.mean())})
    return rows


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------

def generate(params: SIRParams) -> dict:
    rng = np.random.default_rng(params.seed)
    n = params.n_nodes
    out = {}
    for family in ("erdos_renyi", "barabasi_albert"):
        edges_all, edge_graph = [], []
        node_rows = {k: [] for k in ("graph", "node", "degree", "is_seed",
                                     "infected", "t_infected")}
        graph_rows = []
        for g in range(params.n_graphs_per_family):
            if family == "erdos_renyi":
                e = erdos_renyi(n, params.mean_degree, rng)
            else:
                e = barabasi_albert(n, int(round(params.mean_degree / 2)), rng)
            A = adjacency(e, n)
            s = int(rng.integers(n))
            inf, t_inf = simulate(A, s, params.beta, params.gamma, rng)
            deg = np.bincount(e.ravel(), minlength=n)
            edges_all.append(e); edge_graph.append(np.full(len(e), g))
            node_rows["graph"].append(np.full(n, g)); node_rows["node"].append(np.arange(n))
            node_rows["degree"].append(deg); node_rows["is_seed"].append(np.arange(n) == s)
            node_rows["infected"].append(inf); node_rows["t_infected"].append(t_inf)
            graph_rows.append({"graph": g, "n_edges": len(e), "seed_node": s,
                               "final_size": float(inf.mean()),
                               "R0": r0(e, n, params.beta, params.gamma),
                               "mean_degree": float(deg.mean()),
                               "max_degree": int(deg.max())})
        out[family] = {
            "edges": np.concatenate(edges_all).astype(np.int32),
            "edge_graph": np.concatenate(edge_graph).astype(np.int32),
            "nodes": {k: np.concatenate(v) for k, v in node_rows.items()},
            "graphs": graph_rows,
        }
    out["threshold_check"] = check_threshold(params, rng)
    return out


def save(data: dict, params: SIRParams, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for family in ("erdos_renyi", "barabasi_albert"):
        d = data[family]
        np.savez_compressed(outdir / f"sir_{family}.npz",
                            edges=d["edges"], edge_graph=d["edge_graph"],
                            **{f"node_{k}": v for k, v in d["nodes"].items()},
                            **{f"graph_{k}": np.array([r[k] for r in d["graphs"]])
                               for k in d["graphs"][0]},
                            **params.as_dict())
    summary = {"params": params.as_dict(), "threshold_check": data["threshold_check"]}
    for family in ("erdos_renyi", "barabasi_albert"):
        gr = data[family]["graphs"]
        fs = np.array([r["final_size"] for r in gr])
        summary[family] = {
            "n_graphs": len(gr),
            "mean_R0": float(np.mean([r["R0"] for r in gr])),
            "mean_degree": float(np.mean([r["mean_degree"] for r in gr])),
            "max_degree_median": float(np.median([r["max_degree"] for r in gr])),
            "major_outbreak_share": float((fs >= 0.10).mean()),
            "mean_final_size_given_major": float(fs[fs >= 0.10].mean()) if (fs >= 0.10).any() else 0.0,
        }
    (outdir / "sir_params.json").write_text(json.dumps(summary, indent=2))


def plot(data: dict, params: SIRParams, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    for family, c in (("erdos_renyi", "tab:blue"), ("barabasi_albert", "tab:orange")):
        deg = data[family]["nodes"]["degree"]
        ax.hist(deg, bins=np.arange(0, 60), density=True, alpha=0.6, color=c,
                label=family.replace("_", "-"))
    ax.set_yscale("log"); ax.set_xlabel("degree"); ax.set_ylabel("density (log)")
    ax.set_title("same mean degree, different tails"); ax.legend()

    ax = axes[1]
    for family, c in (("erdos_renyi", "tab:blue"), ("barabasi_albert", "tab:orange")):
        fs = np.array([r["final_size"] for r in data[family]["graphs"]])
        ax.hist(fs, bins=30, alpha=0.6, color=c, label=family.replace("_", "-"))
    ax.set_xlabel("final size (share of nodes ever infected)"); ax.set_ylabel("graphs")
    ax.set_title(f"outcomes at beta={params.beta}, gamma={params.gamma}"); ax.legend()

    ax = axes[2]
    tc = data["threshold_check"]
    ax.plot([r["R0"] for r in tc], [r["major_outbreak_share"] for r in tc], "o-", color="tab:blue")
    ax.axvline(1.0, color="black", ls="--", lw=1)
    ax.text(1.04, 0.45, "R0 = 1\n(theory)", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 1, 2, 3], ["0.3", "0.5", "1", "2", "3"])
    ax.set_xlabel("R0 from the formula"); ax.set_ylabel("share of major outbreaks")
    ax.set_title("the threshold, on Erdos-Renyi graphs")
    fig.suptitle("SIR epidemics on random graphs", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    for f in fields(SIRParams):
        ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    params = SIRParams(**vars(ap.parse_args()))

    data = generate(params)
    save(data, params, OUTPUT_DIR)
    plot(data, params, OUTPUT_DIR / "sir_overview.png")

    summary = json.loads((OUTPUT_DIR / "sir_params.json").read_text())
    for family in ("erdos_renyi", "barabasi_albert"):
        s = summary[family]
        print(f"{family:<16} {s['n_graphs']} graphs, {params.n_nodes} nodes, "
              f"<k> {s['mean_degree']:.2f}, median max degree {s['max_degree_median']:.0f}, "
              f"R0 {s['mean_R0']:.2f}")
        print(f"                 major outbreaks (>=10%): {s['major_outbreak_share']:.0%}, "
              f"mean size given major {s['mean_final_size_given_major']:.0%}")
    print("\nthreshold check on Erdos-Renyi (theory says major outbreaks need R0 > 1):")
    for r in data["threshold_check"]:
        bar = "#" * int(40 * r["major_outbreak_share"])
        print(f"  beta {r['beta']:.2f}  R0 {r['R0']:5.2f}  major {r['major_outbreak_share']:4.0%}  {bar}")
    tc = data["threshold_check"]
    low = [r for r in tc if r["R0"] < 0.6]; high = [r for r in tc if r["R0"] > 2.0]
    ok = (all(r["major_outbreak_share"] < 0.05 for r in low)
          and all(r["major_outbreak_share"] > 0.30 for r in high))
    print(f"  {'PASS' if ok else 'FAIL'}: rare below R0 = 0.6, common above R0 = 2")
    print(f"-> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
