"""
generate.py -- Multi-query associative recall: a list of key-value pairs,
then every key again, and the model must answer each with its value.

The task
--------
    k1 v1  k2 v2  ...  kN vN   q1 q2 ... qN

Keys and values are drawn from two separate vocabularies of 256 tokens
each. The N keys in a sequence are distinct; each value is drawn
independently, so two keys may share one. The queries are the same N keys
in a fresh random order. At query position i the right answer is the value
that followed q_i in the first half. Nothing else is scored.

This is the multi-query associative recall task of Arora et al. (2023),
"Zoology", in its plainest form. It isolates one ability: carrying N
arbitrary bindings from where they were stated to where they are asked for.
The pairing is random in every sequence, so there is nothing to learn about
which key goes with which value - only how to remember what this sequence
said. A model that stores the sequence (attention keeps every token) can
look the answer up; a model that compresses it into a fixed-size state has
to fit N bindings, 8N bits of them, into that state.

Checks, on every generation
---------------------------
  1. a lookup table built from each sequence's own first half answers every
     query correctly - the task is well posed
  2. keys are distinct within a sequence and the queries are a permutation
     of them
  3. no shortcut: the best fixed key -> value table, fitted on 200,000 fresh
     sequences, scores chance on the test sets; so does the best fixed
     query-position -> value table
  4. values are uniform (chi-square)

Writes reference test sets, one per pair count. Training data is drawn on
the fly from `sample()` with a seed stream the test sets never use.
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
from scipy.stats import binomtest, chisquare  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "outputs"
IGNORE = -1                         # target at every position that is not a query


@dataclass(frozen=True)
class RecallParams:
    n_keys: int = 256
    n_values: int = 256
    pair_counts: tuple = (4, 8, 16, 32, 64)
    n_test: int = 1000              # test sequences per pair count
    n_shortcut: int = 200_000       # sequences used to fit the shortcut tables
    seed: int = 20261003

    def as_dict(self) -> dict:
        d = asdict(self)
        d["pair_counts"] = list(self.pair_counts)
        return d

    @property
    def vocab(self) -> int:
        return self.n_keys + self.n_values


def sample(rng: np.random.Generator, n: int, n_pairs: int,
           n_keys: int = 256, n_values: int = 256):
    """n sequences of n_pairs pairs. Returns tokens (n, 3 n_pairs) and
    targets of the same shape: the value token at query positions, IGNORE
    everywhere else. Keys are ids [0, n_keys), values [n_keys, n_keys +
    n_values)."""
    keys = np.argsort(rng.random((n, n_keys)), axis=1)[:, :n_pairs]      # distinct
    values = n_keys + rng.integers(0, n_values, size=(n, n_pairs))
    order = np.argsort(rng.random((n, n_pairs)), axis=1)                 # query order
    tokens = np.empty((n, 3 * n_pairs), dtype=np.int64)
    tokens[:, 0:2 * n_pairs:2] = keys
    tokens[:, 1:2 * n_pairs:2] = values
    tokens[:, 2 * n_pairs:] = np.take_along_axis(keys, order, axis=1)
    targets = np.full_like(tokens, IGNORE)
    targets[:, 2 * n_pairs:] = np.take_along_axis(values, order, axis=1)
    return tokens, targets


def test_stream(seed: int, n_pairs: int) -> np.random.Generator:
    """The test sets' seed stream. Training streams use a different second
    word, so the two can never coincide."""
    return np.random.default_rng([seed, 0, n_pairs])


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def oracle(tokens: np.ndarray, n_pairs: int) -> np.ndarray:
    """Answer each query by looking it up in the sequence's own first half."""
    pred = np.full_like(tokens, IGNORE)
    for i in range(len(tokens)):
        table = dict(zip(tokens[i, 0:2 * n_pairs:2], tokens[i, 1:2 * n_pairs:2]))
        pred[i, 2 * n_pairs:] = [table[q] for q in tokens[i, 2 * n_pairs:]]
    return pred


def check(p: RecallParams, tests: dict) -> dict:
    out = {"oracle_accuracy": {}, "distinct_keys": True, "queries_are_permutation": True}
    for n, (tok, tgt) in tests.items():
        q = tgt != IGNORE
        out["oracle_accuracy"][n] = float((oracle(tok, n)[q] == tgt[q]).mean())
        keys = tok[:, 0:2 * n:2]
        out["distinct_keys"] &= bool(all(len(set(r)) == n for r in keys))
        out["queries_are_permutation"] &= bool(
            (np.sort(keys, 1) == np.sort(tok[:, 2 * n:], 1)).all())

    # shortcut tables, fitted on fresh sequences from a third stream
    rng = np.random.default_rng([p.seed, 2])
    key_counts = np.zeros((p.n_keys, p.n_values), dtype=np.int64)
    pos_counts = {n: np.zeros((n, p.n_values), dtype=np.int64) for n in p.pair_counts}
    per = p.n_shortcut // len(p.pair_counts)
    for n in p.pair_counts:
        tok, tgt = sample(rng, per, n, p.n_keys, p.n_values)
        qk, qv = tok[:, 2 * n:], tgt[:, 2 * n:] - p.n_keys
        np.add.at(key_counts, (qk.ravel(), qv.ravel()), 1)
        np.add.at(pos_counts[n], (np.tile(np.arange(n), per), qv.ravel()), 1)
    best_by_key = key_counts.argmax(1) + p.n_keys
    hits_k = hits_p = total = 0
    value_hist = np.zeros(p.n_values, dtype=np.int64)
    for n, (tok, tgt) in tests.items():
        qk, qv = tok[:, 2 * n:], tgt[:, 2 * n:]
        hits_k += int((best_by_key[qk] == qv).sum())
        hits_p += int((pos_counts[n].argmax(1)[None, :] + p.n_keys == qv).sum())
        total += qv.size
        value_hist += np.bincount((qv - p.n_keys).ravel(), minlength=p.n_values)
    chance = 1 / p.n_values
    for name, hits in (("key_table", hits_k), ("position_table", hits_p)):
        bt = binomtest(hits, total, chance)
        out[f"shortcut_{name}_accuracy"] = hits / total
        out[f"shortcut_{name}_p_value"] = float(bt.pvalue)
    out["chance"] = chance
    out["n_queries_tested"] = total
    out["values_chi2_p_value"] = float(chisquare(value_hist).pvalue)
    return out


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------

def generate(p: RecallParams) -> dict:
    tests = {n: sample(test_stream(p.seed, n), p.n_test, n, p.n_keys, p.n_values)
             for n in p.pair_counts}
    return {"tests": tests, "checks": check(p, tests)}


def save(data: dict, p: RecallParams, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for n, (tok, tgt) in data["tests"].items():
        arrays[f"tokens_{n}"] = tok.astype(np.int16)
        arrays[f"targets_{n}"] = tgt.astype(np.int16)
    np.savez_compressed(outdir / "recall_test.npz", pair_counts=np.array(p.pair_counts),
                        **arrays)
    c = data["checks"]
    meta = {"params": p.as_dict(), "vocab": p.vocab,
            "bits_to_carry": {str(n): n * float(np.log2(p.n_values)) for n in p.pair_counts},
            "checks": {**c, "oracle_accuracy": {str(k): v for k, v in c["oracle_accuracy"].items()}}}
    (outdir / "recall_params.json").write_text(json.dumps(meta, indent=2))


def plot(data: dict, p: RecallParams, path: Path) -> None:
    n = 4
    tok, tgt = data["tests"][n]
    row, trow = tok[0], tgt[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    for i, t in enumerate(row):
        is_key = t < p.n_keys
        query = i >= 2 * n
        colour = "#e69f00" if query else ("#0284C7" if is_key else "#8c8c8c")
        ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, color=colour, alpha=0.85))
        label = f"k{t}" if is_key else f"v{t - p.n_keys}"
        ax.text(i + 0.46, 0.5, label, ha="center", va="center", color="white", fontsize=9)
        if trow[i] != IGNORE:
            ax.text(i + 0.46, -0.35, f"-> v{trow[i] - p.n_keys}", ha="center", va="center",
                    fontsize=9, color="#e69f00")
    ax.text(n - 0.04, 1.25, "pairs", ha="center", fontsize=10)
    ax.text(2 * n + n / 2 - 0.04, 1.25, "queries", ha="center", fontsize=10)
    ax.set_xlim(-0.2, 3 * n); ax.set_ylim(-0.7, 1.5); ax.axis("off")
    ax.set_title(f"one sequence with {n} pairs; the answer is scored at the queries only")

    ax = axes[1]
    counts = list(p.pair_counts)
    bits = [k * np.log2(p.n_values) for k in counts]
    ax.bar([str(k) for k in counts], bits, color="#0284C7", alpha=0.85)
    ax.set_xlabel("pairs in the sequence"); ax.set_ylabel("bits that must be carried")
    ax.set_title("what a fixed-size state has to hold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.suptitle("Multi-query associative recall", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    for f in fields(RecallParams):
        if f.name == "pair_counts":
            ap.add_argument("--pair_counts", type=int, nargs="+", default=list(f.default))
        else:
            ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    args = vars(ap.parse_args())
    args["pair_counts"] = tuple(args["pair_counts"])
    p = RecallParams(**args)

    data = generate(p)
    save(data, p, OUTPUT_DIR)
    plot(data, p, OUTPUT_DIR / "recall_overview.png")

    c = data["checks"]
    print(f"{p.n_test} test sequences at each of {list(p.pair_counts)} pairs; "
          f"{p.n_keys} keys, {p.n_values} values")
    print("  oracle (look-up in the sequence's own first half): "
          + ", ".join(f"{k} pairs {v:.3f}" for k, v in c["oracle_accuracy"].items()))
    print(f"  keys distinct within every sequence: {c['distinct_keys']}; "
          f"queries a permutation of them: {c['queries_are_permutation']}")
    print(f"  shortcuts, fitted on {p.n_shortcut:,} fresh sequences, scored on "
          f"{c['n_queries_tested']:,} test queries (chance {c['chance']:.4f}):")
    print(f"    best fixed key -> value table        {c['shortcut_key_table_accuracy']:.4f}"
          f"   (binomial p vs chance {c['shortcut_key_table_p_value']:.2f})")
    print(f"    best fixed position -> value table   {c['shortcut_position_table_accuracy']:.4f}"
          f"   (binomial p vs chance {c['shortcut_position_table_p_value']:.2f})")
    print(f"  values uniform: chi-square p = {c['values_chi2_p_value']:.2f}")
    print(f"-> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
