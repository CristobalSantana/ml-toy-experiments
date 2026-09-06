"""
hdc.py -- Hyperdimensional computing, written out.

A classifier with no gradients, no epochs and no loss function. Every feature
and every quantised value gets a random vector in a space so wide that any two
random vectors are almost exactly orthogonal, and the arithmetic of those
vectors does the work:

    bind    elementwise product. Ties a feature to the value it took.
            The product is dissimilar to both of its operands, which is what
            makes it a "this feature has that value" token rather than a blur.
    bundle  sum, then take the sign. Superposes many tokens into one vector
            that stays similar to each of them.

A record is the bundle of its bound feature-value pairs. A class is the bundle
of its records. Prediction is whichever class vector is closest.

That is the whole algorithm. Training is one pass and consists of addition.

Why the level vectors are built by flipping bits
------------------------------------------------
If each quantisation level got an independent random vector, a fare of $20.00
and a fare of $20.01 would land in unrelated corners of the space, and the
encoding would throw away the ordering that makes a continuous feature useful.
Level 0 is drawn at random and each subsequent level flips a fixed share of
the remaining bits, so neighbouring levels stay similar and the two ends come
out orthogonal. The ordering is in the geometry rather than in a comment.
"""

from __future__ import annotations

import numpy as np


class HDClassifier:
    """Record-based hyperdimensional classifier.

    Parameters
    ----------
    dim : width of the hypervectors. The whole method rests on random vectors
        in high dimension being near-orthogonal, which is a property of the
        dimension rather than of the data; below a few thousand it degrades.
    levels : quantisation levels per feature.
    retrain_epochs : passes of the optional corrective step. Zero is the pure
        one-shot claim - bundle everything once and stop. Anything above zero
        is iterative learning wearing a different hat, and is reported
        separately rather than folded into "HDC".
    """

    def __init__(self, dim: int = 10000, levels: int = 64,
                 retrain_epochs: int = 0, seed: int = 0):
        self.dim = dim
        self.levels = levels
        self.retrain_epochs = retrain_epochs
        self.rng = np.random.default_rng(seed)
        self.position_ = None      # one hypervector per feature
        self.level_ = None         # levels x dim, correlated along the rows
        self.edges_ = None         # quantiser boundaries, fitted on train only
        self.classes_ = None
        self.prototypes_ = None

    # ---------------------------------------------------------------- setup
    def _make_levels(self) -> np.ndarray:
        """Level 0 random; each next level flips dim/(2*levels) more bits.

        After the last level, about half the bits have flipped, so the two
        extremes are orthogonal while neighbours are nearly identical.
        """
        base = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=self.dim)
        out = np.empty((self.levels, self.dim), dtype=np.int8)
        out[0] = base
        flip_per_step = max(1, self.dim // (2 * (self.levels - 1)))
        order = self.rng.permutation(self.dim)
        cur = base.copy()
        for i in range(1, self.levels):
            idx = order[(i - 1) * flip_per_step: i * flip_per_step]
            cur = cur.copy()
            cur[idx] *= -1
            out[i] = cur
        return out

    def _quantise(self, X: np.ndarray) -> np.ndarray:
        """Values to level indices, using quantiles fitted on the training set.

        Quantiles rather than equal-width bins: fare amount and trip distance
        are both heavily right-skewed, and equal-width bins would put 95% of
        the data in the first two.
        """
        q = np.empty(X.shape, dtype=np.int32)
        for j in range(X.shape[1]):
            q[:, j] = np.clip(np.searchsorted(self.edges_[j], X[:, j]), 0,
                              self.levels - 1)
        return q

    # Materialising every row's hypervector at once is not an option: 200,000
    # test rows at dim 10,000 is 8 GB of int32. Everything below works in
    # chunks and keeps only what it is accumulating.
    BATCH = 4096

    def encode(self, X: np.ndarray) -> np.ndarray:
        """One hypervector per row. Only call this on a chunk you can hold."""
        q = self._quantise(X)
        # int16, not int32: every term is +-1 and there is one per feature, so
        # the sum cannot leave [-n_features, n_features]. The accumulator is
        # the whole cost of this method, and halving its width halves the
        # memory traffic.
        acc = np.zeros((len(X), self.dim), dtype=np.int16)
        for j in range(X.shape[1]):
            acc += self.position_[j] * self.level_[q[:, j]]
        return acc

    def _batches(self, X: np.ndarray):
        for i in range(0, len(X), self.BATCH):
            yield i, self.encode(X[i:i + self.BATCH])

    # ----------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HDClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n_feat = X.shape[1]

        qs = np.linspace(0, 100, self.levels + 1)[1:-1]
        self.edges_ = [np.percentile(X[:, j], qs) for j in range(n_feat)]
        self.position_ = self.rng.choice(np.array([-1, 1], dtype=np.int8),
                                         size=(n_feat, self.dim))
        self.level_ = self._make_levels()

        self.classes_ = np.unique(y)
        # accumulated chunk by chunk; the prototypes are sums, so they can be
        # built without ever holding the whole encoded training set
        self.prototypes_ = np.zeros((len(self.classes_), self.dim), dtype=np.float64)
        for i, enc in self._batches(X):
            yb = y[i:i + self.BATCH]
            for ci, c in enumerate(self.classes_):
                sel = yb == c
                if sel.any():
                    self.prototypes_[ci] += enc[sel].sum(axis=0)

        # Optional corrective passes. Off by default: the interesting claim is
        # what one pass of addition buys, and an iterative variant should be
        # compared against iterative methods, not presented as one-shot.
        for _ in range(self.retrain_epochs):
            changed = False
            for i, enc in self._batches(X):
                yb = y[i:i + self.BATCH]
                pred = self._predict_encoded(enc)
                for k in np.flatnonzero(pred != yb):
                    ci = int(np.searchsorted(self.classes_, yb[k]))
                    pi = int(np.searchsorted(self.classes_, pred[k]))
                    self.prototypes_[ci] += enc[k]
                    self.prototypes_[pi] -= enc[k]
                    changed = True
            if not changed:
                break
        return self

    # ------------------------------------------------------------- predict
    def _predict_encoded(self, enc: np.ndarray, mask: np.ndarray | None = None
                         ) -> np.ndarray:
        proto = self.prototypes_ if mask is None else self.prototypes_ * mask
        e = enc if mask is None else enc * mask
        # cosine similarity: the prototypes have wildly different norms because
        # the classes are imbalanced, and an unnormalised dot product would
        # simply return the majority class every time
        pn = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-12)
        en = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        return self.classes_[np.argmax(en @ pn.T, axis=1)]

    def predict(self, X: np.ndarray, dim_mask: np.ndarray | None = None
                ) -> np.ndarray:
        """Predict, optionally with part of the hypervector switched off.

        `dim_mask` is how the robustness claim gets tested: zeroing a share of
        the dimensions is the memory-corruption failure the method is supposed
        to shrug off.
        """
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(len(X), dtype=self.classes_.dtype)
        for i, enc in self._batches(X):
            out[i:i + self.BATCH] = self._predict_encoded(enc, dim_mask)
        return out

    def predict_and_scores(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Both, from one pass of encoding.

        Calling predict() and decision_scores() separately encodes the test
        set twice, and the encoding is the entire cost of this classifier.
        """
        X = np.asarray(X, dtype=np.float64)
        pn = self.prototypes_ / (np.linalg.norm(self.prototypes_, axis=1,
                                                keepdims=True) + 1e-12)
        pred = np.empty(len(X), dtype=self.classes_.dtype)
        score = np.empty(len(X), dtype=np.float64)
        for i, enc in self._batches(X):
            en = enc / (np.linalg.norm(enc, axis=1, keepdims=True) + 1e-12)
            sim = en @ pn.T
            pred[i:i + self.BATCH] = self.classes_[np.argmax(sim, axis=1)]
            score[i:i + self.BATCH] = sim[:, 1] - sim[:, 0]
        return pred, score

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """Similarity margin between the two classes, for an ROC curve."""
        X = np.asarray(X, dtype=np.float64)
        pn = self.prototypes_ / (np.linalg.norm(self.prototypes_, axis=1,
                                                keepdims=True) + 1e-12)
        out = np.empty(len(X), dtype=np.float64)
        for i, enc in self._batches(X):
            en = enc / (np.linalg.norm(enc, axis=1, keepdims=True) + 1e-12)
            sim = en @ pn.T
            out[i:i + self.BATCH] = sim[:, 1] - sim[:, 0]
        return out

    @property
    def n_parameters(self) -> int:
        """What has to be kept after training: the codebook and the prototypes."""
        return (self.position_.size + self.level_.size + self.prototypes_.size)
