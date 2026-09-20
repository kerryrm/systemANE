"""Class representations: a description, k example utterances, or both.

The zero-shot baseline embeds one handwritten description per class. Examples
are strictly more information -- this module exists to measure how much more,
before reaching for fine-tuning.

Examples are drawn from the *train* split only. Selection happens on
validation. Test is never involved.
"""
import numpy as np

import evalset
from evalset import OOS, ROUTES


class Anchors:
    """Each class is a set of vectors; scoring reduces over that set."""

    def __init__(self, labels, mats, mode="centroid"):
        self.labels, self.mode = labels, mode
        if mode == "centroid":
            cen = np.stack([m.mean(0) for m in mats])
            self.M = cen / np.linalg.norm(cen, axis=1, keepdims=True)
        elif mode == "max":
            self.mats = mats
        else:
            raise ValueError(mode)

    def score(self, E):
        """(n, d) embeddings -> (n, n_classes) similarities."""
        if self.mode == "centroid":
            return E @ self.M.T
        return np.stack([(E @ m.T).max(1) for m in self.mats], axis=1)


def _train_examples(k, seed=0):
    rows = evalset.load("train", include_oos=False)
    by = {}
    for text, label in rows:
        by.setdefault(label, []).append(text)
    rng = np.random.default_rng(seed)
    out = {}
    for label, texts in by.items():
        idx = rng.permutation(len(texts))[:k]
        out[label] = [texts[i] for i in idx]
    return out


def build(enc, k=0, mode="centroid", use_description=True, seed=0, cache=None):
    """Anchors for ROUTES. k=0 with use_description reproduces the baseline."""
    labels = list(ROUTES)
    ex = _train_examples(k, seed) if k else {l: [] for l in labels}
    mats = []
    for l in labels:
        vecs = []
        if use_description:
            vecs.append(cache[ROUTES[l]] if cache else enc.embed([ROUTES[l]])[0])
        for t in ex[l]:
            vecs.append(cache[t] if cache else enc.embed([t])[0])
        if not vecs:
            raise ValueError("a class needs at least a description or one example")
        mats.append(np.stack(vecs))
    return Anchors(labels, mats, mode)


def texts_needed(k_max, seed=0):
    """Every string build() could need, so callers can embed once."""
    ex = _train_examples(k_max, seed)
    return list(ROUTES.values()) + [t for v in ex.values() for t in v]
