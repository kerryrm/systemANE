"""A System-1 decision engine running on the Apple Neural Engine.

Typed decisions over a fixed-shape sentence encoder. The engine never generates
text, so it cannot return a label outside the schema you gave it -- that is a
structural property, not a trained behaviour.

Decisions are "compiled" once (option descriptions embedded up front) and then
called per input, which is the only part that costs anything at runtime.
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np

SEQ = 128

# Fallbacks for a schema with no fitted calibration. min_sim is deliberately
# permissive: without knowing where this schema's in-scope similarities sit, a
# high threshold refuses real traffic. On CLINC 0.20 still rejects 81% of
# out-of-scope while keeping 99.3% of in-scope. temp and min_margin are the
# values calibrate.py fitted on CLINC -- a better-informed guess than a round
# number, but still a guess for any other schema. Fit properly with
# calibrate.py.
FALLBACK = {"temp": 0.05, "min_sim": 0.20, "min_margin": 0.40}


def load_calibration(path="calibration.json"):
    """Knobs fitted on a validation split by calibrate.py.

    Calibration is per-schema. A file fitted for one label set does not
    transfer to another: the CLINC banking min_sim (0.3085) refuses genuinely
    in-scope support tickets at 0.27-0.28. Pass calibration=None to Choice()
    for a schema you have not fitted, and treat the fallbacks as placeholders.
    """
    if path and os.path.exists(path):
        with open(path) as fh:
            return {**FALLBACK, **json.load(fh)}
    return dict(FALLBACK)


@dataclass
class Decision:
    label: str
    prob: float
    margin: float          # p(top) - p(runner-up): ambiguity *within* the schema
    sim: float             # raw max cosine: whether the input is in scope at all
    escalate: bool
    reason: str = ""       # "out-of-scope" | "structural" | "ambiguous" | ""
    all: dict = field(default_factory=dict)


class Encoder:
    """MiniLM-L6 as a fixed-shape Core ML model. ~1.2 ms/call, 95% on ANE."""

    def __init__(self, pkg="encoder.mlpackage", tokdir="tok", compute_units="CPU_AND_NE"):
        import coremltools as ct
        from transformers import AutoTokenizer
        self.m = ct.models.MLModel(pkg, compute_units=getattr(ct.ComputeUnit, compute_units))
        self.tok = AutoTokenizer.from_pretrained(tokdir)

    def embed(self, texts):
        out = []
        for t in texts:
            e = self.tok([t], padding="max_length", max_length=SEQ,
                         truncation=True, return_tensors="np")
            am = e["attention_mask"].astype(np.float32)
            r = self.m.predict({
                "input_ids": e["input_ids"].astype(np.int32),
                "attn_bias": ((1 - am)[:, None, None, :] * -1e4).astype(np.float32),
                "pool_mask": am[:, :, None].astype(np.float32)})
            out.append(np.array(r["embedding"]).ravel())
        return np.stack(out)


def _softmax(sims, temp):
    p = np.exp((sims - sims.max()) / temp)
    return p / p.sum()


class Choice:
    """Pick one label from a fixed set, with a calibrated escalate flag.

    Two independent escalation signals, because they catch different failures:

      margin  - relative, post-softmax. "Which of these labels?" Detects an
                input that sits between two categories.
      sim     - absolute max cosine, pre-softmax. "Any of these labels?"
                Detects an input outside the schema entirely.

    The softmax normalises magnitude away, so margin alone cannot see the
    second case: "I'd like to speak to a manager" scores a *higher* margin
    (0.113) than the genuinely ambiguous "it broke" (0.043), while its max
    cosine is 0.169 against 0.273. See README ("Two signals").

    temp and min_sim default to the values calibrate.py fitted on the CLINC150
    validation split (calibration.json), not to anything read off a fixture
    sweep -- that sweep was wrong in both magnitude and direction.

    `always_escalate` names labels that go to tier 2 regardless of confidence,
    for decisions that structurally need a bigger model rather than a more
    certain one -- jev-ultrafast's rule that only TYPE_TEXT needs an LLM is the
    clean example. Confidence cannot express that, and a calibrated threshold
    will sometimes decide not to escalate one.
    """

    def __init__(self, encoder, options, temp=None, min_margin=None,
                 min_prob=0.55, min_sim=None, calibration="calibration.json",
                 always_escalate=()):
        cal = load_calibration(calibration)
        temp = cal["temp"] if temp is None else temp
        min_sim = cal["min_sim"] if min_sim is None else min_sim
        min_margin = cal["min_margin"] if min_margin is None else min_margin
        self.enc, self.labels = encoder, list(options)
        self.anchors = encoder.embed([options[l] for l in self.labels])
        self.temp, self.min_margin = temp, min_margin
        self.min_prob, self.min_sim = min_prob, min_sim
        self.always_escalate = set(always_escalate)
        unknown = self.always_escalate - set(self.labels)
        if unknown:
            raise ValueError(f"always_escalate names unknown labels: {sorted(unknown)}")

    def __call__(self, text, allowed=None):
        """`allowed` restricts the decision to a subset of labels.

        Masking happens before the softmax, so probabilities renormalise over
        the legal set. Note that margins widen as the set shrinks, so a
        min_margin fitted on the full schema is conservative under masking.
        """
        sims = self.enc.embed([text])[0] @ self.anchors.T
        labels = self.labels
        if allowed is not None:
            labels = list(allowed)
            if not labels:
                raise ValueError("allowed must name at least one label")
            unknown = set(labels) - set(self.labels)
            if unknown:
                raise ValueError(f"allowed names unknown labels: {sorted(unknown)}")
            sims = sims[[self.labels.index(l) for l in labels]]
        p = _softmax(sims, self.temp)
        o = np.argsort(-p)
        margin = float(p[o[0]] - p[o[1]]) if len(p) > 1 else 1.0
        top_sim = float(sims.max())
        label = labels[o[0]]
        # Order matters: if nothing in the schema fits, "which label" is moot.
        if top_sim < self.min_sim:
            reason = "out-of-scope"
        elif label in self.always_escalate:
            reason = "structural"
        elif margin < self.min_margin or p[o[0]] < self.min_prob:
            reason = "ambiguous"
        else:
            reason = ""
        return Decision(
            label=label, prob=float(p[o[0]]), margin=margin,
            sim=top_sim, escalate=bool(reason), reason=reason,
            all={labels[i]: round(float(p[i]), 3) for i in o})


class Boolean(Choice):
    def __init__(self, encoder, yes, no, **kw):
        super().__init__(encoder, {"yes": yes, "no": no}, **kw)

    def __call__(self, text, allowed=None):
        d = super().__call__(text, allowed=allowed)
        d.label = d.label == "yes"
        return d


class Score:
    """Ordinal rubric. NOTE: weak -- embeddings capture topic, not intensity.

    On a 0-3 anger rubric every input collapses into a 1.25-1.74 band, and the
    two extremes are the two it gets most wrong. `confidence` is 0.31-0.32 on
    exactly those, against 0.64-0.72 where it is closest, so the uncertainty
    signal survives even where the estimate does not. Treat this as a detector
    of "needs a real model", not as a measurement."""

    def __init__(self, encoder, anchors, temp=0.10):
        self.enc, self.n, self.temp = encoder, len(anchors), temp
        self.a = encoder.embed(anchors)

    def __call__(self, text):
        p = _softmax(self.enc.embed([text])[0] @ self.a.T, self.temp)
        return {"score": float((np.arange(self.n) * p).sum()),
                "confidence": float(p.max()),
                "dist": [round(float(x), 3) for x in p]}
