"""A System-1 decision engine running on the Apple Neural Engine.

Typed decisions over a fixed-shape sentence encoder. The engine never generates
text, so it cannot return a label outside the schema you gave it -- that is a
structural property, not a trained behaviour.

Decisions are "compiled" once (option descriptions embedded up front) and then
called per input, which is the only part that costs anything at runtime. `State`
goes one step further: embed an input once and answer every question about it
from that one vector.
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
        E = np.stack(out)
        # Normalise here rather than trusting the graph's final rsqrt. On
        # compute_units="CPU_ONLY" Core ML returns the *unnormalised* pooled
        # vector (norm ~19.6 instead of 1.0); CPU_AND_NE, CPU_AND_GPU and ALL
        # all normalise correctly. Everything downstream treats these as unit
        # vectors -- cosine, min_sim thresholds, anchor centroids -- so the
        # failure is silent and total. Cheap to make impossible.
        return E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)


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
        # An option is anchored by a description, or by several strings whose
        # centroid becomes the anchor. Examples beat descriptions decisively --
        # 16 real utterances per class cut CLINC errors 44% against one
        # handwritten sentence -- so the list form is the one worth using when
        # there is any labelled data at all. See FINDINGS.md.
        texts, spans = [], []
        for l in self.labels:
            v = options[l]
            v = [v] if isinstance(v, str) else list(v)
            if not v:
                raise ValueError(f"option {l!r} has no anchor text")
            spans.append((len(texts), len(texts) + len(v)))
            texts.extend(v)
        E = encoder.embed(texts)
        cen = np.stack([E[a:b].mean(0) for a, b in spans])
        self.anchors = cen / np.clip(
            np.linalg.norm(cen, axis=1, keepdims=True), 1e-12, None)
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
        return self.decide(self.enc.embed([text])[0], allowed=allowed)

    def decide(self, vec, allowed=None):
        """Same decision from an embedding that has already been computed.

        This is the whole runtime cost of a question: one (d, K) matmul over a
        vector someone else paid for. See `State`.
        """
        sims = vec @ self.anchors.T
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
        return self.decide(self.enc.embed([text])[0], allowed=allowed)

    def decide(self, vec, allowed=None):
        d = super().decide(vec, allowed=allowed)
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
        return self.decide(self.enc.embed([text])[0])

    def decide(self, vec):
        p = _softmax(vec @ self.a.T, self.temp)
        return {"score": float((np.arange(self.n) * p).sum()),
                "confidence": float(p.max()),
                "dist": [round(float(x), 3) for x in p]}


class State:
    """One input, embedded once, answered by any number of typed questions.

    Other decision engines engineer this: kev caches a state representation
    across questions (772 tokens, 861 ms -> 242 ms), Laya batches ten questions
    to reach ~7.2 ms each against 32.8 ms for one. In an embedding architecture
    it is not an optimisation, it is the shape of the thing -- N questions about
    one input is one encode plus N dot products, and the encode is ~1.4 ms
    against microseconds for the rest.

        s = State(enc, "I was charged twice for last month")
        s.ask(department=route, urgency=anger, churn_risk=at_risk)
        # {"department": Decision(label='billing', ...),
        #  "urgency":    {"score": 1.74, ...},
        #  "churn_risk": Decision(label=True, ...)}

    Questions must share the encoder this State was built with: two encoders
    produce two unrelated vector spaces, and comparing across them yields
    confident nonsense rather than an error. `ask` checks.

    For a masked question, call the primitive directly against the public
    vector: `route.decide(s.vec, allowed=["billing", "auth"])`.
    """

    def __init__(self, encoder, text):
        self.enc, self.text = encoder, text
        self.vec = encoder.embed([text])[0]

    def ask(self, **questions):
        for name, q in questions.items():
            if q.enc is not self.enc:
                raise ValueError(
                    f"question {name!r} was built with a different encoder; "
                    f"its anchors live in another vector space")
        return {name: q.decide(self.vec) for name, q in questions.items()}
