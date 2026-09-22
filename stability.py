#!/usr/bin/env python3
"""Is the same question on the same input answered the same way twice?

TypeSafe's self-consistency cookbook runs one moderation rubric on one post 15
times, varying only a throwaway `uid` field, and reports Jev agreeing with
itself 90.8% of the time -- 99.2% if you refuse to act below p=0.60, which
costs 25.8% of traffic to human review. kev ships a /v1/systemone/permute
endpoint for the same reason.

There is no sampling anywhere in this engine, so the answer should be "yes,
exactly". That is worth proving rather than asserting, and the second half of
their experiment is not flattering to us: their variation comes from an
*irrelevant field* changing the answer, and mean-pooling 128 tokens is a very
direct way to be sensitive to irrelevant text.

  ./stability.py            # all three: determinism, uid, dilution
"""
import argparse
import json
import random

import numpy as np

import evalset
from evalset import OOS
from system1 import Choice, Encoder


def _decide(q, texts, enc):
    E = enc.embed(texts)
    return [q.decide(v) for v in E]


def determinism(enc, q, texts, runs):
    """Same input, repeatedly. Bit-identical, or not."""
    print(f"--- determinism: {len(texts)} inputs x {runs} runs ---")
    base = enc.embed(texts)
    exact = drift = 0
    for _ in range(runs - 1):
        E = enc.embed(texts)
        exact += int(np.array_equal(E, base))
        drift = max(drift, float(np.abs(E - base).max()))
    print(f"  embeddings bit-identical: {exact}/{runs - 1} repeat runs")
    print(f"  max |delta| across runs:  {drift:.3g}")
    d0 = [q.decide(v) for v in base]
    same = all(q.decide(v).label == d.label for v, d in zip(base, d0))
    print(f"  decisions identical:      {same}")
    # A second MLModel over the same package is the deployment-relevant case:
    # a restarted server must agree with the one it replaced.
    e2 = Encoder()
    E2 = e2.embed(texts)
    print(f"  fresh Encoder, same package: bit-identical "
          f"{np.array_equal(E2, base)}, max |delta| {float(np.abs(E2 - base).max()):.3g}")
    print()


def uid_test(enc, q, texts, runs, seed=0):
    """Their experiment: same state, a fresh irrelevant `uid` on every call."""
    print(f"--- uid distractor: {len(texts)} inputs x {runs} calls, "
          f"fresh uid each ---")
    rng = random.Random(seed)
    agree_n = agree_d = 0
    spreads, flipped = [], 0
    for t in texts:
        labels, probs = [], []
        for _ in range(runs):
            state = {"post": t, "uid": f"{rng.getrandbits(64):016x}"}
            d = q.decide(enc.embed([json.dumps(state, sort_keys=True)])[0])
            labels.append(d.label)
            probs.append(d.prob)
        top = max(set(labels), key=labels.count)
        agree_n += labels.count(top)
        agree_d += runs
        spreads.append(max(probs) - min(probs))
        flipped += len(set(labels)) > 1
    print(f"  self-agreement:        {agree_n / agree_d:.1%}   "
          f"(TypeSafe report 90.8% for Jev)")
    print(f"  inputs that ever flip: {flipped}/{len(texts)}")
    print(f"  p(top) spread:         mean {np.mean(spreads):.4f}  "
          f"max {np.max(spreads):.4f}")
    # The honest control: a uid moves the embedding, so this is not
    # determinism -- it is sensitivity to irrelevant text, measured separately.
    clean = [q.decide(v).label for v in enc.embed(texts)]
    wrapped = [q.decide(enc.embed([json.dumps({"post": t, "uid": "0" * 16},
                                              sort_keys=True)])[0]).label
               for t in texts]
    kept = sum(a == b for a, b in zip(clean, wrapped))
    print(f"  bare text vs JSON-wrapped with a fixed uid: {kept}/{len(texts)} "
          f"unchanged")
    print()


FILLER = ("The office will be closed on Monday for the public holiday. "
          "Parking is available at the rear of the building. "
          "Our newsletter goes out on the first Tuesday of every month. "
          "The cafeteria now accepts contactless payment. ")


def dilution(enc, q, texts, steps=(0, 1, 2, 3, 6)):
    """Jaggedness #5: large irrelevant state as a distractor.

    Mean pooling averages every unmasked token, so irrelevant text does not
    merely distract the way it distracts a prompted model -- it is
    arithmetically mixed into the vector in proportion to its length. This
    should be our worst result in any comparison, not our best.

    `signal` is the share of the pooled window the real input occupies. It
    stops falling once the sequence hits SEQ=128 and the rest is truncated,
    which is also why the last rows stop changing: they are the same 128
    tokens.
    """
    print(f"--- dilution: {len(texts)} inputs, irrelevant text appended ---")
    base = [q.decide(v) for v in enc.embed(texts)]
    bare = [len(enc.tok(t)["input_ids"]) for t in texts]
    print(f"  {'filler':>7} {'tokens':>8} {'trunc':>6} {'signal':>7} "
          f"{'same label':>11} {'mean sim':>9} {'mean p':>8}")
    for n in steps:
        padded = [f"{t} {FILLER * n}".strip() for t in texts]
        n_tok = [len(enc.tok(s)["input_ids"]) for s in padded]
        real = [min(x, 128) for x in n_tok]
        E = enc.embed(padded)
        ds = [q.decide(v) for v in E]
        same = sum(a.label == b.label for a, b in zip(ds, base))
        print(f"  {n:6d}x {np.mean(n_tok):8.0f} "
              f"{sum(x > 128 for x in n_tok):5d}  "
              f"{np.mean([b / r for b, r in zip(bare, real)]):6.1%} "
              f"{same:6d}/{len(texts)} {np.mean([d.sim for d in ds]):9.3f} "
              f"{np.mean([d.prob for d in ds]):8.3f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="inputs to test")
    ap.add_argument("--runs", type=int, default=15, help="repeats per input")
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()

    import anchors as A
    enc = Encoder()
    rows = [r for r in evalset.load("test") if r[1] != OOS][:args.n]
    texts = [t for t, _ in rows]
    an = A.build(enc, k=args.k, mode="centroid", use_description=False)
    q = Choice(enc, {l: l for l in an.labels})   # anchors replaced below
    q.anchors, q.labels = an.M, an.labels
    print(f"CLINC test, {len(texts)} in-scope inputs, k={args.k} centroids, "
          f"T={q.temp}\n")
    determinism(enc, q, texts[:20], args.runs)
    uid_test(enc, q, texts[:40], args.runs)
    dilution(enc, q, texts)


if __name__ == "__main__":
    main()
