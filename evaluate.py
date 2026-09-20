#!/usr/bin/env python3
"""Measure the tier-1 engine on CLINC150. No tuning happens here.

Reports what a cascade actually cares about:
  * in-scope accuracy            -- is the answer right when it answers?
  * ECE                          -- does its confidence mean anything?
  * OOS AUROC                    -- can it tell "not in this schema" at all,
                                    and which signal does it best with?
  * risk/coverage                -- error rate among the inputs it keeps,
                                    as a function of how many it escalates
"""
import argparse

import numpy as np

import anchors as A
import evalset
from evalset import OOS, ROUTES
from system1 import Encoder, _softmax, load_calibration


def _encoder(name):
    """<name>.mlpackage plus <name>_tok/ if present, else the shared tok/."""
    import os
    tok = f"{name}_tok" if os.path.isdir(f"{name}_tok") else "tok"
    return Encoder(f"{name}.mlpackage", tok)


def auroc(scores, pos):
    """Rank-based AUROC; pos is a boolean mask of the positive class."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total, n = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            total += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--max-oos", type=int, default=None)
    ap.add_argument("--k", type=int, default=None, help="train examples per class")
    ap.add_argument("--encoder", default="encoder",
                    help="basename: <name>.mlpackage + <name>_tok/ (or tok/)")
    ap.add_argument("--calibration", default="calibration.json")
    args = ap.parse_args()
    cal = load_calibration(args.calibration)
    if args.temp is None:
        args.temp = cal["temp"]
    k = cal.get("k", 0) if args.k is None else args.k
    mode, use_desc = cal.get("mode", "centroid"), cal.get("use_description", True)

    rows = evalset.load(args.split, max_oos=args.max_oos)
    print(f"{args.split}: {evalset.summary(rows)}\n")

    enc = _encoder(args.encoder)
    an = A.build(enc, k=k, mode=mode, use_description=use_desc)
    labels = an.labels
    print(f"anchors: k={k} mode={mode} description={use_desc}\n")
    texts = [t for t, _ in rows]
    truth = np.array([l for _, l in rows])
    E = enc.embed(texts)

    sims = an.score(E)
    pred = np.array([labels[i] for i in sims.argmax(1)])
    P = np.stack([_softmax(s, args.temp) for s in sims])
    top = P.max(1)
    order = np.sort(P, axis=1)
    margin = order[:, -1] - order[:, -2]
    maxsim = sims.max(1)

    ins = truth != OOS
    correct = (pred == truth) & ins

    print(f"in-scope accuracy      {correct[ins].mean():.3f}  ({correct[ins].sum()}/{ins.sum()})")
    print(f"ECE (in-scope, T={args.temp})  {ece(top[ins], correct[ins].astype(float)):.3f}")
    print(f"mean confidence        {top[ins].mean():.3f}  vs accuracy {correct[ins].mean():.3f}")

    print("\nout-of-scope detection (AUROC, higher = separates better)")
    for name, s in (("max cosine (sim)", maxsim), ("margin", margin), ("top prob", top)):
        print(f"  {name:18s} {auroc(s, ins):.3f}")

    # Sweep by in-scope retention, not by absolute threshold: different
    # encoders put their similarities in completely different ranges (MiniLM
    # fits min_sim 0.31, BGE 0.75), so a fixed 0.10-0.40 grid is meaningless
    # across encoders and reported 0% rejected for BGE.
    print("\noperating curve — threshold set to retain a share of in-scope")
    print(f"  {'keep in-scope':>13} {'threshold':>10} {'oos rejected':>13} {'acc on kept':>12}")
    for keep in (1.00, 0.99, 0.98, 0.95, 0.90):
        th = float(np.quantile(maxsim[ins], 1.0 - keep)) if keep < 1.0 else -1.0
        sel = maxsim >= th
        oos_rej = (~sel & ~ins).sum() / max(1, (~ins).sum())
        m = sel & ins
        acc = correct[m].mean() if m.sum() else float("nan")
        mark = "  <- fitted" if abs(keep - cal.get("keep_in_scope", 0.99)) < 1e-9 else ""
        print(f"  {keep:12.0%} {th:10.3f} {oos_rej:12.1%} {acc:11.3f}{mark}")

    # Masking robustness: remove each item's true label from the legal set. A
    # softmax renormalises over what remains, so `prob` becomes confidently
    # wrong by construction. The question is whether `sim` still notices.
    hidden_flag = 0
    hidden_conf = []
    for i in np.where(ins)[0]:
        keep = [j for j, l in enumerate(labels) if l != truth[i]]
        sub = sims[i, keep]
        pp = _softmax(sub, args.temp)
        srt = np.sort(pp)
        marg = srt[-1] - srt[-2]
        flagged = (sub.max() < cal["min_sim"] or marg < cal["min_margin"]
                   or pp.max() < 0.55)
        hidden_flag += flagged
        hidden_conf.append(pp.max())
    n_ins = int(ins.sum())
    print(f"\nrobustness: true label masked out of the legal set ({n_ins} items)")
    print(f"  flagged rather than silently mislabelled: {hidden_flag}/{n_ins} "
          f"({hidden_flag/n_ins:.1%})")
    print(f"  mean p(top) on those forced-wrong decisions: {np.mean(hidden_conf):.3f} "
          f"-- prob alone would not have noticed")

    print("\nmax cosine also carries signal about in-scope correctness:")
    wrong = ins & ~correct
    print(f"  mean max-cosine, correct {maxsim[ins & correct].mean():.3f}  "
          f"wrong {maxsim[wrong].mean():.3f}")
    print(f"  AUROC separating correct from wrong, in-scope only: "
          f"{auroc(maxsim[ins], correct[ins]):.3f}")

if __name__ == "__main__":
    main()
