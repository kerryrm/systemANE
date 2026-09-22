#!/usr/bin/env python3
"""Measure the tier-1 engine on CLINC150. No tuning happens here.

Reports what a cascade actually cares about:
  * in-scope accuracy            -- is the answer right when it answers?
  * ECE                          -- does its confidence mean anything?
  * confident errors             -- and does it mean anything in the tail,
                                    where a wrong answer is also a sure one?
  * OOS AUROC                    -- can it tell "not in this schema" at all,
                                    and which signal does it best with?
  * risk/coverage                -- error rate among the inputs it keeps,
                                    as a function of how many it escalates
"""
import argparse

import numpy as np

import anchors as A
import evalset
from evalset import OOS
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
    ap.add_argument("--dataset", default="clinc", choices=["clinc", "massive"],
                    help="clinc = evalset.py (10 routes + OOS); massive = 60 intents, no OOS")
    ap.add_argument("--encoder", default="encoder",
                    help="basename: <name>.mlpackage + <name>_tok/ (or tok/)")
    ap.add_argument("--calibration", default="calibration.json")
    args = ap.parse_args()
    import importlib
    ds = importlib.import_module(
        {'clinc': 'evalset', 'massive': 'massive'}[args.dataset])
    cal = load_calibration(args.calibration)
    if args.temp is None:
        args.temp = cal["temp"]
    k = cal.get("k", 0) if args.k is None else args.k
    mode, use_desc = cal.get("mode", "centroid"), cal.get("use_description", True)

    rows = ds.load(args.split, max_oos=args.max_oos)
    print(f"{args.dataset} {args.split}: {ds.summary(rows)}\n")

    enc = _encoder(args.encoder)
    an = A.build(enc, k=k, mode=mode, use_description=use_desc, ds=ds)
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
    n_ins = int(ins.sum())

    print(f"in-scope accuracy      {correct[ins].mean():.3f}  ({correct[ins].sum()}/{ins.sum()})")
    print(f"ECE (in-scope, T={args.temp})  {ece(top[ins], correct[ins].astype(float)):.3f}")
    print(f"mean confidence        {top[ins].mean():.3f}  vs accuracy {correct[ins].mean():.3f}")

    # ECE averages the whole reliability diagram, so it is silent about the
    # tail that actually costs something: a wrong answer the engine is sure
    # about. Nothing downstream can catch one -- that is the definition -- so
    # the rate is a hard floor on what a confidence gate can do.
    print("\nconfident errors -- in-scope, wrong, and sure about it")
    print(f"  {'p(top) >=':>9} {'items':>6} {'wrong':>6} {'err|conf':>9}"
          f" {'share of in-scope':>18}")
    for th in (0.90, 0.95, 0.99):
        m = ins & (top >= th)
        w = int((m & ~correct).sum())
        rate = w / m.sum() if m.sum() else float("nan")
        print(f"  {th:9.2f} {int(m.sum()):6d} {w:6d} {rate:9.3f}"
              f" {w / n_ins:17.1%}")

    # The rate is a property of the temperature, not of the encoder, and
    # fitting moved temp *down* -- which sharpens. Calibration improved ECE
    # and should be expected to make this tail worse, not better.
    print("  by temperature, at p >= 0.90:")
    for t in sorted({0.10, round(args.temp, 4), 0.03, 0.02}, reverse=True):
        pt = np.stack([_softmax(s, t) for s in sims]).max(1)
        conf = ins & (pt >= 0.90)
        w = int((conf & ~correct).sum())
        mark = "  <- fitted" if abs(t - args.temp) < 1e-4 else ""
        # The item count matters: a temperature that never reaches 0.90 scores
        # zero confident errors by being uselessly underconfident, not by being
        # safe. T=0.10 is that case, and it is the one fitting moved away from.
        print(f"    T={t:<7.4f} {int(conf.sum()):3d}/{n_ins} reach 0.90,"
              f" {w:3d} wrong ({w / n_ins:5.1%} of in-scope){mark}")

    # A confident error that min_sim or min_margin flags is not silent: the
    # cascade still escalates or refuses it. The residue is the real number.
    cw = ins & ~correct & (top >= 0.90)
    gated = cw & ((maxsim < cal["min_sim"]) | (margin < cal["min_margin"]))
    print(f"  of the {int(cw.sum())} at p >= 0.90: {int(gated.sum())} flagged by"
          f" min_sim/min_margin, {int((cw & ~gated).sum())} silent"
          f" ({(cw & ~gated).sum() / n_ins:.1%} of in-scope)")

    n_oos = int((~ins).sum())
    if n_oos:
        print("\nout-of-scope detection (AUROC, higher = separates better)")
        for name, s in (("max cosine (sim)", maxsim), ("margin", margin), ("top prob", top)):
            print(f"  {name:18s} {auroc(s, ins):.3f}")
    else:
        print(f"\nout-of-scope detection: not measurable — {args.dataset} ships no "
              f"out-of-scope rows.\n  Refusal is untested on this dataset; the OOS "
              f"numbers in README.md are CLINC's.")

    # Sweep by in-scope retention, not by absolute threshold: different
    # encoders put their similarities in completely different ranges (MiniLM
    # fits min_sim 0.31, BGE 0.75), so a fixed 0.10-0.40 grid is meaningless
    # across encoders and reported 0% rejected for BGE.
    print("\noperating curve — threshold set to retain a share of in-scope")
    if not n_oos:
        print("  (oos-rejected column is vacuous here: no out-of-scope rows)")
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
