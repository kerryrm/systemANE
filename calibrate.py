#!/usr/bin/env python3
"""Fit the two knobs on the validation split, then write them to calibration.json.

Temperature is fitted by minimising negative log-likelihood over in-scope
validation items -- the standard choice, and it is what makes `prob` mean
something. min_sim is fitted to a target in-scope retention, since losing real
traffic is the expensive error and rejecting out-of-scope is the cheap win.

Nothing here reads the test split. Run evaluate.py afterwards to see the result.
"""
import argparse
import json

import numpy as np

import anchors as A
import evalset
from evalset import OOS, ROUTES
from system1 import Encoder, _softmax


def _encoder(name):
    import os
    tok = f"{name}_tok" if os.path.isdir(f"{name}_tok") else "tok"
    return Encoder(f"{name}.mlpackage", tok)


def nll(sims, y, temp):
    p = np.stack([_softmax(s, temp) for s in sims])
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            total += m.sum() / len(conf) * abs(correct[m].mean() - conf[m].mean())
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-in-scope", type=float, default=0.99,
                    help="target fraction of in-scope traffic to retain")
    ap.add_argument("--target-error", type=float, default=0.05,
                    help="target in-scope error rate among inputs not escalated")
    ap.add_argument("--k", type=int, default=16, help="train examples per class")
    ap.add_argument("--mode", default="centroid", choices=["centroid", "max"])
    ap.add_argument("--no-description", action="store_true")
    ap.add_argument("--encoder", default="encoder",
                    help="basename: <name>.mlpackage + <name>_tok/ (or tok/)")
    ap.add_argument("--out", default="calibration.json")
    args = ap.parse_args()

    rows = evalset.load("validation")
    print(f"validation: {evalset.summary(rows)}\n")
    enc = _encoder(args.encoder)
    an = A.build(enc, k=args.k, mode=args.mode, use_description=not args.no_description)
    labels = an.labels
    print(f"anchors: k={args.k} mode={args.mode} "
          f"description={not args.no_description}\n")
    E = enc.embed([t for t, _ in rows])
    truth = np.array([l for _, l in rows])
    sims = an.score(E)
    ins = truth != OOS
    y = np.array([labels.index(l) if l != OOS else -1 for l in truth])

    # -- temperature, by NLL on in-scope validation items ------------------
    grid = np.exp(np.linspace(np.log(0.01), np.log(3.0), 300))
    losses = [nll(sims[ins], y[ins], t) for t in grid]
    temp = float(grid[int(np.argmin(losses))])
    print(f"fitted temperature      {temp:.4f}   (NLL {min(losses):.4f})")
    for t in (0.05, 0.10, temp, 1.0):
        P = np.stack([_softmax(s, t) for s in sims[ins]])
        conf, corr = P.max(1), (P.argmax(1) == y[ins]).astype(float)
        tag = "  <- fitted" if abs(t - temp) < 1e-9 else ""
        print(f"  T={t:6.4f}  NLL {nll(sims[ins], y[ins], t):.4f}  "
              f"ECE {ece(conf, corr):.4f}  mean conf {conf.mean():.3f} "
              f"vs acc {corr.mean():.3f}{tag}")

    # -- min_sim, to a target in-scope retention ---------------------------
    maxsim = sims.max(1)
    cut = float(np.quantile(maxsim[ins], 1.0 - args.keep_in_scope))
    rejected = (maxsim[~ins] < cut).mean()
    print(f"\nfitted min_sim          {cut:.4f}   "
          f"(keeps {args.keep_in_scope:.0%} in-scope, rejects {rejected:.1%} of out-of-scope)")

    # -- min_margin, to a target in-scope error rate ----------------------
    # margin is post-softmax, so it is only meaningful at the fitted
    # temperature; refit it whenever temp changes.
    P = np.stack([_softmax(s_, temp) for s_ in sims[ins]])
    srt = np.sort(P, axis=1)
    marg = srt[:, -1] - srt[:, -2]
    ok = (P.argmax(1) == y[ins])
    print(f"\nmargin threshold sweep (in-scope validation, at fitted T)")
    print(f"  {'margin':>7} {'escalated':>10} {'error on kept':>14}")
    best = 0.0
    for th in np.round(np.arange(0.0, 0.96, 0.05), 2):
        keep = marg >= th
        err = 1 - ok[keep].mean() if keep.sum() else 0.0
        print(f"  {th:7.2f} {1-keep.mean():9.1%} {err:14.3f}")
        if err <= args.target_error and best == 0.0:
            best = float(th)
    print(f"fitted min_margin       {best:.2f}   "
          f"(targets <= {args.target_error:.0%} error among kept in-scope)")

    json.dump({"temp": round(temp, 4), "min_sim": round(cut, 4),
               "min_margin": best,
               "k": args.k, "mode": args.mode,
               "use_description": not args.no_description,
               "fitted_on": "clinc_oos validation", "keep_in_scope": args.keep_in_scope},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out} — evaluate.py and Choice() read it by default")


if __name__ == "__main__":
    main()
