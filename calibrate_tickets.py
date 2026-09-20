#!/usr/bin/env python3
"""Fit the support-ticket schema's thresholds, and write calibration_tickets.json.

Same method as calibrate.py -- temperature by NLL, min_sim by in-scope
retention, min_margin by target error -- but for the five routes in tickets.py
rather than the CLINC banking ones. It exists because calibration does not
transfer between schemas: the CLINC min_sim of 0.3085 refuses genuinely
in-scope tickets here, at 0.27-0.28.

Fitting uses tickets.VALIDATION only. The CLEAR/AMBIGUOUS/OUT_OF_SCOPE fixtures
that cascade.py reports on are never read here, so the demo stays a held-out
test rather than a restatement of the fit.
"""
import argparse
import json

import numpy as np

from calibrate import ece, nll
from system1 import Encoder, _softmax
from tickets import OOS, ROUTES, VALIDATION


def main():
    ap = argparse.ArgumentParser()
    # 0.80, not calibrate.py's 0.99. This schema's anchors are descriptions,
    # not example centroids, and its in-scope and out-of-scope similarities
    # genuinely overlap: a concrete bug report ("dates display as 1970") sits
    # at cosine 0.051 to the abstract `bug` description, below chatty
    # out-of-scope like "I have attached the document" at 0.279. At 0.99
    # retention the threshold collapses under the entire out-of-scope range
    # and rejects nothing. 0.80 is the knee of the curve printed below.
    ap.add_argument("--keep-in-scope", type=float, default=0.80,
                    help="target fraction of in-scope traffic to retain")
    ap.add_argument("--target-error", type=float, default=0.05,
                    help="target in-scope error rate among inputs not escalated")
    ap.add_argument("--out", default="calibration_tickets.json")
    args = ap.parse_args()

    labels = list(ROUTES)
    n_in = sum(1 for _, l in VALIDATION if l != OOS)
    print(f"validation: {len(VALIDATION)} tickets, {len(labels)} in-scope "
          f"classes, {len(VALIDATION) - n_in} out-of-scope\n")

    enc = Encoder()
    A = enc.embed([ROUTES[l] for l in labels])
    E = enc.embed([t for t, _ in VALIDATION])
    sims = E @ A.T

    truth = np.array([l for _, l in VALIDATION])
    ins = truth != OOS
    y = np.array([labels.index(l) if l != OOS else -1 for l in truth])

    # -- temperature, by NLL on in-scope validation items ------------------
    grid = np.exp(np.linspace(np.log(0.01), np.log(3.0), 300))
    losses = [nll(sims[ins], y[ins], t) for t in grid]
    temp = float(grid[int(np.argmin(losses))])
    print(f"fitted temperature      {temp:.4f}   (NLL {min(losses):.4f})")
    for t in (0.05, 0.10, temp):
        P = np.stack([_softmax(s, t) for s in sims[ins]])
        conf, corr = P.max(1), (P.argmax(1) == y[ins]).astype(float)
        tag = "  <- fitted" if abs(t - temp) < 1e-9 else ""
        print(f"  T={t:6.4f}  NLL {nll(sims[ins], y[ins], t):.4f}  "
              f"ECE {ece(conf, corr):.4f}  mean conf {conf.mean():.3f} "
              f"vs acc {corr.mean():.3f}{tag}")

    # -- min_sim, to a target in-scope retention ---------------------------
    maxsim = sims.max(1)
    cut = float(np.quantile(maxsim[ins], 1.0 - args.keep_in_scope))
    rejected = float((maxsim[~ins] < cut).mean())
    print(f"\nfitted min_sim          {cut:.4f}   (keeps {args.keep_in_scope:.0%} "
          f"in-scope, rejects {rejected:.1%} of out-of-scope)")
    print(f"  in-scope  max-cosine range {maxsim[ins].min():.3f} - {maxsim[ins].max():.3f}")
    print(f"  oos       max-cosine range {maxsim[~ins].min():.3f} - {maxsim[~ins].max():.3f}")
    print(f"\n  {'thresh':>7} {'in-scope kept':>14} {'oos rejected':>13}")
    for th in np.round(np.arange(0.05, 0.41, 0.05), 2):
        tag = "  <- fitted" if abs(th - round(cut, 2)) < 0.025 else ""
        print(f"  {th:7.2f} {(maxsim[ins] >= th).mean():13.1%} "
              f"{(maxsim[~ins] < th).mean():12.1%}{tag}")
    if rejected < 0.20:
        print("\n  WARNING: this threshold rejects almost no out-of-scope traffic.")
        print("  The two populations overlap; lower --keep-in-scope or use "
              "example\n  anchors instead of descriptions.")

    # -- min_margin, to a target in-scope error rate -----------------------
    # margin is post-softmax, so it only means anything at the fitted
    # temperature; refit it whenever temp changes.
    P = np.stack([_softmax(s, temp) for s in sims[ins]])
    srt = np.sort(P, axis=1)
    marg = srt[:, -1] - srt[:, -2]
    ok = P.argmax(1) == y[ins]
    print(f"\nmargin sweep (in-scope validation, at fitted T)")
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

    print(f"\nin-scope accuracy on validation: {ok.mean():.3f} "
          f"({int(ok.sum())}/{int(ins.sum())})")

    json.dump({"temp": round(temp, 4), "min_sim": round(cut, 4),
               "min_margin": best,
               "fitted_on": "tickets.VALIDATION",
               "keep_in_scope": args.keep_in_scope},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out} — cascade.py reads it by default")


if __name__ == "__main__":
    main()
