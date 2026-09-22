#!/usr/bin/env python3
"""Can the encoder reproduce a domain rule from the link card's text alone?

If it can, it reaches the 82% of links whose domain no rule covers. The split
is by DOMAIN, never by post: centroids are built from one set of sites and
tested on different sites in the same category, so the encoder cannot pass by
memorising one publication's title formatting.

  ./evaluate.py cards.jsonl.gz                      # every category with data
  ./evaluate.py cards.jsonl.gz --cats video music news --balanced
"""
import argparse, collections, gzip, json, os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from rules import label                                    # noqa: E402
from system1 import Encoder                                # noqa: E402


def load(path, field="both"):
    op = gzip.open if path.endswith(".gz") else open
    rows = [json.loads(l) for l in op(path, "rt")]
    def txt(r):
        if field == "title" or not r["desc"]:
            return r["title"]
        return (r["title"] + ". " + r["desc"]).strip(" .")
    return rows, txt


def unit(v):
    return v / np.clip(np.linalg.norm(v), 1e-12, None)


def auroc(s, pos):
    o = np.argsort(s); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    n1, n0 = pos.sum(), (~pos).sum()
    return (rk[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cards")
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--field", default="both", choices=["both", "title"])
    ap.add_argument("--balanced", action="store_true",
                    help="also report accuracy on an equal-sized sample per class")
    ap.add_argument("--ks", type=int, nargs="*", default=[8, 16, 32, 64, 128])
    args = ap.parse_args()

    rows, txt = load(args.cards, args.field)
    bycat = collections.defaultdict(lambda: collections.defaultdict(list))
    tail = []
    for r in rows:
        c, t = label(r["domain"]), txt(r)
        if len(t) <= 12:
            continue
        if c:
            bycat[c][r["domain"]].append(t)
        else:
            tail.append((t, r["domain"]))
    n_lab = sum(len(v) for d in bycat.values() for v in d.values())
    print(f"{len(rows)} cards -> {n_lab} rule-labelled ({n_lab/len(rows):.1%}), "
          f"{len(tail)} tail, {len({r['domain'] for r in rows})} distinct domains\n")

    # Split each category's DOMAINS in two, alternating by size so neither side
    # gets only the small ones.
    train, test = {}, {}
    print(f"{'category':10s} {'cards':>6} {'domains':>8}   split")
    for c in sorted(bycat):
        if args.cats and c not in args.cats:
            continue
        doms = sorted(bycat[c], key=lambda d: -len(bycat[c][d]))
        n = sum(len(v) for v in bycat[c].values())
        if len(doms) < 2:
            print(f"{c:10s} {n:6d} {len(doms):8d}   SKIPPED (one domain only)")
            continue
        tr = [d for i, d in enumerate(doms) if i % 2 == 0]
        te = [d for i, d in enumerate(doms) if i % 2 == 1]
        train[c] = [t for d in tr for t in bycat[c][d]]
        test[c] = [t for d in te for t in bycat[c][d]]
        print(f"{c:10s} {n:6d} {len(doms):8d}   "
              f"train {len(tr)}d/{len(train[c])}c  test {len(te)}d/{len(test[c])}c")

    cats = sorted(c for c in train if len(train[c]) >= 8 and len(test[c]) >= 5)
    print(f"\nusable ({len(cats)}): {cats}\n")
    if not cats:
        return

    enc = Encoder(os.path.join(ROOT, "encoder.mlpackage"),
                  os.path.join(ROOT, "tok"))
    X, y = [], []
    for c in cats:
        X += test[c]; y += [c] * len(test[c])
    E, truth = enc.embed(X), np.array(y)
    rng = np.random.default_rng(0)
    nmin = min(int((truth == c).sum()) for c in cats)
    idx = np.concatenate([rng.permutation(np.where(truth == c)[0])[:nmin] for c in cats])

    hdr = f"{'k':>5} {'accuracy':>9}" + (f" {'balanced':>9}" if args.balanced else "")
    print(hdr + f" {'macro-F1':>9}")
    best = None
    for k in args.ks:
        M = np.stack([unit(enc.embed(train[c][:k]).mean(0)) for c in cats])
        pred = np.array([cats[i] for i in (E @ M.T).argmax(1)])
        f1 = []
        for c in cats:
            tp = ((pred == c) & (truth == c)).sum()
            p = tp / max((pred == c).sum(), 1); r = tp / max((truth == c).sum(), 1)
            f1.append(2 * p * r / max(p + r, 1e-9))
        acc, bal = (pred == truth).mean(), (pred[idx] == truth[idx]).mean()
        line = f"{k:5d} {acc:9.3f}" + (f" {bal:9.3f}" if args.balanced else "")
        print(line + f" {np.mean(f1):9.3f}")
        score = bal if args.balanced else acc
        if best is None or score > best[0]:
            best = (score, k, M, pred)

    score, k, M, pred = best
    maj = max((truth == c).mean() for c in cats)
    print(f"\nbest k={k}: {score:.3f} over {len(truth)} held-out cards from "
          f"unseen domains   (majority-class baseline {maj:.3f})\n")
    print(f"{'':9s}" + "".join(f"{c[:7]:>8s}" for c in cats) + "   recall")
    for c in cats:
        m = truth == c
        print(f"{c:9s}" + "".join(f"{int((pred[m] == p2).sum()):8d}" for p2 in cats)
              + f"   {(pred[m] == c).mean():.2f}")

    # The tail is not ground-truth out-of-scope -- much of it genuinely is one
    # of these categories from a domain no rule covers. Treat this as a noisy
    # target, not as an OOS benchmark.
    Et = enc.embed([t for t, _ in tail[:1500]])
    st, sl = (Et @ M.T).max(1), (E @ M.T).max(1)
    a = np.concatenate([sl, st])
    p = np.concatenate([np.ones(len(sl), bool), np.zeros(len(st), bool)])
    print(f"\nmax-cosine  in-schema {sl.mean():.3f}   tail {st.mean():.3f}   "
          f"AUROC {auroc(a, p):.3f}")
    for keep in (0.95, 0.90, 0.80):
        th = float(np.quantile(sl, 1 - keep))
        print(f"  keep {keep:.0%} of in-schema (min_sim {th:.3f}) -> "
              f"{(st < th).mean():.1%} of tail refused")

    print("\nmost confident tail placements — domains no rule covers:")
    for i in np.argsort(-st)[:8]:
        c = cats[int((Et[i] @ M.T).argmax())]
        print(f"  {st[i]:.3f} {c:6s} [{tail[i][1][:20]:20s}] {tail[i][0][:58]}")


if __name__ == "__main__":
    main()
