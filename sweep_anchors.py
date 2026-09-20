#!/usr/bin/env python3
"""Select the class representation on validation. Test is not touched."""
import numpy as np

import anchors as A
import evalset
from evalset import OOS, ROUTES
from evaluate import auroc
from system1 import Encoder

K_MAX = 32


def main():
    enc = Encoder()
    need = A.texts_needed(K_MAX)
    cache = dict(zip(need, enc.embed(need)))
    print(f"embedded {len(need)} anchor candidates\n")

    rows = evalset.load("validation")
    truth = np.array([l for _, l in rows])
    E = enc.embed([t for t, _ in rows])
    ins = truth != OOS

    print(f"{'repr':34s} {'acc':>6} {'OOS AUROC':>10}")
    print("-" * 54)
    configs = [("description only", 0, "centroid", True)]
    for k in (1, 2, 4, 8, 16, 32):
        configs.append((f"{k} examples (centroid)", k, "centroid", False))
        configs.append((f"{k} examples (max-sim)", k, "max", False))
        configs.append((f"desc + {k} examples (centroid)", k, "centroid", True))
    best = None
    for name, k, mode, desc in configs:
        an = A.build(enc, k=k, mode=mode, use_description=desc, cache=cache)
        sims = an.score(E)
        pred = np.array([an.labels[i] for i in sims.argmax(1)])
        acc = float(((pred == truth) & ins)[ins].mean())
        au = auroc(sims.max(1), ins)
        flag = ""
        if best is None or acc > best[0]:
            best, flag = (acc, name, k, mode, desc), "  <-"
        print(f"{name:34s} {acc:6.3f} {au:10.3f}{flag}")
    print(f"\nbest on validation: {best[1]}  (acc {best[0]:.3f})")


if __name__ == "__main__":
    main()
