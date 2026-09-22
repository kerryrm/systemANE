#!/usr/bin/env python3
"""What fp16 on the ANE costs, measured on decisions rather than on fixtures.

`build.py` checks the reimplemented forward pass against HuggingFace to cosine
0.99999994, which bounds the *embedding* error. That is not the quantity anyone
cares about: the question is whether the deployed path ever returns a different
*decision* than the fp32 reference, and by how much the probability moves when
it does not.

An earlier check answered that with ten hand-written strings at K=5. This runs
the whole evaluation set, end to end -- anchors and queries both embedded by
each path, so anchor drift is included rather than assumed away -- and it runs
at K=10 (clinc) and K=60 (massive), because the standing caveat was that
precision should matter more as the label count grows.

  ./drift.py --dataset clinc
  ./drift.py --dataset massive --k 16 --calibration calibration_massive.json
"""
import argparse
import importlib

import numpy as np
import torch

import anchors as A
from system1 import SEQ, Encoder, _softmax, load_calibration

NAME = "sentence-transformers/all-MiniLM-L6-v2"


class TorchEncoder:
    """The fp32 reference: minilm.py on CPU, same weights, same tokenizer."""

    def __init__(self, tokdir="tok"):
        from transformers import AutoModel, AutoTokenizer
        from minilm import MiniLM
        hf = AutoModel.from_pretrained(NAME).eval()
        self.m = MiniLM(hf.state_dict(), T=SEQ).eval().float()
        self.tok = AutoTokenizer.from_pretrained(tokdir)

    def embed(self, texts):
        out = []
        with torch.no_grad():
            for t in texts:
                e = self.tok([t], padding="max_length", max_length=SEQ,
                             truncation=True, return_tensors="pt")
                am = e["attention_mask"]
                bias = (1 - am)[:, None, None, :].float() * -1e4
                pm = am[:, :, None].float()
                out.append(self.m(e["input_ids"].int(), bias, pm).numpy().ravel())
        return np.stack(out)


def decisions(enc, ds, rows, k, mode, use_desc, temp):
    """Embed anchors and queries with one encoder; return (pred, p_top, sims)."""
    an = A.build(enc, k=k, mode=mode, use_description=use_desc, ds=ds)
    sims = an.score(enc.embed([t for t, _ in rows]))
    P = np.stack([_softmax(s, temp) for s in sims])
    return np.array([an.labels[i] for i in sims.argmax(1)]), P.max(1), sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="clinc", choices=["clinc", "massive"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--calibration", default="calibration.json")
    ap.add_argument("--limit", type=int, default=None, help="first N rows only")
    args = ap.parse_args()

    ds = importlib.import_module(
        {"clinc": "evalset", "massive": "massive"}[args.dataset])
    cal = load_calibration(args.calibration)
    temp = cal["temp"]
    k = cal.get("k", 0) if args.k is None else args.k
    mode, use_desc = cal.get("mode", "centroid"), cal.get("use_description", True)

    rows = ds.load(args.split)
    if args.limit:
        rows = rows[:args.limit]
    truth = np.array([l for _, l in rows])
    ins = truth != ds.OOS
    n_lab = len(ds.ROUTES)
    print(f"{args.dataset} {args.split}: {len(rows)} rows, K={n_lab}, "
          f"k={k} mode={mode} description={use_desc}, T={temp}\n")

    ref = TorchEncoder()
    ane = Encoder(compute_units="CPU_AND_NE")
    cpu = Encoder(compute_units="CPU_ONLY")

    E_ref = ref.embed([t for t, _ in rows])
    p_ref, t_ref, _ = decisions(ref, ds, rows, k, mode, use_desc, temp)

    for tag, enc in (("Core ML fp16, CPU+ANE", ane), ("Core ML fp16, CPU only", cpu)):
        E = enc.embed([t for t, _ in rows])
        pred, top, _ = decisions(enc, ds, rows, k, mode, use_desc, temp)
        cos = (E_ref * E).sum(1) / (np.linalg.norm(E_ref, axis=1) * np.linalg.norm(E, axis=1))
        agree = pred == p_ref
        dp = np.abs(top - t_ref)
        flips = np.where(~agree)[0]
        # A flip only matters if it changes whether the answer was right. A
        # flip from one wrong label to another wrong label costs nothing.
        made_wrong = int(((p_ref == truth) & ~agree).sum())
        made_right = int(((pred == truth) & ~agree).sum())
        print(f"--- {tag} ---")
        print(f"  embedding cosine vs fp32   mean {cos.mean():.7f}  min {cos.min():.7f}")
        print(f"  same decision              {int(agree.sum())}/{len(rows)} "
              f"({agree.mean():.4%})")
        # In-scope is the only population where a flip can cost anything. An
        # out-of-scope item has no correct label to lose: it is refused on
        # `sim`, which is a magnitude and barely moves, not on the argmax.
        if ins.any() and (~ins).any():
            print(f"    in-scope                 {int(agree[ins].sum())}/{int(ins.sum())} "
                  f"({agree[ins].mean():.4%})")
            print(f"    out-of-scope             {int(agree[~ins].sum())}/{int((~ins).sum())} "
                  f"({agree[~ins].mean():.4%})")
        print(f"  |delta p(top)|             mean {dp.mean():.6f}  max {dp.max():.6f}")
        print(f"  flips that broke a correct answer: {made_wrong}")
        print(f"  flips that fixed a wrong answer:   {made_right}")
        for i in flips[:5]:
            print(f"    {t_ref[i]:.3f}->{top[i]:.3f}  {p_ref[i]} -> {pred[i]}"
                  f"   {rows[i][0][:52]}")
        print()


if __name__ == "__main__":
    main()
