#!/usr/bin/env python3
"""Compile any BERT-architecture sentence encoder to Core ML for the ANE.

Generalises build.py. The forward pass is hand-written (see minilm.py) because
HuggingFace's own graph will not convert, so a new model must share BERT's
layout -- same state_dict keys, same attention shape. Layer count, hidden size
and head count are read from the checkpoint; pooling has to be told, since
getting it wrong degrades embeddings silently instead of failing.

  ./build_encoder.py --model BAAI/bge-small-en-v1.5 --pooling cls --out bge
"""
import argparse

import numpy as np
import torch

from minilm import MiniLM

SEQ = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    ap.add_argument("--out", default="encoder")
    args = ap.parse_args()

    import coremltools as ct
    from transformers import AutoModel, AutoTokenizer

    hf = AutoModel.from_pretrained(args.model).eval()
    cfg = hf.config
    print(f"{args.model}: {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, "
          f"{cfg.num_attention_heads} heads, pooling {args.pooling}", flush=True)
    model = MiniLM(hf.state_dict(), H=cfg.hidden_size, nh=cfg.num_attention_heads,
                   L=cfg.num_hidden_layers, T=SEQ, pooling=args.pooling).eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {n:.1f}M", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    e = tok(["a sanity check sentence about billing and refunds"],
            padding="max_length", max_length=SEQ, truncation=True, return_tensors="pt")
    ids, am = e["input_ids"].int(), e["attention_mask"]
    bias, pm = (1 - am)[:, None, None, :].float() * -1e4, am[:, :, None].float()
    with torch.no_grad():
        mine = model(ids, bias, pm)
        h = hf(input_ids=ids.long(), attention_mask=am).last_hidden_state
        ref = h[:, 0] if args.pooling == "cls" else (h * pm).sum(1) / pm.sum(1)
        ref = ref / ref.norm(dim=-1, keepdim=True)
    cos = float((mine * ref).sum())
    print(f"cosine(reimplementation, huggingface): {cos:.8f}", flush=True)
    if cos < 0.999:
        raise SystemExit("reimplementation diverged -- aborting")

    with torch.no_grad():
        traced = torch.jit.trace(model, (torch.ones(1, SEQ, dtype=torch.int32),
                                         torch.zeros(1, 1, 1, SEQ),
                                         torch.ones(1, SEQ, 1)))
    ml = ct.convert(
        traced, convert_to="mlprogram",
        inputs=[ct.TensorType(name="input_ids", shape=(1, SEQ), dtype=np.int32),
                ct.TensorType(name="attn_bias", shape=(1, 1, 1, SEQ), dtype=np.float32),
                ct.TensorType(name="pool_mask", shape=(1, SEQ, 1), dtype=np.float32)],
        outputs=[ct.TensorType(name="embedding")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE)
    ml.save(f"{args.out}.mlpackage")
    tok.save_pretrained(f"{args.out}_tok")
    print(f"wrote {args.out}.mlpackage and {args.out}_tok/", flush=True)


if __name__ == "__main__":
    main()
