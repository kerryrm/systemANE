#!/usr/bin/env python3
"""Convert all-MiniLM-L6-v2 into a fixed-shape Core ML encoder for the ANE.

Run once: ./build.py  ->  encoder.mlpackage + tok/

Two things this file exists to work around, both documented in the README:
  * HuggingFace's BertModel graph does not convert (its attention-mask code
    emits an int cast coremltools cannot fold), so minilm.py reimplements the
    forward pass; it is verified against HF to cosine 0.99999994 below.
  * Any `x.shape[1]` read becomes aten::Int under tracing and fails the same
    way, so every shape is a compile-time constant.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from minilm import MiniLM  # noqa: E402

NAME, SEQ = "sentence-transformers/all-MiniLM-L6-v2", 128


def main():
    import coremltools as ct
    from transformers import AutoModel, AutoTokenizer

    hf = AutoModel.from_pretrained(NAME).eval()
    model = MiniLM(hf.state_dict(), T=SEQ).eval()
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    tok = AutoTokenizer.from_pretrained(NAME)
    e = tok(["a sanity check sentence"], padding="max_length",
            max_length=SEQ, truncation=True, return_tensors="pt")
    ids, am = e["input_ids"].int(), e["attention_mask"]
    bias, pm = (1 - am)[:, None, None, :].float() * -1e4, am[:, :, None].float()
    with torch.no_grad():
        mine = model(ids, bias, pm)
        h = hf(input_ids=ids.long(), attention_mask=am).last_hidden_state
        p = (h * pm).sum(1) / pm.sum(1)
        ref = p / p.norm(dim=-1, keepdim=True)
    cos = float((mine * ref).sum())
    print(f"cosine(reimplementation, huggingface): {cos:.8f}")
    if cos < 0.999:
        raise SystemExit("reimplementation diverged from HuggingFace -- aborting")

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
    ml.save("encoder.mlpackage")
    tok.save_pretrained("tok")
    print("wrote encoder.mlpackage and tok/")


if __name__ == "__main__":
    main()
