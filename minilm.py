"""MiniLM-L6 forward pass written directly, so the traced graph contains only
ops we choose. Masks are precomputed outside the graph to avoid int casts."""
import torch, torch.nn as nn, torch.nn.functional as Fn

class Layer(nn.Module):
    def __init__(s, sd, p, H, nh, T, B=1):
        super().__init__(); s.H, s.nh, s.hd, s.T, s.B = H, nh, H // nh, T, B
        g = lambda k: sd[f"{p}.{k}"]
        s.q, s.k, s.v = (nn.Linear(H, H) for _ in range(3))
        s.q.weight.data, s.q.bias.data = g("attention.self.query.weight"), g("attention.self.query.bias")
        s.k.weight.data, s.k.bias.data = g("attention.self.key.weight"), g("attention.self.key.bias")
        s.v.weight.data, s.v.bias.data = g("attention.self.value.weight"), g("attention.self.value.bias")
        s.o = nn.Linear(H, H); s.o.weight.data, s.o.bias.data = g("attention.output.dense.weight"), g("attention.output.dense.bias")
        s.ln1 = nn.LayerNorm(H, eps=1e-12); s.ln1.weight.data, s.ln1.bias.data = g("attention.output.LayerNorm.weight"), g("attention.output.LayerNorm.bias")
        I = g("intermediate.dense.weight").shape[0]
        s.fc1 = nn.Linear(H, I); s.fc1.weight.data, s.fc1.bias.data = g("intermediate.dense.weight"), g("intermediate.dense.bias")
        s.fc2 = nn.Linear(I, H); s.fc2.weight.data, s.fc2.bias.data = g("output.dense.weight"), g("output.dense.bias")
        s.ln2 = nn.LayerNorm(H, eps=1e-12); s.ln2.weight.data, s.ln2.bias.data = g("output.LayerNorm.weight"), g("output.LayerNorm.bias")

    def forward(s, x, bias):
        B, T, H = s.B, s.T, s.H
        shape = (B, T, s.nh, s.hd)
        q = s.q(x).view(shape).permute(0, 2, 1, 3)
        k = s.k(x).view(shape).permute(0, 2, 3, 1)
        v = s.v(x).view(shape).permute(0, 2, 1, 3)
        att = torch.softmax(torch.matmul(q, k) / (s.hd ** 0.5) + bias, dim=-1)
        ctx = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(B, T, H)
        x = s.ln1(x + s.o(ctx))
        return s.ln2(x + s.fc2(Fn.gelu(s.fc1(x))))

class MiniLM(nn.Module):
    def __init__(s, sd, H=384, nh=12, L=6, T=128, B=1, pooling="mean"):
        super().__init__(); s.T, s.B, s.pooling = T, B, pooling
        s.we = nn.Embedding.from_pretrained(sd["embeddings.word_embeddings.weight"], freeze=True)
        s.pe = nn.Embedding.from_pretrained(sd["embeddings.position_embeddings.weight"], freeze=True)
        s.te = nn.Embedding.from_pretrained(sd["embeddings.token_type_embeddings.weight"], freeze=True)
        s.ln = nn.LayerNorm(H, eps=1e-12)
        s.ln.weight.data, s.ln.bias.data = sd["embeddings.LayerNorm.weight"], sd["embeddings.LayerNorm.bias"]
        s.layers = nn.ModuleList([Layer(sd, f"encoder.layer.{i}", H, nh, T, B)
                                  for i in range(L)])
        s.register_buffer("pos", torch.arange(T, dtype=torch.long).unsqueeze(0))
        s.register_buffer("tt", torch.zeros(1, T, dtype=torch.long))

    def forward(s, input_ids, attn_bias, pool_mask):
        x = s.ln(s.we(input_ids.long()) + s.pe(s.pos) + s.te(s.tt))
        for l in s.layers:
            x = l(x, attn_bias)
        # BGE-family models pool the CLS token; MiniLM/GTE mean-pool. Using the
        # wrong one silently degrades the embedding rather than erroring.
        if s.pooling == "cls":
            pooled = x[:, 0]
        else:
            pooled = (x * pool_mask).sum(1) / pool_mask.sum(1).clamp(min=1e-9)
        return pooled * torch.rsqrt((pooled * pooled).sum(-1, keepdim=True) + 1e-12)
