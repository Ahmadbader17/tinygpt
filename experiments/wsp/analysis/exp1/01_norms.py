"""Step 1: sizes. Which heads have large or small query-key and output-value circuits?"""

from __future__ import annotations

import torch
from common import D_MODEL, HEADS, N_HEAD, N_LAYER, centre, embedding_scale, ln1, load, qkv, w_out

p = load()
print("embedding scale s0 =", round(embedding_scale(p), 4))
print("rms of centred wte rows:", round(float(centre(p["wte.weight"]).pow(2).mean().sqrt()), 4))
pos_rms = centre(p["wpe.weight"]).pow(2).mean(dim=1).sqrt()
print(
    "rms of centred wpe rows: mean",
    round(float(pos_rms.mean()), 4),
    "first 4:",
    [round(float(v), 3) for v in pos_rms[:4]],
    "last 2:",
    [round(float(v), 3) for v in pos_rms[-2:]],
)
for layer in range(N_LAYER):
    g1 = p[f"blocks.{layer}.ln1.weight"]
    g2 = p[f"blocks.{layer}.ln2.weight"]
    print(
        f"layer {layer}: ln1 gain mean {float(g1.mean()):.3f} (min {float(g1.min()):.3f}, "
        f"max {float(g1.max()):.3f}); ln2 gain mean {float(g2.mean()):.3f}; "
        f"mlp.proj fro {float(p[f'blocks.{layer}.mlp.proj.weight'].norm()):.2f}; "
        f"attn.proj fro {float(p[f'blocks.{layer}.attn.proj.weight'].norm()):.2f}"
    )
print("ln_f gain mean", round(float(p["ln_f.weight"].mean()), 3))

print("\nPer head: |W_Q g|, |W_K g|, top singular value of QK = (W_Q g)^T (W_K g),")
print("|OV| = fro(W_O W_V g), top singular value of OV, and |OV| as a share of the layer.")
print("head    |Qg|    |Kg|  QK_s1  QK_s2   |OV|  OV_s1  OV_share  |b_Q|  |b_K|")
rows = {}
for layer, head in HEADS:
    wq, wk, wv, bq, bk, bv = qkv(p, layer, head)
    g, _ = ln1(p, layer)
    q, k, v = wq * g, wk * g, wv * g  # fold the LayerNorm gain into the input side
    qk = torch.linalg.svdvals(q.T @ k)
    ov_m = w_out(p, layer, head) @ v
    ov = torch.linalg.svdvals(ov_m)
    rows[(layer, head)] = (
        q.norm(),
        k.norm(),
        qk[0],
        qk[1],
        ov_m.norm(),
        ov[0],
        bq.norm(),
        bk.norm(),
    )
for layer in range(N_LAYER):
    total = sum(float(rows[(layer, h)][4]) ** 2 for h in range(N_HEAD)) ** 0.5
    for head in range(N_HEAD):
        r = rows[(layer, head)]
        print(
            f"{layer}.{head}  {float(r[0]):6.2f}  {float(r[1]):6.2f} {float(r[2]):6.2f} "
            f"{float(r[3]):6.2f} {float(r[4]):6.2f} {float(r[5]):6.2f}    "
            f"{float(r[4]) / total:5.2f}  {float(r[6]):5.2f}  {float(r[7]):5.2f}"
        )
    print()
assert D_MODEL == 256
