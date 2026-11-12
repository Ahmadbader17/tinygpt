"""Step 3: can position embeddings tell neighbours apart at all, and how do layer-0/1 scores
fall off with distance? Still the direct path, still weights only."""

from __future__ import annotations

import math

import torch
from common import HEADS, centre, embedding_scale, ln1, load, qkv

p = load()
s0 = embedding_scale(p)
wpe = centre(p["wpe.weight"])
norms = wpe.norm(dim=1)
unit = wpe / norms[:, None]
print("wpe row norms at positions 0,1,2,3,4,8,16,32,64,128,255:")
print([round(float(norms[i]), 3) for i in (0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 255)])
for d in (1, 2, 4, 8, 16, 64):
    cos = (unit[d:] * unit[:-d]).sum(dim=1)
    print(
        f"cosine between positions i and i-{d}: mean {float(cos[16:].mean()):.3f}, "
        f"min {float(cos[16:].min()):.3f}  (i >= 16+{d})"
    )
sv = torch.linalg.svdvals(wpe)
energy = (sv**2).cumsum(0) / (sv**2).sum()
print("wpe singular values (top 10):", [round(float(v), 3) for v in sv[:10]])
print(
    "components for 50/90/99% of the energy:",
    [int((energy < f).sum()) + 1 for f in (0.5, 0.9, 0.99)],
)
diff = (wpe[1:] - wpe[:-1]).norm(dim=1)
print(f"|wpe_i - wpe_(i-1)| / |wpe_i|: mean {float((diff[16:] / norms[17:]).mean()):.3f}")

T = wpe.shape[0]
print("\nMean position-only score at key offset d, minus the row mean (nats), i >= 32:")
print("head     d=0    d=1    d=2    d=3    d=4    d=8   d=16   d=32 | key0   max-min of a row")
for layer, head in HEADS:
    if layer > 2:
        break
    wq, wk, _, bq, _, _ = qkv(p, layer, head)
    g, b = ln1(p, layer)
    q = (wpe * g) @ wq.T / s0 + wq @ b + bq
    k = (wpe * g) @ wk.T / s0
    s = q @ k.T / math.sqrt(32)
    rows = torch.arange(32, T)
    centred = []
    for i in rows:
        centred.append(s[i, : i + 1] - s[i, : i + 1].mean())
    prof = [
        sum(float(c[-1 - d]) for c in centred) / len(centred) for d in (0, 1, 2, 3, 4, 8, 16, 32)
    ]
    key0 = sum(float(c[0]) for c in centred) / len(centred)
    spread = sum(float(c.max() - c.min()) for c in centred) / len(centred)
    print(
        f"{layer}.{head}  " + " ".join(f"{v:6.2f}" for v in prof) + f" | {key0:5.2f}  {spread:5.2f}"
    )
    if head == 7:
        print()
