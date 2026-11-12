"""Step 6: if a head attends diffusely, what does it write?

A head that spreads its attention evenly writes roughly O V applied to the average input,
which is nearly the same vector at every position. We estimate the size of that constant
write and of the part that varies with which tokens are in the context, relative to the
size of a token-plus-position embedding (the residual stream entering layer 0).
Used only to judge which heads could be removed at little cost ("mostly inactive").
"""

from __future__ import annotations

import torch
from common import HEADS, centre, embedding_scale, ln1, load, qkv, w_out

p = load()
s0 = embedding_scale(p)
wte, wpe = centre(p["wte.weight"]), centre(p["wpe.weight"])
resid = float((wte[1:].pow(2).sum(dim=1).mean() + wpe[16:].pow(2).sum(dim=1).mean()).sqrt())
mean_in = (wte[1:1024].mean(dim=0) + wpe[16:].mean(dim=0)) / s0  # average normalised input
print(
    f"embedding residual norm ~ {resid:.3f}; |mean normalised input| = {float(mean_in.norm()):.2f} of 16"
)
print(
    "head  |const|  const/resid  |one token|/resid   (one token: O V applied to one token's embedding)"
)
for layer, head in HEADS:
    wq, wk, wv, bq, bk, bv = qkv(p, layer, head)
    g, b = ln1(p, layer)
    wo = w_out(p, layer, head)
    const = wo @ (wv @ (g * mean_in) + wv @ b + bv)
    per_token = ((wte[1:1024] - wte[1:1024].mean(dim=0)) * g / s0) @ wv.T @ wo.T
    one = float(per_token.norm(dim=1).mean())
    print(
        f"{layer}.{head}   {float(const.norm()):5.2f}     {float(const.norm()) / resid:5.2f}      {one / resid:5.2f}"
    )
    if head == 7:
        print()
