"""Step 5: how much can content move a head's attention scores?

If the scores of different keys differ by a standard deviation sigma because of content, two
windows with different text give patterns whose cosine is roughly exp(-sigma^2) (log-normal
weights, many keys). That is the "positional" score of the plan, so sigma is what to estimate.

(a) iso: sigma if the normalised residual were isotropic: fro((W_Q g)^T (W_K g)) / sqrt(32).
    Valid at any layer as a rough ceiling, since real residuals share a mean direction.
(b) direct-path sigma from token embeddings, once over the whole vocabulary and once over ids
    below 1024 (bytes and the earliest, most frequent merges), with the layer-0 scale s0.
    Exact for layer 0; for deeper layers the embedding share of the residual is smaller.
"""

from __future__ import annotations

import math

import torch
from common import HEADS, centre, embedding_scale, ln1, load, qkv

p = load()
s0 = embedding_scale(p)
wte, wpe = centre(p["wte.weight"]), centre(p["wpe.weight"])
gen = torch.Generator().manual_seed(1)
norm_all = wte[1:].norm(dim=1)
print(
    f"token embedding norms: all tokens mean {float(norm_all.mean()):.3f}; "
    f"ids<1024 mean {float(wte[1:1024].norm(dim=1).mean()):.3f}; "
    f"ids>=3000 mean {float(wte[3000:].norm(dim=1).mean()):.3f}; eot {float(wte[0].norm()):.3f}"
)


def spread(
    q_tok: torch.Tensor, k_tok: torch.Tensor, q_base: torch.Tensor, ids: torch.Tensor
) -> float:
    """Std over random keys of the score, for random queries (key-dependent part only)."""
    a = ids[torch.randint(0, len(ids), (4000,), generator=gen)]
    b = ids[torch.randint(0, len(ids), (4000,), generator=gen)]
    q = q_base + q_tok[a]  # (4000, 32) queries
    s = q @ k_tok[b].T / math.sqrt(32)  # each query against 4000 random keys
    return float(s.std(dim=1).mean())


print("head   iso  exp(-iso^2) | sig_all  sig_freq  exp(-sig_freq^2)")
frequent = torch.arange(1, 1024)
everything = torch.arange(1, 4096)
for layer, head in HEADS:
    wq, wk, _, bq, _, _ = qkv(p, layer, head)
    g, b = ln1(p, layer)
    m = (wq * g).T @ (wk * g)
    iso = float(m.norm()) / math.sqrt(32)
    q_tok, k_tok = (wte * g) @ wq.T / s0, (wte * g) @ wk.T / s0
    q_base = ((wpe * g) @ wq.T / s0)[16:].mean(dim=0) + wq @ b + bq
    sa = spread(q_tok, k_tok, q_base, everything)
    sf = spread(q_tok, k_tok, q_base, frequent)
    print(
        f"{layer}.{head}  {iso:5.2f}   {math.exp(-(iso**2)):5.2f}     | {sa:6.2f}   {sf:6.2f}    "
        f"{math.exp(-(sf**2)):5.2f}"
    )
    if head == 7:
        print()
