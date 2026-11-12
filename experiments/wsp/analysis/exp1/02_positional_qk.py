"""Step 2: the positional part of each head's query-key circuit, on the direct path.

Residual at a position = token embedding + position embedding (+ what earlier layers wrote).
On the direct path the attention score from query position i to key position j is
    (q_pos(i) + q_const + q_tok(a)) . (k_pos(j) + k_tok(b)) / sqrt(32),
with q_pos(i) = W_Q (g * centre(wpe_i)) / s, q_const = W_Q b_ln + b_Q, and so on.
We read: (1) the position-only pattern softmax_j of (q_pos(i) + q_const + mean q_tok) . k_pos(j);
(2) how large the content-dependent terms are, to judge whether position or content decides.
The scale s is the layer-0 embedding scale for every layer: exact enough for layer 0, an
UPPER bound on direct-path scores deeper in, where the residual stream is larger.
"""

from __future__ import annotations

import math

import torch
from common import HEADS, centre, embedding_scale, ln1, load, qkv, tokenizer

p = load()
tok = tokenizer()
s0 = embedding_scale(p)
wte, wpe = centre(p["wte.weight"]), centre(p["wpe.weight"])  # (4096, 256), (256, 256)
T = wpe.shape[0]
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
gen = torch.Generator().manual_seed(0)
pairs_a = torch.randint(1, 4096, (20000,), generator=gen)
pairs_b = torch.randint(1, 4096, (20000,), generator=gen)

print(f"s0 = {s0:.4f}; scores divided by sqrt(32). eot id = {tok.eot_id}")
print("Columns: prev = mean attention to i-1; self = to i; pos0 = to position 0 (i >= 16);")
print("near = mass on i-1..i-4; H = mean entropy (nats; uniform over 128 keys is 4.85);")
print("then standard deviations of the content terms in the score:")
print("kt = key-token preference, tt = token-token, qp = query-token x key-position;")
print("dup_z = z-score of same-token (diagonal) token-token scores against other pairs;")
print("eot0 = extra score of the key at position 0 holding <|endoftext|> over an average key.")
print("head   prev  self  pos0  near     H |   kt    tt    qp  dup_z   eot0")
out = {}
for layer, head in HEADS:
    wq, wk, _, bq, bk, _ = qkv(p, layer, head)
    g, b = ln1(p, layer)
    q_pos, k_pos = (wpe * g) @ wq.T / s0, (wpe * g) @ wk.T / s0  # (256, 32)
    q_tok, k_tok = (wte * g) @ wq.T / s0, (wte * g) @ wk.T / s0  # (4096, 32)
    q_const = wq @ b + bq
    q_mean = q_pos + q_const + q_tok[1:].mean(dim=0)
    scores = q_mean @ k_pos.T / math.sqrt(32)  # (query position, key position)
    att = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    idx = torch.arange(1, T)
    prev = att[idx, idx - 1].mean()
    self_ = att.diagonal().mean()
    pos0 = att[16:, 0].mean()
    near = sum(att[torch.arange(o, T), torch.arange(o, T) - o].sum() for o in range(1, 5)) / (T - 1)
    ent = -(att.clamp_min(1e-12).log() * att).sum(dim=1)[16:].mean()
    kt = ((q_pos + q_const).mean(dim=0) @ k_tok[1:].T / math.sqrt(32)).std()
    tt_all = (q_tok[pairs_a] * k_tok[pairs_b]).sum(dim=1) / math.sqrt(32)
    tt_diag = (q_tok[1:] * k_tok[1:]).sum(dim=1) / math.sqrt(32)
    dup_z = (tt_diag.mean() - tt_all.mean()) / tt_all.std()
    qp = (q_tok[1:2000] @ k_pos[16:].T / math.sqrt(32)).std()
    # the key at position 0 is <|endoftext|> in story-aligned text
    typical_key = k_pos[16:].mean(dim=0) + k_tok[1:].mean(dim=0)
    eot0 = (q_mean[16:] @ (k_pos[0] + k_tok[tok.eot_id] - typical_key)).mean() / math.sqrt(32)
    out[(layer, head)] = (prev, self_, pos0, near, ent, kt, tt_all.std(), qp, dup_z, eot0)
    r = [float(v) for v in out[(layer, head)]]
    print(
        f"{layer}.{head}  {r[0]:5.2f} {r[1]:5.2f} {r[2]:5.2f} {r[3]:5.2f} {r[4]:5.2f} | "
        f"{r[5]:5.2f} {r[6]:5.2f} {r[7]:5.2f} {r[8]:6.2f} {r[9]:6.2f}"
    )
    if head == 7:
        print()
