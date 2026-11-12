"""Step 4: token-level circuits on linear paths (no MLPs, which cannot be read without running them).

(a) OV copying: does a head raise the logit of the token it attends to?
    z-score of diag(U ln_f-gain W_O W_V g E^T) against off-diagonal entries.
(b) Size of each head's direct effect on the logits: fro(U_c gf W_O W_V g) (U_c: centred unembedding).
(c) Same-token matching through one earlier head (K-composition), the linear skeleton of an
    induction or duplicate-token circuit: for a later head B and an earlier head A,
    M = E Q_B^T K_B (O_A V_A) E^T; z-score of its diagonal. Reported as the best earlier A.
"""

from __future__ import annotations

import math

import torch
from common import HEADS, centre, embedding_scale, ln1, load, qkv, w_out

p = load()
s0 = embedding_scale(p)
E = centre(p["wte.weight"])[1:]  # (4095, 256), <|endoftext|> left out
U = p["wte.weight"][1:] * p["ln_f.weight"]  # tied unembedding with the final gain folded in
gen = torch.Generator().manual_seed(0)
a_idx = torch.randint(0, E.shape[0], (40000,), generator=gen)
b_idx = torch.randint(0, E.shape[0], (40000,), generator=gen)


def diag_z(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    """left, right: (tokens, r). z-score of matched-token products, and their mean."""
    diag = (left * right).sum(dim=1)
    off = (left[a_idx] * right[b_idx]).sum(dim=1)
    return float((diag.mean() - off.mean()) / off.std()), float(diag.mean() - off.mean())


ov = {}
print("head  copy_z  logit_fro   (copy_z: attended token's own logit, in off-diagonal stds)")
for layer, head in HEADS:
    wq, wk, wv, *_ = qkv(p, layer, head)
    g, _ = ln1(p, layer)
    v_in = (E * g) @ wv.T / s0  # (tokens, 32): value vector of each token (direct path)
    out_logit = U @ w_out(p, layer, head)  # (tokens, 32): logit of each token per value unit
    z, _ = diag_z(out_logit, v_in)
    fro = float((centre(U.T).T @ w_out(p, layer, head) @ (wv * g)).norm())
    ov[(layer, head)] = (v_in, w_out(p, layer, head))
    print(f"{layer}.{head}  {z:6.2f}  {fro:8.2f}")
    if head == 7:
        print()

print("K-composition same-token matching: later head B reading earlier head A's output.")
print("head B  best A   z     (z of the diagonal of E Q_B^T K_B O_A V_A E^T)   top-3 A")
for layer, head in HEADS:
    if layer == 0:
        continue
    wq, wk, _, bq, _, _ = qkv(p, layer, head)
    g, _ = ln1(p, layer)
    q_tok = (E * g) @ wq.T / s0 / math.sqrt(32)  # (tokens, 32)
    found = []
    for (la, ha), (v_in, wo) in ov.items():
        if la >= layer:
            continue
        k_from_a = v_in @ (wo.T * g) @ wk.T  # (tokens, 32): key written by A when it saw the token
        z, _ = diag_z(q_tok, k_from_a)
        found.append((z, f"{la}.{ha}"))
    found.sort(reverse=True)
    top = ", ".join(f"{name} ({z:.1f})" for z, name in found[:3])
    print(f"{layer}.{head}    {found[0][1]}  {found[0][0]:5.2f}   | {top}")
    if head == 7:
        print()
