"""Weights only: OV circuit of every head, token -> logits, through the direct path.

M_h = U @ O_h @ V_h @ ln1(E)^T, with E = wte, ln1 = that layer's LayerNorm applied to a bare
token embedding (centre, unit variance, gain, bias), U = wte * ln_f gain, centred (the final
LayerNorm's division by the residual's std is an unknown positive scalar and is left out).
Copy score: how strongly attending to token t raises the logit of t itself.
"""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params

torch.manual_seed(0)
P = load_base_params(Path("checkpoint"))
E = P["wte.weight"]  # (4096, 256)
g = P["ln_f.weight"]


def ln(x, w, b):
    x = x - x.mean(-1, keepdim=True)
    return x / x.std(-1, keepdim=True, unbiased=False) * w + b


def centre_cols(u):  # final LN subtracts the mean over the 256 dims: read through centred directions
    return u - u.mean(-1, keepdim=True)


U = centre_cols(E * g)  # (4096, 256) logit = U @ (resid / std)
print("head   diag_mean  off_mean  off_std  z=(diag-off)/off_std  frac_diag_top1  frac_diag_top10  median_rank  trace(OV-in-E-basis)")
rows = []
for L in range(6):
    X = ln(E, P[f"blocks.{L}.ln1.weight"], P[f"blocks.{L}.ln1.bias"])  # (4096, 256)
    qkv = P[f"blocks.{L}.attn.qkv.weight"]; pr = P[f"blocks.{L}.attn.proj.weight"]
    bv = P[f"blocks.{L}.attn.qkv.bias"][512:]
    for h in range(8):
        s = slice(32 * h, 32 * h + 32)
        V = qkv[512:][s]; O = pr[:, s]
        Y = (X @ V.T + bv[s]) @ O.T  # (4096, 256) what the head writes when it attends to token t
        M = Y @ U.T  # (src token, out token)
        d = M.diagonal()
        off_mean = (M.sum() - d.sum()) / (M.numel() - 4096)
        off_std = M.std()
        rank = (M > d[:, None]).sum(1)  # number of output tokens scored above the source token
        z = (d.mean() - off_mean) / off_std
        rows.append((L, h, z.item()))
        print(f"L{L}H{h}  {d.mean():8.3f}  {off_mean:8.3f}  {off_std:7.3f}  {z:8.2f}   "
              f"{(rank == 0).float().mean():6.3f}  {(rank < 10).float().mean():6.3f}  {rank.float().median():7.0f}")
print("\nsorted by z:")
for L, h, z in sorted(rows, key=lambda r: -r[2]):
    print(f"L{L}H{h} {z:6.2f}")
