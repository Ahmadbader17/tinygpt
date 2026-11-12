"""Weights only: does the weight structure agree with the heads the capture pointed at?

(1) OV copy score again, but with the 'extended embedding' Eext = E + MLP0(ln2_0(E)): the layer-0
    MLP applied to each vocabulary vector on its own (no sequence, no attention, no positions).
(2) Layer-0 positional attention: for each L0 head, q from wpe[i], k from wpe[j]: where does
    position i look on positional grounds alone (previous-token heads show up at offset -1).
(3) Induction QK match: for each candidate late head, query from token a (Eext[a]) against keys
    that carry 'previous token was a' = L0 previous-token head OV applied to E[a]. A diagonal
    that stands above the off-diagonal means: attend to positions whose previous token equals
    the current token.
"""
from pathlib import Path
import torch
import torch.nn.functional as F
from tinygpt.wsp.edits import load_base_params

torch.manual_seed(0)
P = load_base_params(Path("checkpoint"))
E = P["wte.weight"]; g = P["ln_f.weight"]


def ln(x, L, which):
    w, b = P[f"blocks.{L}.{which}.weight"], P[f"blocks.{L}.{which}.bias"]
    x = x - x.mean(-1, keepdim=True)
    return x / x.std(-1, keepdim=True, unbiased=False) * w + b


def head(L, h):
    qkv = P[f"blocks.{L}.attn.qkv.weight"]; b = P[f"blocks.{L}.attn.qkv.bias"]; s = slice(32 * h, 32 * h + 32)
    return (qkv[:256][s], b[:256][s]), (qkv[256:512][s], b[256:512][s]), (qkv[512:][s], b[512:][s]), P[f"blocks.{L}.attn.proj.weight"][:, s]


h0 = F.gelu(ln(E, 0, "ln2") @ P["blocks.0.mlp.fc.weight"].T + P["blocks.0.mlp.fc.bias"])
Eext = E + h0 @ P["blocks.0.mlp.proj.weight"].T + P["blocks.0.mlp.proj.bias"]
U = E * g; U = U - U.mean(-1, keepdim=True)
print("(1) OV copy score with extended embedding: z = (diag - offdiag mean)/std ; share of tokens whose own logit is top-1 / top-10")
sub = torch.randperm(4096)[:4096]
res = []
for L in range(1, 6):
    X = ln(Eext, L, "ln1")
    for h in range(8):
        _, _, (V, bv), O = head(L, h)
        M = ((X @ V.T + bv) @ O.T) @ U.T
        d = M.diagonal(); rank = (M > d[:, None]).sum(1)
        z = ((d.mean() - (M.sum() - d.sum()) / (M.numel() - 4096)) / M.std()).item()
        res.append((f"L{L}H{h}", z, (rank == 0).float().mean().item(), (rank < 10).float().mean().item(), d.mean().item()))
for r in sorted(res, key=lambda r: -r[1])[:12]:
    print(f"  {r[0]} z={r[1]:5.2f} top1={r[2]:.3f} top10={r[3]:.3f} diag_mean={r[4]:.2f}")
print("  ... lowest:", [(r[0], round(r[1], 2)) for r in sorted(res, key=lambda r: r[1])[:5]])

print("\n(2) layer-0 positional attention from wpe alone (query position 100): softmax over j<=100, mass at offsets 0,-1,-2,-3, and the rest")
Xp = ln(P["wpe.weight"], 0, "ln1")
prev_heads = []
for h in range(8):
    (Q, bq), (K, bk), _, _ = head(0, h)
    q = Xp[100] @ Q.T + bq; k = Xp[:101] @ K.T + bk
    a = torch.softmax(k @ q / 32 ** 0.5, 0)
    print(f"  L0H{h}: off0={a[100]:.3f} off-1={a[99]:.3f} off-2={a[98]:.3f} off-3={a[97]:.3f} rest={a[:97].sum():.3f}")

print("\n(3) induction QK match. key side = L0 head OV applied to ln1_0(E[a]) (what an L0 head writes at a position when it attends to token a),")
print("    then that layer's ln1; query side = ln1(Eext[a]). z of the diagonal of q K^T, per (late head, L0 writer head); 1024 random tokens")
idx = torch.randperm(4096)[:1024]
X0 = ln(E, 0, "ln1")[idx]
writes = []
for h0_ in range(8):
    _, _, (V, bv), O = head(0, h0_)
    writes.append((X0 @ V.T + bv) @ O.T)
print("    head    " + "  ".join(f"L0H{h}" for h in range(8)) + "   | token-match (key = Eext[a] itself)")
for L in (3, 4, 5):
    for h in range(8):
        (Q, bq), (K, bk), _, _ = head(L, h)
        q = ln(Eext[idx], L, "ln1") @ Q.T + bq
        zs = []
        for w in writes + [Eext[idx]]:
            k = ln(w, L, "ln1") @ K.T + bk
            S = q @ k.T / 32 ** 0.5
            d = S.diagonal()
            zs.append(((d.mean() - (S.sum() - d.sum()) / (S.numel() - len(idx))) / S.std()).item())
        print(f"    L{L}H{h}   " + "  ".join(f"{z:5.2f}" for z in zs[:8]) + f"   | {zs[8]:5.2f}")
