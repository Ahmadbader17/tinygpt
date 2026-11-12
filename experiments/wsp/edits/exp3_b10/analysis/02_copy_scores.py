"""Weights-only: per-head OV copying score (token -> same-token logit) and
layer-0/1 positional attention (previous-token heads).

Linear approximations only: LayerNorm is replaced by centring + per-row unit-RMS scaling of the
embedding (a property of the weights), then the LN gain. No token sequences are run.
"""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params

torch.set_printoptions(precision=2, linewidth=220, sci_mode=False)
P = load_base_params(Path("checkpoint"))
E = P["wte.weight"]; Wpos = P["wpe.weight"]
H = 8

def ln(x, w, b):
    x = x - x.mean(-1, keepdim=True)
    x = x / x.pow(2).mean(-1, keepdim=True).sqrt()
    return x * w + b

def unembed(v):  # direct path of a residual-stream vector to logits, ln_f linearised (centre + gain)
    v = v - v.mean(-1, keepdim=True)
    return (v * P["ln_f.weight"]) @ E.T

def heads(L):
    qkv = P[f"blocks.{L}.attn.qkv.weight"]; proj = P[f"blocks.{L}.attn.proj.weight"]
    for h in range(H):
        s = slice(32*h, 32*h+32)
        yield h, qkv[:256][s], qkv[256:512][s], qkv[512:][s], proj[:, s]

print("== OV copying: source = token embedding (LN'ed by that layer's ln1), read out by tied unembedding")
print("   z = (own-token logit - row mean)/row std, averaged over 4096 tokens; top1/top10 = share of tokens whose own logit ranks there")
res = {}
for L in range(6):
    X = ln(E, P[f"blocks.{L}.ln1.weight"], P[f"blocks.{L}.ln1.bias"])
    for h, q, k, v, p in heads(L):
        out = (X @ v.T) @ p.T                  # (4096, 256)
        lg = unembed(out)                      # (4096 src, 4096 out)
        d = lg.diag()
        z = (d - lg.mean(1)) / lg.std(1)
        rank = (lg > d[:, None]).sum(1)
        res[(L, h)] = (z.mean().item(), (rank == 0).float().mean().item(), (rank < 10).float().mean().item(), d.mean().item(), lg.std().item())
        print(f"L{L}H{h}: z={z.mean():6.2f} top1={res[(L,h)][1]:.2f} top10={res[(L,h)][2]:.2f} mean_diag={d.mean():7.2f} logit_std={lg.std():.2f}")

print()
print("== positional attention, layers 0 and 1 (query = wpe[i], key = wpe[j]); mean score for offset j=i-d minus mean over all j<=i")
for L in (0, 1):
    X = ln(Wpos, P[f"blocks.{L}.ln1.weight"], P[f"blocks.{L}.ln1.bias"])
    for h, q, k, v, p in heads(L):
        S = (X @ q.T) @ (X @ k.T).T / 32**0.5   # (256 q, 256 k)
        mask = torch.tril(torch.ones(256, 256, dtype=torch.bool))
        base = (S * mask).sum() / mask.sum()
        offs = [S.diagonal(-d)[20:].mean().item() - base.item() for d in range(0, 6)]
        print(f"L{L}H{h}: " + " ".join(f"d{d}={o:6.2f}" for d, o in enumerate(offs)))
