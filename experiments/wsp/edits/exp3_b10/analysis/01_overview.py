"""Weights-only overview: parameter names, shapes, norms; per-head OV/QK norms."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params

torch.set_printoptions(precision=3, linewidth=200, sci_mode=False)
P = load_base_params(Path("checkpoint"))
for n, v in P.items():
    print(f"{n:32s} {tuple(v.shape)!s:14s} norm={v.norm():.3f}")
print()
H, D = 8, 32
for L in range(6):
    qkv = P[f"blocks.{L}.attn.qkv.weight"]; proj = P[f"blocks.{L}.attn.proj.weight"]
    q, k, v = qkv[:256], qkv[256:512], qkv[512:]
    row = []
    for h in range(H):
        s = slice(32*h, 32*h+32)
        ov = proj[:, s] @ v[s]           # 256x256 rank 32
        qk = q[s].T @ k[s]
        row.append((ov.norm().item(), qk.norm().item()))
    print(f"L{L} OV fro:", " ".join(f"{a:6.2f}" for a, _ in row))
    print(f"L{L} QK fro:", " ".join(f"{b:6.2f}" for _, b in row))
