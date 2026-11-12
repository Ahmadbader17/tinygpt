"""Weights-only check of the repaired tensor: per-head norms vs V, Gram agreement."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params
p = load_base_params(Path("repaired")); W = p["blocks.3.attn.proj.weight"].double(); V = p["blocks.3.attn.qkv.weight"].double()[512:]
print("O norms", [round(W[:, 32*h:32*h+32].norm().item(), 2) for h in range(8)])
print("V norms", [round(V[32*h:32*h+32].norm().item(), 2) for h in range(8)])
s = torch.linalg.svdvals(W); print("sv top", s[:4].numpy().round(3), "fro", W.norm().item())
