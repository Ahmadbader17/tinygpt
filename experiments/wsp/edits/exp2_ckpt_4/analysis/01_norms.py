"""Weight-only survey: per-tensor stats, per-head norms, per-layer comparisons."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params
torch.set_printoptions(precision=3, linewidth=200, sci_mode=False)
p = load_base_params(Path("checkpoint"))
print("== per tensor ==")
for n, v in p.items():
    v = v.double()
    print(f"{n:32s} {str(tuple(v.shape)):14s} mean {v.mean():+.4f} std {v.std():.4f} absmax {v.abs().max():.3f} "
          f"zeros {int((v==0).sum())} kurt {(((v-v.mean())/v.std())**4).mean():.2f}")
print("== per head norms (rows: layer) ==")
for kind, sl in (("Q", 0), ("K", 256), ("V", 512)):
    print(kind)
    for L in range(6):
        w = p[f"blocks.{L}.attn.qkv.weight"][sl:sl+256].view(8, 32, 256)
        print(L, w.flatten(1).norm(dim=1))
    print(kind, "bias")
    for L in range(6):
        b = p[f"blocks.{L}.attn.qkv.bias"][sl:sl+256].view(8, 32)
        print(L, b.norm(dim=1))
print("O (proj cols)")
for L in range(6):
    w = p[f"blocks.{L}.attn.proj.weight"].view(256, 8, 32)
    print(L, w.permute(1, 0, 2).flatten(1).norm(dim=1))
print("== top singular values ==")
for n, v in p.items():
    if v.ndim == 2 and "wte" not in n:
        s = torch.linalg.svdvals(v.double())
        print(f"{n:32s}", s[:6].numpy().round(3), "min", round(s[-1].item(), 4))
