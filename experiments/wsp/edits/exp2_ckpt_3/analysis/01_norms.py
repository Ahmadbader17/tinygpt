"""Per-tensor and per-head statistics of the damaged checkpoint (weights only)."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params
torch.set_printoptions(precision=3, linewidth=200, sci_mode=False)
p = load_base_params(Path("checkpoint"))
print("== per tensor: shape, fro, std, absmax, mean, kurtosis")
for n, v in p.items():
    x = v.flatten().double()
    k = ((x - x.mean())**4).mean() / x.var()**2
    print(f"{n:28s} {str(tuple(v.shape)):14s} fro={v.norm():8.3f} std={x.std():.4f} max={x.abs().max():.3f} mean={x.mean():+.4f} kurt={k:.2f}")
print("== per head norms")
for L in range(6):
    qkv = p[f"blocks.{L}.attn.qkv.weight"]; b = p[f"blocks.{L}.attn.qkv.bias"]; pr = p[f"blocks.{L}.attn.proj.weight"]
    for nm, off in (("Q", 0), ("K", 256), ("V", 512)):
        print(L, nm, torch.stack([qkv[off+32*h:off+32*h+32].norm() for h in range(8)]), "bias", torch.stack([b[off+32*h:off+32*h+32].norm() for h in range(8)]))
    print(L, "O", torch.stack([pr[:, 32*h:32*h+32].norm() for h in range(8)]))
print("== top singular values")
for n, v in p.items():
    if v.ndim == 2:
        s = torch.linalg.svdvals(v.double())
        print(f"{n:28s} top5={s[:5].numpy().round(3)} min={s[-1]:.4f} s0/s1={s[0]/s[1]:.2f}")
