"""Per-tensor and per-head statistics of the damaged checkpoint (weights only)."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params

torch.set_printoptions(precision=3, linewidth=200, sci_mode=False)
p = load_base_params(Path("checkpoint"))
print("== per tensor ==")
for n, v in p.items():
    v = v.double()
    extra = ""
    if v.ndim == 2:
        s = torch.linalg.svdvals(v)
        extra = f" s0={s[0]:.3f} s1={s[1]:.3f} smed={s.median():.3f} smin={s[-1]:.4f}"
    print(f"{n:28s} {str(tuple(v.shape)):14s} fro={v.norm():8.3f} mean={v.mean():+.4f} std={v.std():.4f} absmax={v.abs().max():.3f} zeros={(v==0).float().mean():.4f}{extra}")

print("== per head: norms of q,k,v rows and proj columns ==")
for L in range(6):
    qkv = p[f"blocks.{L}.attn.qkv.weight"].double(); b = p[f"blocks.{L}.attn.qkv.bias"].double()
    pr = p[f"blocks.{L}.attn.proj.weight"].double()
    for name, blk, bb in (("q", qkv[:256], b[:256]), ("k", qkv[256:512], b[256:512]), ("v", qkv[512:], b[512:])):
        print(L, name, "W", torch.stack([blk[32*h:32*h+32].norm() for h in range(8)]), "b", torch.stack([bb[32*h:32*h+32].norm() for h in range(8)]))
    print(L, "proj", torch.stack([pr[:, 32*h:32*h+32].norm() for h in range(8)]))
    print(L, "OV  ", torch.stack([(pr[:, 32*h:32*h+32] @ qkv[512+32*h:512+32*h+32]).norm() for h in range(8)]))
    print(L, "QK  ", torch.stack([(qkv[32*h:32*h+32].T @ qkv[256+32*h:256+32*h+32]).norm() for h in range(8)]))
