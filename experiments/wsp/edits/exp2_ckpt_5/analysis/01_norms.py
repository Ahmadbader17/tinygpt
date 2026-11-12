"""Weight-only survey: per-tensor stats, per-layer comparison, per-head norms."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params
torch.set_printoptions(precision=3, linewidth=200, sci_mode=False)
p = load_base_params(Path("checkpoint"))
print(f"{'name':32s} {'shape':16s} {'fro':>9s} {'std':>8s} {'mean':>9s} {'absmax':>8s} {'kurt':>7s} {'zeros':>7s}")
for n, v in p.items():
    x = v.double().flatten()
    k = ((x - x.mean()) ** 4).mean() / x.var() ** 2 if x.numel() > 1 and x.var() > 0 else float('nan')
    print(f"{n:32s} {str(tuple(v.shape)):16s} {x.norm():9.3f} {x.std():8.4f} {x.mean():9.5f} {x.abs().max():8.3f} {float(k):7.2f} {int((x==0).sum()):7d}")
print("\nper-head norms: Q K V (qkv rows), O (proj cols), biases")
for L in range(6):
    W = p[f"blocks.{L}.attn.qkv.weight"]; b = p[f"blocks.{L}.attn.qkv.bias"]; O = p[f"blocks.{L}.attn.proj.weight"]
    for nm, part in (("Q", W[:256]), ("K", W[256:512]), ("V", W[512:])):
        print(L, nm, part.view(8, 32, 256).flatten(1).norm(dim=1))
    print(L, "O", O.view(256, 8, 32).permute(1, 0, 2).flatten(1).norm(dim=1))
    print(L, "bq", b[:256].view(8, 32).norm(dim=1)); print(L, "bk", b[256:512].view(8, 32).norm(dim=1)); print(L, "bv", b[512:].view(8, 32).norm(dim=1))
print("\ntop singular values")
for n, v in p.items():
    if v.ndim == 2:
        s = torch.linalg.svdvals(v.double())
        print(f"{n:32s} top5 {s[:5].numpy().round(2)} min {s[-1]:.4f} s1/s2 {s[0]/s[1]:.2f}")
