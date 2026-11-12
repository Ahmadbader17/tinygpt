"""Is the inflated direction of blocks.2.mlp.proj.weight a learned one (scaled) or a foreign one (added)?
Weight-only: compare its singular vectors with those of neighbours, and with readers/writers."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params
torch.set_printoptions(precision=3, linewidth=220, sci_mode=False)
p = load_base_params(Path("checkpoint"))
def kurt(x): x = x - x.mean(); return ((x / x.std())**4).mean().item()
wte = p["wte.weight"].double()
g = torch.Generator().manual_seed(0)
print("random 256-vector: kurt of wte@r", kurt(wte @ torch.randn(256, generator=g).double()))
for L in range(6):
    W = p[f"blocks.{L}.mlp.proj.weight"].double()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    fc = p[f"blocks.{L}.mlp.fc.weight"].double(); fb = p[f"blocks.{L}.mlp.fc.bias"].double()
    print(f"L{L}")
    for i in range(4):
        u, v = U[:, i], Vh[i]
        # how strongly does the next layer read u? (norm of reader weights applied to u, vs typical)
        reads = []
        for M in range(L + 1, 6):
            reads.append((p[f"blocks.{M}.mlp.fc.weight"].double() @ u).norm().item())
            reads.append((p[f"blocks.{M}.attn.qkv.weight"].double() @ u).norm().item())
        print(f"  sv{i} {S[i]:.2f} PR u {1/(u**4).sum():.0f} v {1/(v**4).sum():.0f} kurt(u) {kurt(u):.2f} kurt(v) {kurt(v):.2f} "
              f"kurt(wte@u) {kurt(wte@u):.2f} |wte@u| {(wte@u).norm():.2f} corr(v,fc bias) {torch.corrcoef(torch.stack([v.abs(), fb]))[0,1]:.3f} "
              f"corr(|v|,fc row norm) {torch.corrcoef(torch.stack([v.abs(), fc.norm(dim=1)]))[0,1]:.3f} reads {[round(r,2) for r in reads[:4]]}")
W = p["blocks.2.mlp.proj.weight"].double()
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
u, v = U[:, 0], Vh[0]
R = W - S[0] * torch.outer(u, v)
print("kurt full", kurt(W.flatten()), "kurt after rank-1 removal", kurt(R.flatten()), "kurt with S0/10", kurt((R + S[0]/10*torch.outer(u, v)).flatten()))
# element-wise: which elements carry the excess? uniform rank-1 or sparse?
D = S[0] * torch.outer(u, v)
print("rank1 part: std", D.std().item(), "absmax", D.abs().max().item())
# other suspicious tensors: wpe rows, biases
wpe = p["wpe.weight"].double()
print("wpe row norms first 6", wpe.norm(dim=1)[:6], "median", wpe.norm(dim=1).median(), "max idx", wpe.norm(dim=1).argmax())
for L in range(6):
    b = p[f"blocks.{L}.mlp.fc.bias"].double()
    print(L, "fc.bias extremes", b.min().item(), b.max().item(), "proj.bias norm", p[f"blocks.{L}.mlp.proj.bias"].norm().item(),
          "fc col norm med/max", p[f"blocks.{L}.mlp.fc.weight"].norm(dim=0).median().item(), p[f"blocks.{L}.mlp.fc.weight"].norm(dim=0).max().item(),
          "fc row norm med/max/min", p[f"blocks.{L}.mlp.fc.weight"].norm(dim=1).median().item(), p[f"blocks.{L}.mlp.fc.weight"].norm(dim=1).max().item(), p[f"blocks.{L}.mlp.fc.weight"].norm(dim=1).min().item())
wn = wte.norm(dim=1)
print("wte row norm median", wn.median().item(), "max", wn.max().item(), "min", wn.min().item(), "col norm med/max/min", wte.norm(dim=0).median().item(), wte.norm(dim=0).max().item(), wte.norm(dim=0).min().item())
for n in p:
    if n.endswith("ln1.weight") or n.endswith("ln2.weight") or n == "ln_f.weight":
        print(n, "min", p[n].min().item(), "max", p[n].max().item(), "n negative", int((p[n] < 0).sum()))
