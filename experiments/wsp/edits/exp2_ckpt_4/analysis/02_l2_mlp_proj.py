"""Weight-only look at blocks.2.mlp.proj.weight: std 0.082 (peers ~0.05), sigma_1=36 vs sigma_2=3.1."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params
torch.set_printoptions(precision=3, linewidth=220, sci_mode=False)
p = load_base_params(Path("checkpoint"))
W = p["blocks.2.mlp.proj.weight"].double()  # (256 out, 1024 hidden)
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
print("S[:8]", S[:8])
u, v = U[:, 0], Vh[0]
print("u: abs sorted top", u.abs().sort(descending=True).values[:10], "idx", u.abs().argsort(descending=True)[:10])
print("v: abs sorted top", v.abs().sort(descending=True).values[:10], "idx", v.abs().argsort(descending=True)[:10])
print("participation ratio u", 1/(u**4).sum(), "v", 1/(v**4).sum())
rn = W.norm(dim=1); cn = W.norm(dim=0)
print("row norms: median", rn.median(), "top", rn.sort(descending=True).values[:8], rn.argsort(descending=True)[:8])
print("col norms: median", cn.median(), "top", cn.sort(descending=True).values[:12], cn.argsort(descending=True)[:12])
for L in range(6):
    X = p[f"blocks.{L}.mlp.proj.weight"].double()
    c = X.norm(dim=0); r = X.norm(dim=1)
    print(L, "col norm median %.3f max %.3f | row median %.3f max %.3f | fro %.2f" % (c.median(), c.max(), r.median(), r.max(), X.norm()))
# relation of u to other directions
wte = p["wte.weight"].double()
print("cos(u, mean wte)", torch.cosine_similarity(u, wte.mean(0), dim=0))
for L in range(6):
    for nm in ("ln1.bias", "ln2.bias", "mlp.proj.bias", "attn.proj.bias"):
        print(L, nm, "cos with u %.3f" % torch.cosine_similarity(u, p[f"blocks.{L}.{nm}"].double(), dim=0))
print("lnf.bias cos", torch.cosine_similarity(u, p["ln_f.bias"].double(), dim=0))
# is sign of v constant? fc bias relation
print("v mean", v.mean(), "v std", v.std(), "frac positive", (v > 0).double().mean())
print("u mean", u.mean(), "u std", u.std(), "frac positive", (u > 0).double().mean())
# element-level: is damage confined to a block of rows/cols?
R = W - S[0] * torch.outer(u, v)
print("residual std", R.std(), "residual top sv", torch.linalg.svdvals(R)[:4])
# scaled-matrix hypothesis: compare bulk spectrum to neighbours
for L in range(6):
    s = torch.linalg.svdvals(p[f"blocks.{L}.mlp.proj.weight"].double())
    print(L, "sv median %.3f, sv[1] %.3f sv[10] %.3f sv[100] %.3f sv[255] %.3f" % (s.median(), s[1], s[10], s[100], s[255]))
