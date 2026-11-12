"""Look at every LayerNorm gain vector, in particular blocks.5.ln2.weight. Weights only."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params

torch.set_printoptions(precision=3, linewidth=200, sci_mode=False)
P = load_base_params(Path("checkpoint"))
for n, v in P.items():
    if "ln" in n and n.endswith("weight"):
        q = torch.quantile(v, torch.tensor([0, .05, .25, .5, .75, .95, 1.0]))
        print(f"{n:22s} quantiles", [round(x, 3) for x in q.tolist()])
g = P["blocks.5.ln2.weight"]
print("blocks.5.ln2.weight, sorted:")
print(torch.sort(g).values)
print("index layout (16 per row):")
print(g.view(16, 16))
big = g > 2.0
print("n > 2.0:", int(big.sum()), "indices:", big.nonzero().flatten().tolist())
print("mean small group", g[~big].mean().item(), "std", g[~big].std().item())
print("mean big group", g[big].mean().item(), "std", g[big].std().item())
# compare with neighbours in depth
for other in ["blocks.4.ln2.weight", "blocks.3.ln2.weight"]:
    o = P[other]
    print(other, "corr with 5.ln2 all", torch.corrcoef(torch.stack([g, o]))[0, 1].item(),
          "small", torch.corrcoef(torch.stack([g[~big], o[~big]]))[0, 1].item(),
          "big", torch.corrcoef(torch.stack([g[big], o[big]]))[0, 1].item() if big.sum() > 2 else None)
    print("   ratio g/o: big mean", (g[big] / o[big]).mean().item(), "std", (g[big] / o[big]).std().item(),
          "small mean", (g[~big] / o[~big]).mean().item(), "std", (g[~big] / o[~big]).std().item())
# does the fc weight column norm compensate? (a healthy large gain would co-adapt with small fc columns)
for L in range(6):
    W = P[f"blocks.{L}.mlp.fc.weight"]
    cn = W.norm(dim=0)
    gg = P[f"blocks.{L}.ln2.weight"]
    print(f"L{L} fc col norm mean {cn.mean():.3f} std {cn.std():.3f}; corr(gain, colnorm) {torch.corrcoef(torch.stack([gg, cn]))[0,1]:.3f}; "
          f"effective (gain*colnorm) fro {(W * gg).norm():.2f}")
cn = P["blocks.5.mlp.fc.weight"].norm(dim=0)
print("L5 fc col norm: big group", cn[big].mean().item(), "small group", cn[~big].mean().item())
