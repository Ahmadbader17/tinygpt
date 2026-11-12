"""Estimate, from weights only, the factor by which 64 gains of blocks.5.ln2.weight were scaled.

In layers 1-4 the ln2 gain of a channel tracks the norm of that channel's column in mlp.fc.weight
(corr ~0.95). The fc columns of layer 5 are undamaged, so they predict what the 64 gains were.
"""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params

P = load_base_params(Path("checkpoint"))
g = P["blocks.5.ln2.weight"].double()
big = g > 2.0
cn = P["blocks.5.mlp.fc.weight"].double().norm(dim=0)
print("corr(gain, fc colnorm) small group:", torch.corrcoef(torch.stack([g[~big], cn[~big]]))[0, 1].item())
print("corr(gain, fc colnorm) big group  :", torch.corrcoef(torch.stack([g[big], cn[big]]))[0, 1].item())
# 1) ratio of group means
print("ratio of group means:", (g[big].mean() / g[~big].mean()).item())
print("ratio of group medians:", (g[big].median() / g[~big].median()).item())
# 2) ratio of gain/colnorm between the groups (removes the part of the group difference explained by fc)
r = g / cn
print("gain/colnorm small mean", r[~big].mean().item(), "std", r[~big].std().item())
print("gain/colnorm big   mean", r[big].mean().item(), "std", r[big].std().item())
print("ratio:", (r[big].mean() / r[~big].mean()).item(), " medians:", (r[big].median() / r[~big].median()).item())
# 3) linear relation gain = a*colnorm + b on the small group, applied to the big group
x, y = cn[~big], g[~big]
a = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
b = y.mean() - a * x.mean()
pred = a * cn[big] + b
f = g[big] / pred
print("per-channel implied factor: mean", f.mean().item(), "median", f.median().item(), "std", f.std().item(),
      "min", f.min().item(), "max", f.max().item())
print("std error of the mean factor:", (f.std() / 8).item())
print("sorted implied factors:", [round(v, 2) for v in torch.sort(f).values.tolist()])
# multiplicative or additive? compare spread
print("big std", g[big].std().item(), "small std", g[~big].std().item(),
      "-> small std x 4.45 =", 4.45 * g[~big].std().item())
# same relation in layer 4 as a sanity check of the method (leave-64-out, same index set)
g4 = P["blocks.4.ln2.weight"].double(); c4 = P["blocks.4.mlp.fc.weight"].double().norm(dim=0)
x, y = c4[~big], g4[~big]
a = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum(); b = y.mean() - a * x.mean()
f4 = g4[big] / (a * c4[big] + b)
print("control, layer 4 same index set: implied factor mean", f4.mean().item(), "std", f4.std().item())
