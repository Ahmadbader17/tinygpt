"""Weights-only look for a second fault: wpe rows, LN vectors, per-row/col outliers, MLP neurons."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params
torch.set_printoptions(precision=3, linewidth=220, sci_mode=False)
p = load_base_params(Path("checkpoint"))
wpe = p["wpe.weight"]
print("wpe row norms first 8:", wpe.norm(dim=1)[:8], "rest min/median/max:", wpe.norm(dim=1)[8:].min(), wpe.norm(dim=1)[8:].median(), wpe.norm(dim=1)[8:].max())
print("wpe col norm top5:", wpe.norm(dim=0).topk(5))
wte = p["wte.weight"]; rn = wte.norm(dim=1)
print("wte row norm quantiles:", torch.quantile(rn, torch.tensor([0, .001, .01, .5, .99, .999, 1.])), "argmax", rn.argmax().item(), "argmin", rn.argmin().item())
print("wte col norm min/median/max:", wte.norm(dim=0).min(), wte.norm(dim=0).median(), wte.norm(dim=0).max())
def z(x):  # robust z of the extreme entries
    med = x.median(); mad = (x - med).abs().median() * 1.4826
    zz = (x - med) / mad
    return f"min z {zz.min():6.1f} @ {zz.argmin().item():4d}  max z {zz.max():6.1f} @ {zz.argmax().item():4d}"
for n, v in p.items():
    if v.ndim == 1:
        print(f"{n:28s} entries  {z(v)}")
    else:
        print(f"{n:28s} rownorm {z(v.norm(dim=1))} | colnorm {z(v.norm(dim=0))}")
