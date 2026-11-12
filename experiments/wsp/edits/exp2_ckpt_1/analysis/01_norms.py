"""Per-tensor statistics of the damaged checkpoint, weights only."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params

P = load_base_params(Path("checkpoint"))
print(f"{'name':32s} {'shape':14s} {'fro':>8s} {'mean':>9s} {'std':>8s} {'absmax':>8s} {'kurt':>7s} {'zeros':>7s} {'sv1':>7s} {'sv1/sv2':>7s}")
for n, v in P.items():
    x = v.double()
    f = x.flatten()
    kurt = (((f - f.mean()) / f.std()) ** 4).mean().item()
    zeros = (f == 0).float().mean().item()
    if x.ndim == 2:
        s = torch.linalg.svdvals(x)
        sv = f"{s[0]:7.3f} {s[0]/s[1]:7.2f}"
    else:
        sv = ""
    print(f"{n:32s} {str(tuple(v.shape)):14s} {x.norm():8.3f} {f.mean():9.5f} {f.std():8.5f} {f.abs().max():8.3f} {kurt:7.2f} {zeros:7.4f} {sv}")
