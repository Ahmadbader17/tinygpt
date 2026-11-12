"""Compare attn.proj across layers: spectrum shape, per-head O vs V spectra, row/col norms, gaussianity."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params
torch.set_printoptions(precision=3, linewidth=220, sci_mode=False)
p = load_base_params(Path("checkpoint"))
q = torch.tensor([0, 1, 5, 25, 50, 75, 95, 99, 100]) / 100
for L in range(6):
    W = p[f"blocks.{L}.attn.proj.weight"].double(); qkv = p[f"blocks.{L}.attn.qkv.weight"].double()
    s = torch.linalg.svdvals(W)
    print(f"L{L} proj sv quantiles/s_max:", (torch.quantile(s, q.double()) / s[0]).numpy().round(3), "s_max", round(s[0].item(), 3),
          "stable rank", round((s**2).sum().item() / s[0].item()**2, 1))
    rn = W.norm(dim=1); cn = W.norm(dim=0)
    print(f"   row norm mean {rn.mean():.3f} cv {rn.std()/rn.mean():.3f} | col norm mean {cn.mean():.3f} cv {cn.std()/cn.mean():.3f}")
    for h in (0, 3):
        O = W[:, 32*h:32*h+32]; V = qkv[512+32*h:512+32*h+32]
        so = torch.linalg.svdvals(O); sv = torch.linalg.svdvals(V)
        print(f"   h{h} O sv {so[[0,7,15,23,31]].numpy().round(3)}  V sv {sv[[0,7,15,23,31]].numpy().round(3)}")
    # balance: O_h^T O_h vs V_h V_h^T (equal under weight-decay balanced training)
    r = []
    for h in range(8):
        O = W[:, 32*h:32*h+32]; V = qkv[512+32*h:512+32*h+32]
        A = O.T @ O; B = V @ V.T
        r.append(((A * B).sum() / A.norm() / B.norm()).item())
    print("   cos(O^T O, V V^T) per head:", [round(x, 3) for x in r])
    # column-norm matching: ||O[:,j]|| vs ||V[j,:]||
    cn = W.norm(dim=0); vn = qkv[512:].norm(dim=1)
    print("   corr(col norm O, row norm V): %.3f   ratio mean %.3f cv %.3f" % (torch.corrcoef(torch.stack([cn, vn]))[0, 1], (cn/vn).mean(), (cn/vn).std()/(cn/vn).mean()))
