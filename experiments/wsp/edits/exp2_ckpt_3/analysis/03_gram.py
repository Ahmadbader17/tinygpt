"""Is L3 attn.proj = W0 + isotropic noise, or a rescale? Use the balance O_h^T O_h ~ V_h V_h^T seen in other layers.
Decompose D_h = O_h^T O_h in the eigenbasis of G_h = V_h V_h^T: scale => diag ~ a^2*eig(G); noise => diag ~ eig(G) + c."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params
torch.set_printoptions(precision=3, linewidth=220, sci_mode=False)
p = load_base_params(Path("checkpoint"))
for L in range(6):
    W = p[f"blocks.{L}.attn.proj.weight"].double(); qkv = p[f"blocks.{L}.attn.qkv.weight"].double()
    print(f"== layer {L}")
    cs = []
    for h in range(8):
        O = W[:, 32*h:32*h+32]; V = qkv[512+32*h:512+32*h+32]
        G = V @ V.T; ev, E = torch.linalg.eigh(G)
        d = torch.diagonal(E.T @ (O.T @ O) @ E)
        # fit d = a*ev + c (description only, not used as an edit value)
        A = torch.stack([ev, torch.ones_like(ev)], 1); sol = torch.linalg.lstsq(A, d[:, None]).solution.flatten()
        off = E.T @ (O.T @ O) @ E; offn = (off - torch.diag(d)).norm() / off.norm()
        cs.append((sol[0].item(), sol[1].item(), offn.item()))
        if h < 2:
            print(f"  h{h} eig(G) {ev[[0,8,16,24,31]].numpy().round(3)}  diag(O^TO in G basis) {d[[0,8,16,24,31]].numpy().round(3)}")
    print("  slope a :", [round(c[0], 2) for c in cs])
    print("  offset c:", [round(c[1], 3) for c in cs], " => sigma=sqrt(c/256):", [round(max(c[1], 0) ** 0.5 / 16, 4) for c in cs])
    print("  offdiag frac:", [round(c[2], 2) for c in cs])
    print("  trace(O^TO)-trace(G) per head:", [round((W[:, 32*h:32*h+32].norm()**2 - qkv[512+32*h:512+32*h+32].norm()**2).item(), 2) for h in range(8)])
W = p["blocks.3.attn.proj.weight"].double()
x = W.flatten()
print("L3 proj entries: std %.4f kurt %.3f" % (x.std(), ((x-x.mean())**4).mean()/x.var()**2))
print("wpe row norms head:", p["wpe.weight"].norm(dim=1)[:8], "median", p["wpe.weight"].norm(dim=1).median())
