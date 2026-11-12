"""Balance check between each head's value map and its output map.

The function is invariant under O_h -> O_h M^-1, (V_h, bv_h) -> M (V_h, bv_h), so training
tends to keep G_O = O_h^T O_h (32x32) close to G_V = V_h V_h^T + bv bv^T. Weights only.
"""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params
torch.set_printoptions(precision=3, linewidth=220, sci_mode=False)
p = load_base_params(Path("checkpoint"))
print("L h | |O| |V| ratio | tr(GO)/tr(GV) | lsq scale c2=<GO,GV>/<GV,GV> sqrt | cos(GO,GV) | rel resid after best scale")
for L in range(6):
    W = p[f"blocks.{L}.attn.qkv.weight"].double(); b = p[f"blocks.{L}.attn.qkv.bias"].double()
    O = p[f"blocks.{L}.attn.proj.weight"].double()
    for h in range(8):
        V = W[512 + 32 * h: 512 + 32 * h + 32]; bv = b[512 + 32 * h: 512 + 32 * h + 32]
        Oh = O[:, 32 * h: 32 * h + 32]
        GO = Oh.T @ Oh; GV = V @ V.T + torch.outer(bv, bv)
        c2 = (GO * GV).sum() / (GV * GV).sum()
        cos = (GO * GV).sum() / GO.norm() / GV.norm()
        res = (GO - c2 * GV).norm() / GO.norm()
        print(f"{L} {h} | {Oh.norm():7.3f} {V.norm():7.3f} {Oh.norm()/V.norm():6.3f} | {(GO.trace()/GV.trace()).sqrt():6.3f} | {c2.sqrt():6.3f} | {cos:6.4f} | {res:6.4f}")
L, h = 5, 5
O = p["blocks.5.attn.proj.weight"].double(); W = p["blocks.5.attn.qkv.weight"].double()
Oh = O[:, 160:192]; V = W[512 + 160: 512 + 192]
print("\ncolumn norms of O head 5 (L5):", Oh.norm(dim=0))
print("row norms of V head 5 (L5):   ", V.norm(dim=1))
print("ratio col/row:", Oh.norm(dim=0) / V.norm(dim=1))
print("for comparison head 4 ratio:", O[:, 128:160].norm(dim=0) / W[512 + 128:512 + 160].norm(dim=1))
print("row norms of O (all 256 outputs) head5 block, top10:", Oh.norm(dim=1).topk(10))
print("svd O head5:", torch.linalg.svdvals(Oh))
print("svd O head4:", torch.linalg.svdvals(O[:, 128:160]))
print("svd V head5:", torch.linalg.svdvals(V))
print("svd V head4:", torch.linalg.svdvals(W[512 + 128:512 + 160]))
x = Oh.flatten(); print("O head5 entries: mean %.5f std %.5f kurt %.2f" % (x.mean(), x.std(), ((x - x.mean()) ** 4).mean() / x.var() ** 2))
for hh in (4, 6):
    x = O[:, 32 * hh:32 * hh + 32].flatten(); print(f"O head{hh} entries: mean %.5f std %.5f kurt %.2f" % (x.mean(), x.std(), ((x - x.mean()) ** 4).mean() / x.var() ** 2))
# other tensors: Q/K balance per head (Q_h^T-side invariance: Q -> M Q, K -> M^-T K)
print("\nQ/K balance: cos(Q Q^T + bq bq^T, K K^T + bk bk^T) and norm ratio")
for L in range(6):
    W = p[f"blocks.{L}.attn.qkv.weight"].double(); b = p[f"blocks.{L}.attn.qkv.bias"].double()
    row = []
    for h in range(8):
        Q = W[32 * h:32 * h + 32]; K = W[256 + 32 * h:256 + 32 * h + 32]
        bq = b[32 * h:32 * h + 32]; bk = b[256 + 32 * h:256 + 32 * h + 32]
        GQ = Q @ Q.T + torch.outer(bq, bq); GK = K @ K.T + torch.outer(bk, bk)
        row.append(f"{(GQ*GK).sum()/GQ.norm()/GK.norm():.3f}/{(GQ.trace()/GK.trace()).sqrt():.2f}")
    print(L, " ".join(row))
# MLP balance per neuron: |fc row|^2 + b^2 vs |proj col|^2
print("\nMLP neuron balance: (|fc_i|^2+b_i^2) vs |proj_:,i|^2 ; ratio quantiles")
for L in range(6):
    fc = p[f"blocks.{L}.mlp.fc.weight"].double(); fb = p[f"blocks.{L}.mlp.fc.bias"].double(); pr = p[f"blocks.{L}.mlp.proj.weight"].double()
    r = (pr.norm(dim=0) ** 2 / (fc.norm(dim=1) ** 2 + fb ** 2)).sqrt()
    print(L, torch.quantile(r, torch.tensor([0, .01, .25, .5, .75, .99, 1.]).double()))
