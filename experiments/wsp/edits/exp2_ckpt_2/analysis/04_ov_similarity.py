"""Do heads with similar QK circuits also have similar OV circuits (P_h V_h)? If so the lost OV of
L5 h6 could be guessed from its neighbours h1/h7; if not, no weight-only guess is defensible.
Also: does a head's write space sit in the complement of the other heads' write spaces? Weights only."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params

torch.set_printoptions(precision=3, linewidth=220, sci_mode=False)
p = {k: v.double() for k, v in load_base_params(Path("checkpoint")).items()}

def parts(L, h):
    qkv = p[f"blocks.{L}.attn.qkv.weight"]; pr = p[f"blocks.{L}.attn.proj.weight"]
    s = slice(32*h, 32*h+32)
    return qkv[:256][s], qkv[256:512][s], qkv[512:][s], pr[:, s]

cos = lambda a, b: float((a*b).sum() / (a.norm()*b.norm()))
pairs = []
for L in range(6):
    QK = [parts(L, h)[0].T @ parts(L, h)[1] for h in range(8)]
    OV = [parts(L, h)[3] @ parts(L, h)[2] for h in range(8)]
    print(f"-- layer {L}: OV cosine matrix")
    C = torch.tensor([[cos(a, b) if a.norm() > 0 and b.norm() > 0 else float('nan') for b in OV] for a in OV])
    print(C)
    for a in range(8):
        for b in range(a+1, 8):
            if OV[a].norm() > 0 and OV[b].norm() > 0:
                pairs.append((L, a, b, cos(QK[a], QK[b]), cos(OV[a], OV[b])))
print("== pairs with QK cosine > 0.4: (layer, a, b, cosQK, cosOV) ==")
for t in sorted(pairs, key=lambda t: -t[3]):
    if t[3] > 0.4: print(t[0], t[1], t[2], f"{t[3]:+.3f} {t[4]:+.3f}")
x = torch.tensor([t[3] for t in pairs]); y = torch.tensor([t[4] for t in pairs])
print("correlation between cosQK and cosOV over all intact same-layer pairs:", float(torch.corrcoef(torch.stack([x, y]))[0, 1]), "n =", len(pairs))
print("mean cosOV for pairs with cosQK>0.4:", float(y[x > 0.4].mean()), " for all pairs:", float(y.mean()))

print("== fraction of each head's write energy lying in the orthogonal complement of the other 7 heads' columns (chance ~0.125) ==")
for L in range(5):  # layers with all heads intact
    pr = p[f"blocks.{L}.attn.proj.weight"]; row = []
    for h in range(8):
        keep = torch.ones(256, dtype=torch.bool); keep[32*h:32*h+32] = False
        Qo = torch.linalg.qr(pr[:, keep])[0]
        P = pr[:, ~keep]; res = P - Qo @ (Qo.T @ P)
        row.append(float(res.pow(2).sum() / P.pow(2).sum()))
    print(L, [round(r, 3) for r in row])
