"""L5 heads 1, 6, 7 look like a family (QK cosine ~0.47-0.50, shared read subspace).
Question, answered from weights only: do the two intact siblings also share their OV map, well
enough that a sibling's OV predicts a head's write matrix? Leave-one-out check on intact heads:
predict P_a from head b as  P_a_hat = (P_b V_b) pinv(V_a)  and compare with the real P_a."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params

torch.set_printoptions(precision=3, linewidth=220, sci_mode=False)
p = {k: v.double() for k, v in load_base_params(Path("checkpoint")).items()}
L = 5
qkv = p[f"blocks.{L}.attn.qkv.weight"]; pr = p[f"blocks.{L}.attn.proj.weight"]
V = [qkv[512+32*h:512+32*h+32] for h in range(8)]
P = [pr[:, 32*h:32*h+32] for h in range(8)]
OV = [P[h] @ V[h] for h in range(8)]
cos = lambda a, b: float((a*b).sum() / (a.norm()*b.norm() + 1e-30))
orth = lambda M: torch.linalg.qr(M)[0]

print("cos(OV_a, OV_b) for layer 5:")
print(torch.tensor([[cos(OV[a], OV[b]) for b in range(8)] for a in range(8)]))
print("write-space overlap col(P_a) vs col(P_b) (chance 0.125):")
print(torch.tensor([[float((orth(P[a]).T @ orth(P[b])).pow(2).sum()/32) if P[a].norm()>0 and P[b].norm()>0 else float('nan') for b in range(8)] for a in range(8)]))

print("leave-one-out: P_a_hat = OV_b pinv(V_a); cos(P_a_hat, P_a), |P_a_hat|/|P_a|, best-scale residual")
for a in range(8):
    if a == 6: continue
    row = []
    for b in range(8):
        if b in (a, 6): row.append("      -      "); continue
        Ph = OV[b] @ torch.linalg.pinv(V[a])
        row.append(f"{cos(Ph, P[a]):+.3f}/{float(Ph.norm()/P[a].norm()):.2f}")
    print(f"a={a}: " + "  ".join(row))

print("two-sibling predictions:")
for a, bs in ((1, (7,)), (7, (1,)), (1, (7, 4)), (7, (1, 4))):
    Ph = sum(OV[b] for b in bs) @ torch.linalg.pinv(V[a])
    print(a, bs, f"cos {cos(Ph, P[a]):+.3f} norm ratio {float(Ph.norm()/P[a].norm()):.2f} rel.err of raw prediction {float((Ph-P[a]).norm()/P[a].norm()):.3f}")

print("candidate for head 6:")
for bs in ((1,), (7,), (1, 7)):
    Ph = sum(OV[b] for b in bs) @ torch.linalg.pinv(V[6])
    print(bs, "norm", float(Ph.norm()), " (V6 norm", float(V[6].norm()), ") top sv", torch.linalg.svdvals(Ph)[:4])
P61 = OV[1] @ torch.linalg.pinv(V[6]); P67 = OV[7] @ torch.linalg.pinv(V[6])
print("agreement of the two independent predictions of P_6: cos", cos(P61, P67))
# same agreement statistic for intact heads, where truth is known
for a in (1, 7, 4, 0, 3):
    others = [b for b in (1, 7, 4) if b != a][:2]
    x, y = (OV[b] @ torch.linalg.pinv(V[a]) for b in others)
    print(f"  reference: head {a} predicted from {others}: mutual cos {cos(x, y):+.3f}; cos with truth {cos(x, P[a]):+.3f}, {cos(y, P[a]):+.3f}")
