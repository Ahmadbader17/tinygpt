"""Closer look: exact zeros, layer-1 proj null direction, balancedness of v/proj per head,
duplicate rows/columns, outliers per row/column. Weights only."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params

torch.set_printoptions(precision=3, linewidth=220, sci_mode=False)
p = {k: v.double() for k, v in load_base_params(Path("checkpoint")).items()}

pr5 = p["blocks.5.attn.proj.weight"]
z = (pr5 == 0)
print("L5 proj zero columns:", torch.nonzero(z.all(0)).flatten().tolist())
print("L5 proj zero entries outside those:", int(z[:, ~z.all(0)].sum()))
print("L5 proj bias head-agnostic; qkv rows of head 6 (q,k,v) norms:",
      [float(p["blocks.5.attn.qkv.weight"][o+192:o+224].norm()) for o in (0, 256, 512)])

print("== smallest singular values of every attn.proj ==")
for L in range(6):
    s = torch.linalg.svdvals(p[f"blocks.{L}.attn.proj.weight"])
    print(L, s[-6:])
    print("   rank-ish (s>1e-6):", int((s > 1e-6).sum()), " col norms min:", float(p[f"blocks.{L}.attn.proj.weight"].norm(dim=0).min()),
          " row norms min:", float(p[f"blocks.{L}.attn.proj.weight"].norm(dim=1).min()))

print("== balancedness: singular values of proj[:,h] vs [v_h | b_v] per head ==")
for L in range(6):
    qkv = p[f"blocks.{L}.attn.qkv.weight"]; b = p[f"blocks.{L}.attn.qkv.bias"]; pr = p[f"blocks.{L}.attn.proj.weight"]
    for h in range(8):
        P = pr[:, 32*h:32*h+32]; V = torch.cat([qkv[512+32*h:512+32*h+32], b[512+32*h:512+32*h+32, None]], 1)
        G1 = P.T @ P; G2 = V @ V.T
        rel = float((G1 - G2).norm() / G2.norm())
        sp = torch.linalg.svdvals(P); sv = torch.linalg.svdvals(V)
        print(f"L{L} h{h} |P^TP - VV^T|/|VV^T| = {rel:.3f}  sP top3 {sp[:3].tolist()}  sV top3 {sv[:3].tolist()}")

print("== per-tensor row/col norm outliers (z-score > 6) ==")
for n, v in p.items():
    if v.ndim != 2: continue
    for d, lab in ((1, "row"), (0, "col")):
        nr = v.norm(dim=d); zs = (nr - nr.median()) / (nr.std() + 1e-12)
        idx = torch.nonzero(zs.abs() > 6).flatten().tolist()
        if idx: print(n, lab, idx[:20], [round(float(nr[i]), 3) for i in idx[:20]], "median", round(float(nr.median()), 3))
