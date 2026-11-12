"""Is there any weight-only regularity that ties a head's write matrix P_h to its read matrix V_h
(or to another head)? Checked on intact heads, to see what could be said about L5 h6. Weights only."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params

torch.set_printoptions(precision=3, linewidth=220, sci_mode=False)
p = {k: v.double() for k, v in load_base_params(Path("checkpoint")).items()}

def parts(L, h):
    qkv = p[f"blocks.{L}.attn.qkv.weight"]; pr = p[f"blocks.{L}.attn.proj.weight"]
    s = slice(32*h, 32*h+32)
    return qkv[:256][s], qkv[256:512][s], qkv[512:][s], pr[:, s]

def orth(M):  # orthonormal basis of the row space of M (rows x 256)
    return torch.linalg.qr(M.T)[0]

print("== write/read relation per head: overlap of col-space(P) with row-space(V) (chance 0.125), tr(VP)/(|V||P|) ==")
for L in range(6):
    row = []
    for h in range(8):
        Q, K, V, P = parts(L, h)
        if P.norm() == 0: row.append("   zeroed   "); continue
        ov = float((orth(V).T @ orth(P.T)).pow(2).sum() / 32)
        tr = float(torch.trace(V @ P) / (V.norm() * P.norm()))
        row.append(f"{ov:.3f}/{tr:+.3f}")
    print(L, "  ".join(row))

print("== layer 5: similarity of head 6 to the others ==")
Q6, K6, V6, _ = parts(5, 6)
M6 = Q6.T @ K6
for h in range(8):
    Q, K, V, P = parts(5, h)
    M = Q.T @ K
    cosqk = float((M * M6).sum() / (M.norm() * M6.norm()))
    vov = float((orth(V).T @ orth(V6)).pow(2).sum() / 32)
    qov = float((orth(Q).T @ orth(Q6)).pow(2).sum() / 32)
    kov = float((orth(K).T @ orth(K6)).pow(2).sum() / 32)
    print(f"h{h}: cos(QK,QK6)={cosqk:+.3f}  rowspace overlap V={vov:.3f} Q={qov:.3f} K={kov:.3f} (chance 0.125)")

print("== all layers: max |cos| between QK matrices of distinct heads in the same layer (for scale) ==")
for L in range(6):
    Ms = [parts(L, h)[0].T @ parts(L, h)[1] for h in range(8)]
    C = torch.tensor([[float((a*b).sum()/(a.norm()*b.norm())) for b in Ms] for a in Ms])
    print(L, C)

print("== L5 head 6: what does it read / attend to? positional vs token content of QK ==")
wte = p["wte.weight"]; wpe = p["wpe.weight"]
g1 = p["blocks.5.ln1.weight"]
for h in range(8):
    Q, K, V, P = parts(5, h)
    # fraction of Q, K, V energy (after ln1 gain) lying in the span of the positional embeddings' top directions
    U = torch.linalg.svd(wpe, full_matrices=False)[2][:10].T  # top-10 positional directions (256 x 10)
    f = lambda M: float(((M * g1) @ U).pow(2).sum() / (M * g1).pow(2).sum())
    print(f"h{h}: frac of energy in top-10 wpe directions (chance {10/256:.3f}): Q {f(Q):.3f} K {f(K):.3f} V {f(V):.3f}")

print("== direct logit effect structure of intact L5 heads: W_U g_f P V g_1 W_E^T diag vs offdiag (token subsample) ==")
gf = p["ln_f.weight"]
idx = torch.arange(0, 4096, 4)
E = wte[idx]
for h in range(8):
    Q, K, V, P = parts(5, h)
    if P.norm() == 0: print(f"h{h}: zeroed"); continue
    A = (E * gf) @ P            # (n, 32) unembed side
    B = (E * g1) @ V.T          # (n, 32) embed side
    T = A @ B.T
    d = torch.diagonal(T)
    print(f"h{h}: diag mean {float(d.mean()):+.3f}  offdiag mean {float((T.sum()-d.sum())/(T.numel()-len(d))):+.4f}  offdiag std {float(T.std()):.3f}")
