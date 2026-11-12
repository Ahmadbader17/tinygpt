"""Weights only: parameter inventory, per-head norms, layernorm gains."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params

torch.set_printoptions(precision=3, linewidth=200, sci_mode=False)
P = load_base_params(Path("checkpoint"))
for n, v in P.items():
    print(f"{n:32s} {tuple(v.shape)!s:14s} norm={v.norm():8.3f}")
print()
print("per-head Frobenius norms: Q, K, V rows and proj columns; and norm of OV = proj_h @ V_h")
for L in range(6):
    qkv = P[f"blocks.{L}.attn.qkv.weight"]; pr = P[f"blocks.{L}.attn.proj.weight"]
    for h in range(8):
        s = slice(32 * h, 32 * h + 32)
        q, k, v = qkv[:256][s], qkv[256:512][s], qkv[512:][s]
        ov = pr[:, s] @ v
        qk = q.T @ k
        print(f"L{L}H{h}: |Q|={q.norm():5.2f} |K|={k.norm():5.2f} |V|={v.norm():5.2f} |O|={pr[:, s].norm():5.2f} "
              f"|OV|={ov.norm():5.2f} |QK|={qk.norm():6.2f}")
print("ln_f gain mean/std", P["ln_f.weight"].mean(), P["ln_f.weight"].std())
print("wte row norm mean", P["wte.weight"].norm(dim=1).mean(), "wpe row norm mean", P["wpe.weight"].norm(dim=1).mean())
