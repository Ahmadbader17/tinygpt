"""Descriptive test on captured activations of the UNEDITED model (passes 2 and 3):
does the weights-defined 'copy-mode' part of a head's output (eigen-modes of the OV circuit with Re(lambda) > 0,
see 06_ov_eigs.py) carry the head's copy-specific logit contribution, and how much of its ordinary contribution?

The split is fixed by the weights alone (threshold Re > 0, stated before looking); nothing here is written to weights.
usage: 07_copy_mode_split.py <npz> <rows.json>
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, torch
spec = importlib.util.spec_from_file_location("ov", "analysis/06_ov_eigs.py"); ov = importlib.util.module_from_spec(spec); spec.loader.exec_module(ov)
P = ov.P
npz = np.load(sys.argv[1]); meta = json.loads(Path(sys.argv[2]).read_text())
rows = np.array(meta["rows"]); valid = meta["valid"]; plen = meta["prompt_len"]; B, T = rows.shape
E = P["wte.weight"].numpy(); g = P["ln_f.weight"].numpy(); Ec = E - E.mean(0, keepdims=True)
final = npz["resid_post5"].astype(np.float64)
sigma = np.sqrt(((final - final.mean(-1, keepdims=True)) ** 2).mean(-1) + 1e-5)
ctx3 = np.zeros((B, T), bool); copy = np.zeros((B, T), bool); sel = np.zeros((B, T), bool)
for b in range(B):
    x = rows[b]; seen = {}
    for t in range(2, T - 1):
        tri = tuple(x[t - 2 : t + 1])
        if tri in seen: ctx3[b, t] = True; copy[b, t] = x[t + 1] in seen[tri]
        seen.setdefault(tri, []).append(x[t + 1])
    for t in range(T - 1): sel[b, t] = (t + 1 >= plen[b]) and (t + 1 < valid[b])
tgt = np.zeros((B, T), int); tgt[:, :-1] = rows[:, 1:]
def dla(out):
    o = (out - out.mean(-1, keepdims=True)) / sigma[..., None] * g
    return np.einsum("btc,btc->bt", o, Ec[tgt])
A = sel & copy; N = sel & ~ctx3
print(f"{'head':6s} {'n_pos':>5s} | full: copy  other excess | copy-mode part: copy  other excess | rest: copy  other excess")
for L, h in [(5, 7), (5, 6), (4, 5), (5, 1), (5, 4), (4, 6), (5, 0), (4, 4), (3, 0)]:
    M = ov.circuit(L, h)
    lam, R = torch.linalg.eig(M)
    keep = (lam.real <= 0).to(R.dtype)                       # S0 keeps non-copy modes
    S0 = (R @ torch.diag(keep) @ torch.linalg.inv(R)).real.numpy()
    Pr = P[f"blocks.{L}.attn.proj.weight"][:, 32 * h : 32 * h + 32].numpy()
    ho = npz[f"head_out{L}"][:, h].astype(np.float64)        # (B,T,256) = y @ Pr.T
    y = ho @ np.linalg.pinv(Pr).T                            # (B,T,32)
    rest = (y @ S0.T) @ Pr.T; cp = ho - rest
    f, c, r = dla(ho), dla(cp), dla(rest)
    print(f"L{L}H{h}  {int((lam.real > 0).sum()):5d} | {f[A].mean():10.2f} {f[N].mean():6.2f} {f[A].mean()-f[N].mean():6.2f} |"
          f" {c[A].mean():20.2f} {c[N].mean():6.2f} {c[A].mean()-c[N].mean():6.2f} | {r[A].mean():10.2f} {r[N].mean():6.2f} {r[A].mean()-r[N].mean():6.2f}")
