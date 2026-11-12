"""Diagnosis (unedited model, captured passes 2 and 3): per-component direct first-order effects.

For component output o at position t, d/ds of log p(token u) when the component is scaled by (1+s), direct path only,
with the final LayerNorm scale frozen:   dlp(u) = o~ . E[u] - sum_v p(v) o~ . E[v],   o~ = centred o / sigma * ln_f gain.
  loss_help  : mean dlp(actual next token) over real-story positions (pass 3)  -> how much real-text loss leans on it
  copy_push  : mean dlp(copy token) over ctx3 positions of the model's own text (pass 2) -> how much it promotes
               completing an already-seen 4-gram when the model writes
  copy_push_real: same on real text (pass 3)
Used only to decide WHICH components matter; no number from here is written into a weight.
"""
import json
from pathlib import Path
import numpy as np
from tinygpt.wsp.edits import load_base_params
P = load_base_params(Path("checkpoint")); E = P["wte.weight"].numpy().astype(np.float64); g = P["ln_f.weight"].numpy().astype(np.float64)

def load(npzf, rowsf):
    npz = np.load(npzf); meta = json.loads(Path(rowsf).read_text())
    rows = np.array(meta["rows"]); B, T = rows.shape
    final = npz["resid_post5"].astype(np.float64)
    sigma = np.sqrt(((final - final.mean(-1, keepdims=True)) ** 2).mean(-1) + 1e-5)
    lg = npz["logits"].astype(np.float64); p = np.exp(lg - np.logaddexp.reduce(lg, -1, keepdims=True))
    Ebar = p @ E                                              # (B,T,C) expected embedding under the model
    ctx3 = np.zeros((B, T), bool); sel = np.zeros((B, T), bool); ct = np.zeros((B, T), int)
    for b in range(B):
        x = rows[b]; seen = {}
        for t in range(2, T - 1):
            tri = tuple(x[t - 2 : t + 1])
            if tri in seen: ctx3[b, t] = True; ct[b, t] = seen[tri]
            seen[tri] = x[t + 1]
        for t in range(T - 1): sel[b, t] = (t + 1 >= meta["prompt_len"][b]) and (t + 1 < meta["valid"][b])
    tgt = np.zeros((B, T), int); tgt[:, :-1] = rows[:, 1:]
    return npz, sigma, Ebar, ctx3 & sel, sel, ct, tgt

def dlp(out, sigma, Ebar, toks):
    o = (out - out.mean(-1, keepdims=True)) / sigma[..., None] * g
    return np.einsum("btc,btc->bt", o, E[toks] - Ebar)

G = load("work/outputs/read_call002.npz", "work/outputs/p02_rows.json")
R = load("work/outputs/read_call003.npz", "work/outputs/p03_rows.json")
res = []
for L in range(6):
    comps = [(f"L{L}H{h}", lambda z, h=h: z[f"head_out{L}"][:, h].astype(np.float64)) for h in range(8)] + [(f"L{L}mlp", lambda z: z[f"mlp_out{L}"].astype(np.float64))]
    for name, get in comps:
        og, orr = get(G[0]), get(R[0])
        loss_help = dlp(orr, R[1], R[2], R[6])[R[4]].mean()
        cp_gen = dlp(og, G[1], G[2], G[5])[G[3]].mean()
        cp_real = dlp(orr, R[1], R[2], R[5])[R[3]].mean()
        tgt_gen = dlp(og, G[1], G[2], G[6])[G[4]].mean()
        res.append((name, loss_help, cp_gen, cp_real, tgt_gen))
print(f"{'comp':7s} {'loss_help(real)':>15s} {'copy_push(gen ctx3)':>19s} {'copy_push(real ctx3)':>20s} {'ratio gen/loss':>14s}")
for r in sorted(res, key=lambda r: -r[2]):
    print(f"{r[0]:7s} {r[1]:15.3f} {r[2]:19.3f} {r[3]:20.3f} {r[2]/r[1] if abs(r[1])>1e-3 else float('nan'):14.1f}")
