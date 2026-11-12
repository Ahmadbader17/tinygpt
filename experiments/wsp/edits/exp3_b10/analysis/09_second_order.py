"""Diagnosis (unedited model, passes 2 and 3). 08 showed first-order loss effects of scaling any component are ~0
(the trained model sits at a stationary point), so the cost of down-scaling is second order.

Per component, direct path, final-LN scale frozen, with z_v = o~ . E[v] the component's logit contribution:
  fisher     : mean over real-text positions of Var_{v~p}(z_v).  Scaling the component by (1-eps) costs ~ eps^2/2 * fisher nats.
  odds_push  : mean over ctx3 positions of the model's own text (copy-token prob < 0.95) of
               z_c - sum_{v != c} p_v z_v / (1 - p_c): log-odds the component gives the token that would complete a repeated 4-gram.
  odds_other : same quantity for the actual next token at non-ctx3 real-text positions (p < 0.95): ordinary usefulness.
Only used to decide which components matter.
"""
import json, sys
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
    ctx3 = np.zeros((B, T), bool); sel = np.zeros((B, T), bool); ct = np.zeros((B, T), int)
    for b in range(B):
        x = rows[b]; seen = {}
        for t in range(2, T - 1):
            tri = tuple(x[t - 2 : t + 1])
            if tri in seen: ctx3[b, t] = True; ct[b, t] = seen[tri]
            seen[tri] = x[t + 1]
        for t in range(T - 1): sel[b, t] = (t + 1 >= meta["prompt_len"][b]) and (t + 1 < meta["valid"][b])
    tgt = np.zeros((B, T), int); tgt[:, :-1] = rows[:, 1:]
    return dict(npz=npz, sigma=sigma, p=p, ctx3=ctx3 & sel, sel=sel, ct=ct, tgt=tgt)

def z_of(d, out):
    o = (out - out.mean(-1, keepdims=True)) / d["sigma"][..., None] * g
    return o @ E.T                                            # (B,T,V)

def odds(d, z, toks, mask):
    b, t = np.nonzero(mask); pc = d["p"][b, t, toks[b, t]]; ok = pc < 0.95
    b, t, pc = b[ok], t[ok], pc[ok]
    zc = z[b, t, toks[b, t]]; zbar = (d["p"][b, t] * z[b, t]).sum(-1)
    return ((zc - zbar) / (1 - pc)).mean()

def fisher(d, z):
    m = (d["p"] * z).sum(-1, keepdims=True)
    return ((d["p"] * (z - m) ** 2).sum(-1))[d["sel"]].mean()

G = load("work/outputs/read_call002.npz", "work/outputs/p02_rows.json")
R = load("work/outputs/read_call003.npz", "work/outputs/p03_rows.json")
def out_of(d, name):
    L = int(name[1]); return d["npz"][f"mlp_out{L}"].astype(np.float64) if name.endswith("mlp") else d["npz"][f"head_out{L}"][:, int(name[3])].astype(np.float64)
names = [f"L{L}H{h}" for L in range(1, 6) for h in range(8)] + [f"L{L}mlp" for L in range(6)]
res = []
for nme in names:
    zg = z_of(G, out_of(G, nme)); zr = z_of(R, out_of(R, nme))
    res.append((nme, fisher(R, zr), odds(G, zg, G["ct"], G["ctx3"]), odds(R, zr, R["ct"], R["ctx3"]), odds(R, zr, R["tgt"], R["sel"] & ~R["ctx3"])))
print(f"{'comp':7s} {'fisher':>8s} {'odds_push gen':>13s} {'odds_push real':>14s} {'odds_other':>10s} {'push/sqrt(fisher)':>17s}")
for r in sorted(res, key=lambda r: -r[2]):
    print(f"{r[0]:7s} {r[1]:8.3f} {r[2]:13.3f} {r[3]:14.3f} {r[4]:10.3f} {r[2]/np.sqrt(r[1]):17.2f}")
sets = [s.split(",") for s in sys.argv[1:]]
for s in sets:
    zg = sum(z_of(G, out_of(G, n)) for n in s); zr = sum(z_of(R, out_of(R, n)) for n in s)
    print("set", s, f"fisher {fisher(R, zr):.3f} odds_push gen {odds(G, zg, G['ct'], G['ctx3']):.3f} real {odds(R, zr, R['ct'], R['ctx3']):.3f} other {odds(R, zr, R['tgt'], R['sel'] & ~R['ctx3']):.3f}")
