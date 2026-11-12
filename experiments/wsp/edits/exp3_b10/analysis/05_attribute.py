"""Attribute repeat-completing tokens to components, from a captured read call.

usage: 05_attribute.py <npz> <rows.json>
rows.json: {"rows": [[ids]], "valid": [n], "prompt_len": [n]}  (targets analysed: prompt_len <= t+1 < valid)

Definitions at position t (the model predicts x[t+1]):
  ctx3   : the trigram x[t-2..t] occurred earlier, ending at some j < t  (a 4-gram repeat is possible)
  copy   : ctx3 and the actual next token equals the token that followed one of those earlier occurrences
           (i.e. the 4-gram ending at t+1 is a repeat -> exactly what rep-4 / loop rate count)
  ind pos: positions j+1 (<= t) with x[j] == x[t]  (where an induction head would look)
DLA of a component = its output, passed through the final LayerNorm with the actual final-residual scale,
dotted with (E[token] - mean E): its direct contribution to that token's logit relative to the average token.
Diagnosis only: nothing computed here is written into weights.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from tinygpt.wsp.edits import load_base_params

npz = np.load(sys.argv[1]); meta = json.loads(Path(sys.argv[2]).read_text())
rows = np.array(meta["rows"]); valid = meta["valid"]; plen = meta["prompt_len"]
B, T = rows.shape
P = load_base_params(Path("checkpoint"))
E = P["wte.weight"].numpy(); g = P["ln_f.weight"].numpy()
Ec = E - E.mean(0, keepdims=True)

final = npz["resid_post5"]                                    # (B,T,C)
sigma = np.sqrt(((final - final.mean(-1, keepdims=True)) ** 2).mean(-1) + 1e-5)   # (B,T)
logits = npz["logits"]; logp = logits - np.logaddexp.reduce(logits, -1, keepdims=True)

# position classes
ctx3 = np.zeros((B, T), bool); copy = np.zeros((B, T), bool); sel = np.zeros((B, T), bool)
copytok = -np.ones((B, T), int)        # continuation of the most recent earlier occurrence of the trigram
indmask = np.zeros((B, T, T), bool)    # [b,t,k]: k = j+1 with x[j]==x[t], j<t
for b in range(B):
    x = rows[b]
    seen = {}
    for t in range(T - 1):
        if t >= 2:
            tri = tuple(x[t - 2 : t + 1])
            if tri in seen:
                ctx3[b, t] = True; copytok[b, t] = seen[tri][-1]
                copy[b, t] = x[t + 1] in seen[tri]
            seen.setdefault(tri, []).append(x[t + 1])  # registered after use: only strictly earlier occurrences count
        js = np.nonzero(x[:t] == x[t])[0]
        indmask[b, t, js + 1] = True
        sel[b, t] = (t + 1 >= plen[b]) and (t + 1 < valid[b])
n = sel.sum()
print(f"positions analysed {n}; ctx3 {100*(ctx3&sel).sum()/n:.1f}%  copy (4-gram repeat) {100*(copy&sel).sum()/n:.2f}%  "
      f"P(copy|ctx3) {(copy&sel).sum()/(ctx3&sel).sum():.3f}")
bi, ti = np.nonzero(sel & ctx3)
pc = np.exp(logp[bi, ti, copytok[bi, ti]])
hit = rows[bi, ti + 1] == copytok[bi, ti]
print(f"at ctx3 positions: mean model prob of most-recent continuation {pc.mean():.3f}; actual rate it follows {hit.mean():.3f}")
for lo, hi in [(0, .1), (.1, .3), (.3, .6), (.6, .9), (.9, 1.01)]:
    m = (pc >= lo) & (pc < hi)
    if m.sum(): print(f"   p in [{lo},{hi}): n={m.sum():5d} mean p={pc[m].mean():.3f} actual={hit[m].mean():.3f}")
tl = npz["token_loss"]
print(f"mean loss: copy positions {tl[(copy&sel)[:, :-1]].mean():.3f}, ctx3-not-copy {tl[(ctx3&~copy&sel)[:, :-1]].mean():.3f}, no-ctx3 {tl[(~ctx3&sel)[:, :-1]].mean():.3f}")

tgt = np.zeros((B, T), int); tgt[:, :-1] = rows[:, 1:]
def dla(out, tok_ids):          # out (B,T,C) -> (B,T)
    o = (out - out.mean(-1, keepdims=True)) / sigma[..., None] * g
    return np.einsum("btc,btc->bt", o, Ec[tok_ids])

A = sel & copy; N = sel & ~ctx3; C3 = sel & ctx3 & ~copy
ct = np.where(copytok >= 0, copytok, 0)
print("\ncomponent | attn on induction positions (ctx3 pos) | DLA to actual token: copy pos / non-ctx pos | DLA to copy-token at ctx3 pos where it was NOT taken")
table = []
for L in range(6):
    att = npz[f"attn{L}"]; ho = npz[f"head_out{L}"]
    for h in range(8):
        ia = (att[:, h] * indmask).sum(-1)       # (B,T)
        d_act = dla(ho[:, h], tgt); d_cp = dla(ho[:, h], ct)
        table.append((f"L{L}H{h}", ia[sel & ctx3].mean(), ia[N].mean(), d_act[A].mean(), d_act[N].mean(), d_cp[sel & ctx3].mean(), d_cp[C3].mean()))
    m = npz[f"mlp_out{L}"]; d_act = dla(m, tgt); d_cp = dla(m, ct)
    table.append((f"L{L}mlp", float("nan"), float("nan"), d_act[A].mean(), d_act[N].mean(), d_cp[sel & ctx3].mean(), d_cp[C3].mean()))
print(f"{'comp':7s} {'indattn@ctx3':>12s} {'indattn@other':>13s} {'DLAact@copy':>11s} {'DLAact@other':>12s} {'DLAcopytok@ctx3':>15s} {'DLAcopytok@nottaken':>19s}")
for r in table: print(f"{r[0]:7s} {r[1]:12.3f} {r[2]:13.3f} {r[3]:11.2f} {r[4]:12.2f} {r[5]:15.2f} {r[6]:19.2f}")
tot = sum(r[3] for r in table); print(f"sum DLAact@copy over components {tot:.2f}; @other {sum(r[4] for r in table):.2f}")
print("\nranked by DLA to actual token at copy positions minus at other positions (the copy-specific part):")
for r in sorted(table, key=lambda r: -(r[3] - r[4]))[:14]: print(f"  {r[0]:7s} excess {r[3]-r[4]:6.2f}  (copy {r[3]:.2f}, other {r[4]:.2f})  indattn {r[1]:.3f}")
