"""Analysis of the pass-2 capture (diagnosis only; nothing here is written into a weight).

For every position t (predicting token t+1) classify the true next token:
  R4: the 4-gram ending at t+1 already occurred earlier in the row
  R2: the bigram (t, t+1) occurred earlier, but not the 4-gram
  R1: the token occurred earlier, the bigram did not
  N : new token
Then (a) direct logit attribution (DLA) of each head and MLP to the true next token per class,
(b) attention of each head to 'induction' positions j with tok[j-1]==tok[t], and to positions
holding the true next token, (c) calibration of the model's repeat probability on real stories.
"""
import json
from pathlib import Path
import numpy as np
import torch
from tinygpt.wsp.edits import load_base_params

np.set_printoptions(precision=2, linewidth=220, suppress=True)
P = load_base_params(Path("checkpoint"))
E = P["wte.weight"].numpy(); g = P["ln_f.weight"].numpy()
rows = json.load(open("analysis/06_rows.json"))
seqs = np.array(rows["gen"] + rows["real"])  # (64, 160)
B, T = seqs.shape
is_gen = np.arange(B) < 32
Z = np.load("work/outputs/read_call002.npz")

# ---- classes
cls = np.full((B, T - 1), "N ", dtype=object)
ind_mask = np.zeros((B, T, T), bool)   # [b, t, j]: tok[j-1]==tok[t], 1<=j<=t
tgt_mask = np.zeros((B, T, T), bool)   # [b, t, j]: tok[j]==tok[t+1], j<=t
for b in range(B):
    s = seqs[b].tolist()
    seen1, seen2, seen4 = set(), set(), set()
    for t in range(T - 1):
        seen1.add(s[t])
        if t >= 1: seen2.add((s[t - 1], s[t]))
        if t >= 3: seen4.add(tuple(s[t - 3:t + 1]))
        nxt = s[t + 1]
        if t >= 2 and tuple(s[t - 2:t + 2]) in seen4: cls[b, t] = "R4"
        elif (s[t], nxt) in seen2: cls[b, t] = "R2"
        elif nxt in seen1: cls[b, t] = "R1"
        for j in range(1, t + 1):
            if s[j - 1] == s[t] and j - 1 < t: ind_mask[b, t, j] = True
        for j in range(t + 1):
            if s[j] == nxt: tgt_mask[b, t, j] = True
for name, m in (("gen", is_gen), ("real", ~is_gen)):
    c = cls[m]
    print(name, {k: round(float((c == k).mean()), 3) for k in ("R4", "R2", "R1", "N ")},
          "mean token loss by class", {k: round(float(Z["token_loss"][m][c == k].mean()), 3) for k in ("R4", "R2", "R1", "N ")})

# ---- (a) DLA
x = Z["resid_post5"]  # (B, T, 256)
std = x.std(-1, keepdims=True)  # final LN divides by this (eps negligible)
Uc = (E * g)  # logits = ((x - mean)/std) * g @ E^T + b @ E^T
Uc = Uc - Uc.mean(0, keepdims=True)  # logit relative to the mean logit over the vocabulary
tgt = seqs[:, 1:]  # (B, T-1)
Ut = Uc[tgt]  # (B, T-1, 256)


def dla(c):  # c (B, T, 256) -> (B, T-1)
    c = c[:, :-1]
    c = (c - c.mean(-1, keepdims=True)) / std[:, :-1]
    return (c * Ut).sum(-1)


res = {}
for L in range(6):
    ho = Z[f"head_out{L}"]  # (B, H, T, 256)
    for h in range(8):
        res[f"L{L}H{h}"] = dla(ho[:, h])
    res[f"MLP{L}"] = dla(Z[f"mlp_out{L}"])
print("\n(a) mean DLA to the true next token (logit above vocab mean), by class; gen and real rows pooled, and R4 split")
print("comp      R4     R2     R1     N    | R4-N  | R4 gen  R4 real")
tab = []
for k, v in res.items():
    m = {c: v[cls == c].mean() for c in ("R4", "R2", "R1", "N ")}
    r4g = v[is_gen][cls[is_gen] == "R4"].mean(); r4r = v[~is_gen][cls[~is_gen] == "R4"].mean()
    tab.append((k, m["R4"], m["R2"], m["R1"], m["N "], m["R4"] - m["N "], r4g, r4r))
for r in tab:
    print(f"{r[0]:6s} {r[1]:6.2f} {r[2]:6.2f} {r[3]:6.2f} {r[4]:6.2f} | {r[5]:6.2f} | {r[6]:6.2f} {r[7]:6.2f}")
print("\nsorted by R4-N:")
for r in sorted(tab, key=lambda r: -r[5])[:14]:
    print(f"{r[0]:6s} R4={r[1]:6.2f} N={r[4]:6.2f} diff={r[5]:6.2f}")
tot = sum(res.values())
print("sum of all heads+MLPs: ", {c: round(float(tot[cls == c].mean()), 2) for c in ("R4", "R2", "R1", "N ")})

# ---- (b) attention
print("\n(b) attention mass, averaged over positions t>=8 where the target set is non-empty")
print("head   induction(j: tok[j-1]==tok[t])   to-true-next-token(j: tok[j]==tok[t+1]) at R4 | at R1 ;  uniform baseline in brackets")
valid_t = np.arange(T) >= 8
att_rows = []
for L in range(6):
    A = Z[f"attn{L}"]  # (B, H, T, T)
    for h in range(8):
        a = A[:, h]
        im = (a * ind_mask).sum(-1)  # (B, T)
        has = ind_mask.any(-1) & valid_t[None]
        base = (ind_mask.sum(-1) / (np.arange(T) + 1)[None])[has].mean()
        tm = (a * tgt_mask).sum(-1)[:, :-1]  # (B, T-1)
        hasT = tgt_mask.any(-1)[:, :-1] & valid_t[None, :-1]
        r4 = hasT & (cls == "R4"); r1 = hasT & (cls == "R1")
        att_rows.append((f"L{L}H{h}", im[has].mean(), base, tm[r4].mean(), tm[r1].mean()))
for r in att_rows:
    print(f"{r[0]}  ind={r[1]:.3f} [{r[2]:.3f}]   next@R4={r[3]:.3f}  next@R1={r[4]:.3f}")
print("top induction heads:", [(r[0], round(float(r[1]), 3)) for r in sorted(att_rows, key=lambda r: -r[1])[:8]])
print("top attend-to-next-token@R4 heads:", [(r[0], round(float(r[3]), 3)) for r in sorted(att_rows, key=lambda r: -r[3])[:8]])

# ---- (c) calibration on rows (real and gen separately)
logits = torch.from_numpy(Z["logits"])  # (B, T, V)
def repeat_mass(logp):  # mass on tokens that would complete a seen 4-gram / seen bigram
    out4 = np.zeros((B, T - 1)); out2 = np.zeros((B, T - 1)); out1 = np.zeros((B, T - 1))
    for b in range(B):
        s = seqs[b].tolist()
        nxt2, nxt4, seen1 = {}, {}, set()
        for t in range(T - 1):
            seen1.add(s[t])
            if t >= 1: nxt2.setdefault(s[t - 1], set()).add(s[t])
            if t >= 3: nxt4.setdefault(tuple(s[t - 3:t]), set()).add(s[t])
            p = logp[b, t]
            c2 = nxt2.get(s[t], ()); out2[b, t] = p[list(c2)].sum() if c2 else 0.0
            c4 = nxt4.get(tuple(s[t - 2:t + 1]), ()) if t >= 2 else (); out4[b, t] = p[list(c4)].sum() if c4 else 0.0
            out1[b, t] = p[list(seen1)].sum()
    return out4, out2, out1
p1 = torch.softmax(logits.double(), -1).numpy()
l8 = logits.double() / 0.8
kth = torch.topk(l8, 40).values[..., -1:]
p8 = torch.softmax(l8.masked_fill(l8 < kth, float("-inf")), -1).numpy()
act4 = (cls == "R4"); act2 = act4 | (cls == "R2"); act1 = act2 | (cls == "R1")
print("\n(c) calibration: predicted probability mass on repeat-completing tokens vs how often the true next token is one")
for nm, p in (("T=1", p1), ("T=0.8,top40", p8)):
    m4, m2, m1 = repeat_mass(p)
    for rn, m in (("real", ~is_gen), ("gen", is_gen)):
        print(f"{nm:12s} {rn:4s} 4gram: pred {m4[m].mean():.4f} actual {act4[m].mean():.4f} | bigram: pred {m2[m].mean():.4f} actual {act2[m].mean():.4f}"
              f" | seen-token: pred {m1[m].mean():.4f} actual {act1[m].mean():.4f}")
