"""Free analysis of the pass-2 capture: which components push the next token when it is a repeat?

Position t predicts x[t+1]. Categories of t:
  R4: the 4-gram x[t-2..t+1] already occurred earlier in the row (the token *completes a repeated 4-gram*)
  R1: x[t+1] occurred earlier, but not R4
  N : x[t+1] is new to the context
Direct logit attribution (DLA) of a component writing c into the final residual:
  ((c - mean(c)) / std(final resid) * ln_f.weight) . (E[target] - mean_v E[v])   -> logit of target minus mean logit
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "analysis")
from common import ws
from tinygpt.wsp.edits import load_base_params
P = load_base_params(ws / "checkpoint")
E = P["wte.weight"]; w = P["ln_f.weight"]
Z = np.load("work/outputs/read_call002.npz")
seqs = json.load(open("analysis/04_sequences.json"))
X = torch.tensor(seqs["gen"] + seqs["real"])           # (64, T)
B, T = X.shape
final = torch.tensor(Z["resid_post5"])
fc = final - final.mean(-1, keepdim=True)
std = fc.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()  # (B,T,1)
Ebar = E.mean(0)
tgt = torch.zeros(B, T, dtype=torch.long); tgt[:, :-1] = X[:, 1:]
U = (E[tgt] - Ebar) * w                                  # (B,T,C) direction read by the target logit

def dla(c):                                              # c (B,T,C) -> (B,T)
    c = c - c.mean(-1, keepdim=True)
    return ((c / std) * U).sum(-1)

cat = np.zeros((B, T), dtype=int)                        # 0 N, 1 R1, 2 R4, -1 invalid
src = torch.zeros(B, T, T)                               # src[b,t,j]=1 if x[j]==x[t+1], j<=t
ind = torch.zeros(B, T, T)                               # ind[b,t,j]=1 if x[j]==x[t+1] and x[j-1]==x[t] (induction source)
for b in range(B):
    x = X[b].tolist(); seen4 = set(); seen1 = set(x[:1])
    for t in range(T - 1):
        nxt = x[t + 1]
        g = tuple(x[t - 2:t + 2]) if t >= 2 else None
        cat[b, t] = 2 if (g in seen4) else (1 if nxt in seen1 else 0)
        for j in range(t + 1):
            if x[j] == nxt:
                src[b, t, j] = 1
                if j >= 1 and x[j - 1] == x[t]: ind[b, t, j] = 1
        if g: seen4.add(g)
        seen1.add(nxt)
    cat[b, T - 1] = -1
cat[:, :8] = -1
cat = torch.tensor(cat)
grp = {"gen": slice(0, 32), "real": slice(32, 64)}
loss = torch.tensor(Z["token_loss"])
for name, s in grp.items():
    c = cat[s]
    print(name, "share of positions N/R1/R4:", [round(float((c == k).float().sum() / (c >= 0).sum()), 3) for k in (0, 1, 2)],
          "mean loss N/R1/R4:", [round(float(loss[s][(c[:, :-1] == k)].mean()), 3) for k in (0, 1, 2)])

def table(vals, label):                                  # vals (B,T)
    out = []
    for name, s in grp.items():
        for k in (0, 1, 2):
            out.append(float(vals[s][cat[s] == k].mean()))
    return out

print("\nDLA to the actual next token (logit minus mean logit). cols: gen N,R1,R4 | real N,R1,R4 ; then attention mass")
print("on any earlier copy of the next token (src) and on induction sources (ind), at R4 positions: gen, real")
rows = []
emb = dla(torch.tensor(Z["resid_pre0"]))
print("embed   ", " ".join(f"{v:6.2f}" for v in table(emb, "")))
tot = dla(final)
for l in range(6):
    ho = torch.tensor(Z[f"head_out{l}"])                # (B,H,T,C)
    at = torch.tensor(Z[f"attn{l}"])                    # (B,H,T,T)
    for h in range(8):
        d = dla(ho[:, h])
        a_src = (at[:, h] * src).sum(-1); a_ind = (at[:, h] * ind).sum(-1)
        tb = table(d, "")
        att = [float(a_src[s][cat[s] == 2].mean()) for s in grp.values()] + [float(a_ind[s][cat[s] == 2].mean()) for s in grp.values()]
        rows.append((f"{l}.{h}", tb, att))
        print(f"head {l}.{h}", " ".join(f"{v:6.2f}" for v in tb), "  | src", " ".join(f"{v:5.2f}" for v in att[:2]), " ind", " ".join(f"{v:5.2f}" for v in att[2:]))
    m = dla(torch.tensor(Z[f"mlp_out{l}"]))
    print(f"mlp  {l}  ", " ".join(f"{v:6.2f}" for v in table(m, "")))
print("total   ", " ".join(f"{v:6.2f}" for v in table(tot, "")))
print("\nHeads ranked by (R4 - N) DLA gap on generated text:")
for n, tb, att in sorted(rows, key=lambda r: -(r[1][2] - r[1][0]))[:12]:
    print(n, f"gen R4-N {tb[2]-tb[0]:5.2f}  R1-N {tb[1]-tb[0]:5.2f} | real R4-N {tb[5]-tb[3]:5.2f} R1-N {tb[4]-tb[3]:5.2f} | attn src gen {att[0]:.2f} real {att[1]:.2f} ind gen {att[2]:.2f} real {att[3]:.2f}")
print("most negative:")
for n, tb, att in sorted(rows, key=lambda r: (r[1][2] - r[1][0]))[:6]:
    print(n, f"gen R4-N {tb[2]-tb[0]:5.2f}  R1-N {tb[1]-tb[0]:5.2f} | real R4-N {tb[5]-tb[3]:5.2f} R1-N {tb[4]-tb[3]:5.2f}")
