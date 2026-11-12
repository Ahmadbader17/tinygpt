"""Free: compare teacher-forced token losses (base pass 2 vs an edited tf pass) by repeat category.
usage: 08_tf_compare.py work/outputs/read_callNNN.npz"""
import json, sys
import numpy as np
seqs = json.load(open("analysis/04_sequences.json"))
X = np.array(seqs["gen"] + seqs["real"]); B, T = X.shape
a = np.load("work/outputs/read_call002.npz")["token_loss"]; b = np.load(sys.argv[1])["token_loss"]
cat = np.full((B, T - 1), -1)
for r in range(B):
    x = X[r].tolist(); seen4 = set(); seen1 = {x[0]}
    for t in range(T - 1):
        g = tuple(x[t - 2:t + 2]) if t >= 2 else None
        cat[r, t] = 2 if g in seen4 else (1 if x[t + 1] in seen1 else 0)
        if g: seen4.add(g)
        seen1.add(x[t + 1])
cat[:, :8] = -1
for name, s in (("gen", slice(0, 32)), ("real", slice(32, 64))):
    print(name, f"all: base {a[s].mean():.4f} edited {b[s].mean():.4f} diff {b[s].mean()-a[s].mean():+.4f}")
    for k, lab in ((0, "N "), (1, "R1"), (2, "R4")):
        m = cat[s] == k
        d = (b[s] - a[s])[m]
        print(f"   {lab} n={m.sum():5d} base {a[s][m].mean():.3f} edited {b[s][m].mean():.3f} diff {d.mean():+.4f} (se {d.std()/np.sqrt(m.sum()):.4f})")
