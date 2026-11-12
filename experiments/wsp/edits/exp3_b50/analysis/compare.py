"""No model: paired comparison of saved generations. Usage: compare.py base=1,8,9,10,11 v1=13,14,15,16,17
Call numbers must be listed in the same block order for each label. Paired bootstrap over rows."""
import json, sys
from pathlib import Path
import numpy as np
from tinygpt.wsp.metrics import has_loop, repeated_ngram_fraction

def load(calls):
    rows = []
    for c in calls:
        rows += json.loads(Path(f"work/outputs/generations_call{int(c):03d}.json").read_text())
    return rows

def stats(rows):
    t = [r["tokens"] for r in rows]
    return {"loop": np.array([has_loop(x) for x in t], float), "long": np.array([has_loop(x, 8) for x in t], float),
            "rep4": np.array([repeated_ngram_fraction(x, 4) for x in t]), "rep2": np.array([repeated_ngram_fraction(x, 2) for x in t]),
            "len": np.array([len(x) for x in t], float), "eot": np.array([r["hit_eot"] for r in rows], float)}

sets = {a.split("=")[0]: stats(load(a.split("=")[1].split(","))) for a in sys.argv[1:]}
names = list(sets)
b = sets[names[0]]
n = len(b["loop"])
rng = np.random.default_rng(0)
idx = rng.integers(0, n, size=(10000, n))
print(f"rows {n}")
print(f"{names[0]:10s} " + "  ".join(f"{k}={v.mean():.4f}" for k, v in b.items()))
for nm in names[1:]:
    s = sets[nm]
    assert len(s["loop"]) == n
    print(f"{nm:10s} " + "  ".join(f"{k}={v.mean():.4f}" for k, v in s.items()))
    for k in ("loop", "rep4", "rep2", "len"):
        d = s[k] - b[k]
        lo, hi = np.quantile(d[idx].mean(1), [0.025, 0.975])
        print(f"   {k:5s} diff {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]   relative {d.mean()/b[k].mean():+.1%}")
