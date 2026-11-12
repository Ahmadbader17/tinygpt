"""Paired comparison of generation calls (same prompts and seed): usage 11_paired.py base1,base2 edit1,edit2"""
import json, sys
from pathlib import Path
import numpy as np
from tinygpt.wsp.metrics import has_loop, repeated_ngram_fraction
def load(calls):
    rows = []
    for c in calls.split(","): rows += json.loads(Path(f"work/outputs/generations_call{c}.json").read_text())
    return rows
a, b = load(sys.argv[1]), load(sys.argv[2]); n = len(a); assert n == len(b)
rng = np.random.default_rng(0); idx = rng.integers(0, n, (10000, n))
for name, f in [("loop", lambda r: float(r["loop"])), ("rep4", lambda r: r["rep4"]), ("rep3", lambda r: repeated_ngram_fraction(r["tokens"], 3)),
                ("rep2", lambda r: repeated_ngram_fraction(r["tokens"], 2)), ("long_loop", lambda r: float(has_loop(r["tokens"], 8))),
                ("length", lambda r: float(len(r["tokens"]))), ("eot", lambda r: float(r["hit_eot"]))]:
    x = np.array([f(r) for r in a]); y = np.array([f(r) for r in b]); d = y - x; bm = d[idx].mean(1)
    print(f"{name:9s} base {x.mean():8.4f}  edit {y.mean():8.4f}  diff {d.mean():+8.4f}  95% [{np.quantile(bm,.025):+.4f}, {np.quantile(bm,.975):+.4f}]")
la = np.array([r["loop"] for r in a]); lb = np.array([r["loop"] for r in b])
print(f"n={n}: loops lost {int((la & ~lb).sum())}, gained {int((~la & lb).sum())}, kept {int((la & lb).sum())}; identical texts {sum(r['tokens']==s['tokens'] for r,s in zip(a,b))}")
