"""Free: look at which 4-grams loop in the baseline generations (no model run)."""
import json, sys, glob
from collections import Counter
sys.path.insert(0, "analysis")
from common import tok
files = sys.argv[1:] or sorted(glob.glob("work/outputs/generations_call001.json"))
allc = Counter()
for f in files:
    rows = json.load(open(f))
    for r in rows:
        t = r["tokens"]
        c = Counter(tuple(t[i:i+4]) for i in range(len(t)-3))
        top = [(g, n) for g, n in c.most_common(3) if n >= 3]
        if top:
            print(r["prompt"], len(t), "|", "; ".join(f"{tok.decode(list(g))!r}x{n}" for g, n in top))
            for g, n in top: allc[tok.decode(list(g))] += 1
print(allc.most_common(30))
rows = json.load(open(files[0]))
for r in rows[:3]:
    print("----", r["loop"], r["rep4"]); print(r["text"])
