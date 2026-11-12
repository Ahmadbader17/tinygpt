"""No model: read saved generations, print looping ones and the n-grams that loop."""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, "analysis")
from common import tok

f = sys.argv[1]
rows = json.loads(Path(f).read_text())
cnt = Counter()
for r in rows:
    t = r["tokens"]
    g = Counter(tuple(t[i:i+4]) for i in range(len(t)-3))
    top = [(k, v) for k, v in g.items() if v >= 3]
    if top:
        print(f"--- prompt {r['prompt']} len {len(t)} rep4 {r['rep4']:.3f}")
        for k, v in sorted(top, key=lambda kv: -kv[1])[:6]:
            print("   ", v, repr(tok.decode(list(k))))
            cnt[tok.decode(list(k))] += 1
        if len(sys.argv) > 2:
            print("   TEXT:", r["text"])
print(cnt.most_common(25))
