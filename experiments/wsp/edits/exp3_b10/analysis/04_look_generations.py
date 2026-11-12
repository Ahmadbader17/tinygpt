"""Inspect generations of a generate call: which 4-grams repeat, and how."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp.metrics import ngrams
tok = BPETokenizer.load(Path("tokenizer.json"))
call = sys.argv[1] if len(sys.argv) > 1 else "001"
rows = json.loads(Path(f"work/outputs/generations_call{call}.json").read_text())
prompts = [x["tokens"] for x in json.loads(Path("dev/prompts.json").read_text())]
print("n", len(rows), "loop", np.mean([r["loop"] for r in rows]), "len", np.mean([len(r["tokens"]) for r in rows]))
allc = Counter()
for r in rows:
    c = Counter(ngrams(r["tokens"], 4))
    rep = [(g, n) for g, n in c.items() if n >= 3]
    for g, n in c.items():
        if n >= 2: allc[tok.decode(list(g))] += n - 1
    if rep:
        print(f"--- prompt {r['prompt']} len {len(r['tokens'])} rep4 {r['rep4']:.3f}: " + "; ".join(f"{tok.decode(list(g))!r}x{n}" for g, n in rep[:8]))
print("most common repeated 4-grams over all generations:")
for g, n in allc.most_common(25): print(n, repr(g))
if len(sys.argv) > 2:
    for i in map(int, sys.argv[2:]):
        print("=====", i, repr(tok.decode(prompts[rows[i]['prompt']])), "=>", rows[i]["text"])
