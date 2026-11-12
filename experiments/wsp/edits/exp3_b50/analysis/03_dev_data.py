"""No model: look at the dev prompts and stories, and repetition statistics of real stories."""
import json
from pathlib import Path
import numpy as np
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp.metrics import has_loop, repeated_ngram_fraction

tok = BPETokenizer.load(Path("tokenizer.json"))
prompts = [p["tokens"] for p in json.loads(Path("dev/prompts.json").read_text())]
print(type(prompts), len(prompts))
if isinstance(prompts, dict):
    print(prompts.keys())
ls = [len(p) for p in prompts]
print("prompt lengths", min(ls), max(ls), np.mean(ls))
for p in prompts[:5]:
    print(repr(tok.decode(p)))
stories = [s.strip() for s in Path("dev/stories.txt").read_text().split("\n\n") if s.strip()]
print("stories", len(stories))
enc = [tok.encode(s) for s in stories[:600]]
ln = [len(e) for e in enc]
print("story token lengths: mean", np.mean(ln), "median", np.median(ln), "share>=256", np.mean([l >= 256 for l in ln]))
print(repr(stories[0][:300]))
print("first tokens of story 0", enc[0][:10], "prompt0", prompts[0])
cont = [e[: 200] for e in enc]
print("real stories first 200 tok: loop", np.mean([has_loop(c) for c in cont]), "rep4", np.mean([repeated_ngram_fraction(c, 4) for c in cont]))
