"""Look at dev data (no model): prompt lengths, story lengths, repetition in real stories."""
import json
from pathlib import Path
import numpy as np
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp.metrics import has_loop, repeated_ngram_fraction
tok = BPETokenizer.load(Path("tokenizer.json"))
prompts = json.loads(Path("dev/prompts.json").read_text())
print(type(prompts), len(prompts))
p = [x["tokens"] for x in prompts]
print("prompt lens", np.bincount([len(x) for x in p]))
for x in p[:5]: print(repr(tok.decode(x)), x[:6])
stories = [s.strip() for s in Path("dev/stories.txt").read_text().split("\n\n") if s.strip()]
print("stories", len(stories))
ids = [tok.encode(s) for s in stories[:2000]]
L = np.array([len(i) for i in ids]); print("len quantiles", np.quantile(L, [0.1, 0.25, 0.5, 0.75, 0.9]), "share>=255", (L >= 255).mean())
print("real loop rate (first 200 tok)", np.mean([has_loop(i[:200]) for i in ids]), "rep4", np.mean([repeated_ngram_fraction(i[:200], 4) for i in ids]))
print(repr(stories[0][:300]))
