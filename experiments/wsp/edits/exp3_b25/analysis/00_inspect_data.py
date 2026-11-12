"""Free: look at the format of dev/prompts.json and dev/stories.txt (no model run).
Saved after the fact: this was first run as an inline snippet, see journal.md 'Rule notes'."""
import json
from pathlib import Path
import numpy as np
from tinygpt.tokenizer import BPETokenizer
tok = BPETokenizer.load(Path("tokenizer.json"))
pr = json.load(open("dev/prompts.json"))
print(len(pr), pr[0], pr[1])
st = open("dev/stories.txt").read().split("\n\n")
print(len(st)); print(repr(st[0][:300]))
ls = [len(tok.encode(s)) for s in st[:300]]
print(np.mean(ls), np.percentile(ls, [10, 50, 90]))
