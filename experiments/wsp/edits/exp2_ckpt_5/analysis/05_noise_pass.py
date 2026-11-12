"""One pass: is the repaired model's remaining gap to the healthy reference just sampling noise?"""
import random, sys
from pathlib import Path
import numpy as np
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp.sandbox import Sandbox

ws = Path(".")
tok = BPETokenizer.load(Path("tokenizer.json"))
text = Path("dev/stories.txt").read_text(encoding="utf-8")
stories = [s.strip() for s in text.split("\n\n") if s.strip()][1:]
rng = random.Random(98765)  # a different draw from analysis/evalsets.py
pick = rng.sample(range(len(stories)), 600)
stream = [t for i in pick for t in [tok.eot_id] + tok.encode(stories[i])]
windows = [stream[s:s + 256] for s in (rng.randrange(0, len(stream) - 256) for _ in range(64))]
sb = Sandbox.open(ws / "work")
o = sb.read(ws / "repaired", windows,
            purpose="H3: the repaired model's 0.07 gap to the healthy 1.688 on the first 64 windows is sampling noise; "
                    "a fresh independent draw of 64 windows should scatter around 1.69 rather than repeat 1.76")
per = np.array(o["nll"]) / np.array(o["tokens"])
print("repaired, fresh windows mean_nll", o["mean_nll"], "per-window std", per.std().round(3), "s.e.", (per.std() / 8).round(3))
print("used", sb.used, "remaining", sb.remaining)
