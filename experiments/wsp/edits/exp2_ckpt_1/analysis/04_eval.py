"""Forward passes 1-3: damaged vs repaired on the same fixed dev windows, then whole stories."""
import json
import random
import sys
from pathlib import Path

from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp.sandbox import Sandbox

ws = Path(".")
tok = BPETokenizer.load(ws / "tokenizer.json")
text = (ws / "dev" / "stories.txt").read_text(encoding="utf-8")
stories = [s.strip() for s in text.split("\n\n") if s.strip()]
rng = random.Random(0)
picked = rng.sample(stories[1:-1], 400)  # skip the possibly truncated first and last story
# (a) 64 windows of 256 tokens at random offsets of a token stream of stories joined by <|endoftext|>
stream: list[int] = []
for s in picked[:300]:
    stream += [tok.eot_id] + tok.encode(s)
starts = [rng.randrange(0, len(stream) - 256) for _ in range(64)]
windows = [stream[a:a + 256] for a in starts]
# (b) 64 whole stories from their start
whole = [([tok.eot_id] + tok.encode(s))[:256] for s in picked[300:364]]

sb = Sandbox.open(ws / "work")
which = sys.argv[1]
if which == "damaged_windows":
    out = sb.read(ws / "checkpoint", windows, purpose="Baseline: how far is the damaged checkpoint from the healthy 1.688 on 64 random 256-token dev windows? Hypothesis: the 64 inflated gains of blocks.5.ln2.weight overdrive the last MLP and raise the loss clearly.")
elif which == "repaired_windows":
    out = sb.read(ws / "repaired", windows, purpose="Test of the repair: dividing the 64 inflated gains of blocks.5.ln2.weight by 4.5 should bring the loss on the same 64 windows back to about the healthy 1.688.")
elif which == "repaired_whole":
    out = sb.read(ws / "repaired", whole, purpose="Second check of the repair on 64 whole dev stories scored from <|endoftext|>: expect about the healthy 1.605.")
elif which == "damaged_whole":
    out = sb.read(ws / "checkpoint", whole, purpose="Baseline of the damaged checkpoint on the same 64 whole dev stories, to size the damage on that measure.")
else:
    raise SystemExit("unknown")
print(which, "mean_nll", out["mean_nll"], "tokens", sum(out["tokens"]), "used", sb.used, "remaining", sb.remaining)
(ws / "analysis" / f"04_{which}.json").write_text(json.dumps({"mean_nll": out["mean_nll"], "nll": out["nll"], "tokens": out["tokens"]}))
