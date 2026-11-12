"""Generate with a checkpoint under the standard decoder on a block of 64 dev prompts, seed 0.
usage: gen_edit.py <checkpoint dir> <first prompt index> <purpose text file>"""
import json, sys
sys.path.insert(0, "passes")
from common import *
ckpt, start, purpose = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]).read_text().strip()
sb = sandbox()
out = sb.generate(ckpt, PROMPTS[start : start + 64], seeds=(0,), decode=SAMPLE, purpose=purpose)
print(json.dumps(out, indent=1)); print("used", sb.used, "remaining", sb.remaining)
