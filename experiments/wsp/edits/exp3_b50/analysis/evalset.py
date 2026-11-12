"""Run the fixed evaluation sets on one checkpoint. Usage: evalset.py <checkpoint dir> <label> <what...>
what: g0..g4 = sampled generation on prompt block k (prompts 64k..64k+63, seed 0), h0..h4 = same blocks, seed 1,
      l0 = loss on dev stories 1-64, l1 = loss on dev stories 65-128.
Each item is one pass. The purpose string is given per label below.
"""
import sys
sys.path.insert(0, "analysis")
from common import *

ckpt, label, purpose = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sb = sandbox()
P = prompts()
for w in sys.argv[4:]:
    k = int(w[1])
    if w[0] in "gh":
        seed = 0 if w[0] == "g" else 1
        o = sb.generate(ckpt, P[64 * k : 64 * k + 64], seeds=(seed,), decode=SAMPLED,
                        purpose=f"{purpose} [generation, prompts {64*k}-{min(64*k+63, len(P)-1)}, seed {seed}]")
    else:
        o = sb.read(ckpt, story_rows(1 + 64 * k, 64), purpose=f"{purpose} [loss, dev stories {1+64*k}-{64+64*k}]")
    log(f"{label} {w}", o)
print("used", sb.used, "remaining", sb.remaining)
