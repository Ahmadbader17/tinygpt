"""PASS 2 (read with capture, unedited). 32 model generations from pass 1 (looping ones first) and
32 real dev stories, all cut to 160 tokens."""
import json, sys
sys.path.insert(0, "analysis")
from common import *

T = 160
P = prompts()
gens = json.loads((WS / "work/outputs/generations_call001.json").read_text())
gens = sorted(gens, key=lambda r: (not r["loop"], r["prompt"]))
rows_gen = [(P[r["prompt"]] + r["tokens"])[:T] for r in gens if len(P[r["prompt"]]) + len(r["tokens"]) >= T][:32]
real = [tok.encode(s) for s in stories()[1:200]]
rows_real = [e[:T] for e in real if len(e) >= T][:32]
assert len(rows_gen) == 32 and len(rows_real) == 32
json.dump({"gen": rows_gen, "real": rows_real}, open("analysis/06_rows.json", "w"))
sb = sandbox()
out = sb.read(WS / "checkpoint", rows_gen + rows_real,
    purpose="Which heads copy from context. Hypothesis: a few late-layer heads (weights-only candidates "
            "L4H6, L5H6, L5H0, L5H7) attend to the token that followed the previous occurrence of the "
            "current token and write that token's logit; their direct logit contribution to the true next "
            "token is much larger at positions that complete an already-seen n-gram than elsewhere, on "
            "model generations and on real stories. Second hypothesis: the model's probability of "
            "repeating an n-gram is calibrated on real stories at T=1, so the excess repetition comes "
            "from sharpened decoding of a copy signal, not from a mis-trained component.",
    capture=("attn", "head_out", "mlp_out", "resid_post", "logits", "token_loss"),
    layers=(0, 1, 2, 3, 4, 5))
log("pass2", out); print("used", sb.used, "remaining", sb.remaining)
