"""Pass 2: one read with captures on 32 baseline generations + 32 real dev stories (160 tokens each)."""
import json, sys
sys.path.insert(0, "analysis")
from common import ws, tok, PROMPTS, sandbox
T = 160
rows = json.load(open("work/outputs/generations_call001.json"))
gen_seqs = [PROMPTS[r["prompt"]] + r["tokens"] for r in rows]
gen_seqs = [s[:T] for s in gen_seqs if len(s) >= T][:32]
stories = open("dev/stories.txt").read().split("\n\n")
real = []
for s in stories[1:]:
    ids = tok.encode(s.strip())
    if len(ids) >= T:
        real.append(ids[:T])
    if len(real) == 32:
        break
print(len(gen_seqs), len(real))
json.dump({"gen": gen_seqs, "real": real}, open("analysis/04_sequences.json", "w"))
sb = sandbox()
out = sb.read(ws / "checkpoint", gen_seqs + real,
    purpose="Test H1: at positions where the next token completes an already-seen 4-gram, the direct logit "
            "contribution to that token comes mainly from the weight-identified late copy heads (4.6, 5.0, 5.6, 5.1, 5.7), "
            "which there attend to the earlier occurrence; compare own generations with real stories.",
    capture=("attn", "head_out", "mlp_out", "resid_pre", "resid_post", "token_loss"),
    layers=(0, 1, 2, 3, 4, 5))
print({k: v for k, v in out.items() if k not in ("nll", "tokens")}, "used", sb.used, "remaining", sb.remaining)
