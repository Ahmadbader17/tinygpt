"""Pass 2: read (unedited) with captures on the model's own pass-1 generations."""
import json, sys
sys.path.insert(0, "passes")
from common import *
T = 212
gens = json.loads((WS / "work/outputs/generations_call001.json").read_text())
filler = dev_story_rows(n=64, length=256, skip=1000)   # only used to pad rows to equal length; masked out in analysis
rows, valid = [], []
for g, f in zip(gens, filler):
    core = GEN_PROMPTS[g["prompt"]] + g["tokens"]
    row = (core + [tok.eot_id] + f[1:])[:T]
    rows.append(row); valid.append(min(len(core), T))
(WS / "work/outputs/p02_rows.json").write_text(json.dumps({"rows": rows, "valid": valid, "prompt_len": [len(GEN_PROMPTS[g["prompt"]]) for g in gens]}))
sb = sandbox()
out = sb.read(WS / "checkpoint", rows,
    purpose="Hypothesis: on the model's own sampled text, the logit of tokens that complete an already-seen 4-gram is "
            "supplied by a small set of late attention heads that attend to the token following an earlier occurrence "
            "of the current token (induction/copy heads), not spread evenly over heads and MLPs. Captures attention, "
            "per-head outputs, MLP outputs and residuals at all layers to attribute repeat-token logits per component.",
    capture=("attn", "head_out", "mlp_out", "resid_post", "logits", "token_loss"), layers=(0, 1, 2, 3, 4, 5))
out.pop("nll"); out.pop("tokens")
print(json.dumps(out, indent=1)); print("used", sb.used, "remaining", sb.remaining)
