"""Pass 3: read (unedited) with captures on 64 real dev stories; also the baseline loss for later paired comparisons."""
import json, sys
sys.path.insert(0, "passes")
from common import *
rows = dev_story_rows(n=64, length=256, skip=0)
(WS / "work/outputs/p03_rows.json").write_text(json.dumps({"rows": rows, "valid": [256] * 64, "prompt_len": [1] * 64}))
sb = sandbox()
out = sb.read(WS / "checkpoint", rows,
    purpose="Hypothesis: on real stories the same late heads (L5H7, L5H6, L4H5, L5H1, L5H4) carry the copy-specific logit "
            "of 4-gram-repeating tokens, and the model is roughly calibrated there (its probability of the copied "
            "continuation matches how often real text takes it), so the excess repetition in samples comes from "
            "sharpened sampling of this copy pathway rather than from a miscalibrated one. Also gives the unedited "
            "per-token loss on these 64 stories as the paired baseline for testing edits.",
    capture=("attn", "head_out", "mlp_out", "resid_post", "logits", "token_loss"), layers=(0, 1, 2, 3, 4, 5))
(WS / "work/outputs/p03_nll.json").write_text(json.dumps({"nll": out.pop("nll"), "tokens": out.pop("tokens")}))
print(json.dumps(out, indent=1)); print("used", sb.used, "remaining", sb.remaining)
