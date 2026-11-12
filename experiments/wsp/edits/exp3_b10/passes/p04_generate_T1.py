"""Pass 4: unedited model, same 64 prompts and seed, but sampled at temperature 1.0 (top-k 40 kept)."""
import json, sys
sys.path.insert(0, "passes")
from common import *
sb = sandbox()
out = sb.generate(WS / "checkpoint", GEN_PROMPTS, seeds=(0,), decode=DecodeConfig(200, temperature=1.0, top_k=40),
    purpose="Hypothesis: the excess repetition over real stories is produced by the sampler's sharpening (T=0.8), not by an "
            "over-active copy circuit: passes 2-3 showed the model's probability of a 4-gram-completing token is calibrated "
            "(slightly under-confident) on real text. Prediction: at T=1.0 with the same prompts, seed and top-k, loop rate "
            "and rep-4 fall to about the real-story level (0.19, 0.018) from 0.234 / 0.0255.")
print(json.dumps(out, indent=1)); print("used", sb.used, "remaining", sb.remaining)
