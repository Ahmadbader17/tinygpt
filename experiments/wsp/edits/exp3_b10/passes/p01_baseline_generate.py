"""Pass 1: baseline sampled generations (unedited) on 64 fixed dev prompts, seed 0."""
import json, sys
sys.path.insert(0, "passes")
from common import *
sb = sandbox()
out = sb.generate(WS / "checkpoint", GEN_PROMPTS, seeds=(0,), decode=SAMPLE,
    purpose="Baseline for paired comparison and source of model-written text: hypothesis that the unedited "
            "model, sampled at T=0.8/top-k 40 on the first 64 dev prompts, shows loop rate near 24% and that its "
            "loops are verbatim re-use of earlier 4-grams (to be attributed to heads in the next read).")
print(json.dumps(out, indent=1)); print("used", sb.used, "remaining", sb.remaining)
