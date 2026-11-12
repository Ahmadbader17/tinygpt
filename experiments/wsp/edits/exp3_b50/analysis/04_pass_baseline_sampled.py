"""PASS 1 (generate, unedited, sampled). Prompts 0..63, seed 0."""
import sys
sys.path.insert(0, "analysis")
from common import *

sb = sandbox()
out = sb.generate(WS / "checkpoint", prompts()[:64], seeds=(0,), decode=SAMPLED,
    purpose="Baseline for paired comparisons and source of loop examples: the unedited model under "
            "the evaluation decoding (T=0.8, top-k 40) on dev prompts 0-63, seed 0. Hypothesis: loops are "
            "re-uses of earlier 4-grams of the same story (copying from context), not token stutter.")
log("pass1", out); print("used", sb.used, "remaining", sb.remaining)
