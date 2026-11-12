"""Pass 9: read A2 (copyheads_075) on the unedited model's own generations (the pass-2 rows) with logits,
to compare, on identical contexts, the sampler's probability of completing an already-seen 4-gram."""
import json, sys
import numpy as np
sys.path.insert(0, "passes")
from common import *
meta = json.loads((WS / "work/outputs/p02_rows.json").read_text())
sb = sandbox()
out = sb.read(WS / "edited" / "copyheads_075", meta["rows"],
    purpose="Sampled generations are too noisy to resolve the expected ~10% effect of edit copyheads_075 (64 texts give +-0.008 on "
            "rep-4). Low-noise test of the same hypothesis on identical contexts: on the unedited model's own pass-1 texts, the "
            "edited model, under the standard decoder (T=0.8, top-k 40), assigns lower probability than the unedited model "
            "(pass-2 logits) to tokens that would complete an already-seen 4-gram. Prediction: the expected number of "
            "4-gram-repeating tokens per position falls by roughly 5-15%; no fall would mean the edit is not justified.",
    capture=("logits",))
out.pop("nll"); out.pop("tokens")
print(json.dumps(out, indent=1)); print("used", sb.used, "remaining", sb.remaining)
