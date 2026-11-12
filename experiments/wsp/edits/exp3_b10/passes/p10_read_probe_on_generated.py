"""Pass 10: read probe A (four heads zeroed) on the pass-2 rows with logits: the one-step copy-probability measure
for the ablation, to calibrate dose-response and the link between the one-step measure and sampled rep-4."""
import json, sys
sys.path.insert(0, "passes")
from common import *
meta = json.loads((WS / "work/outputs/p02_rows.json").read_text())
sb = sandbox()
out = sb.read(WS / "edited" / "probe_copyheads_ablate", meta["rows"],
    purpose="The factor-0.75 edit lowers the one-step sampler probability of completing a seen 4-gram by only 1.6% (pass 9). "
            "Hypothesis: the effect of scaling these four heads is linear in the step and the one-step measure understates "
            "sampled rep-4 by a constant factor of about 5 (as it does for temperature: -6.5% one-step vs -36% sampled at "
            "T=1.0). Prediction for the full ablation on the same contexts: one-step ratio between 0.90 and 0.94 (4 x 1.6% "
            "if linear; 10% if the factor 5 links it to the sampled -49%). A ratio far below 0.90 would mean the dose-response "
            "is convex and the 0.75 edit keeps less than a quarter of the ablation effect.",
    capture=("logits",))
out.pop("nll"); out.pop("tokens")
print(json.dumps(out, indent=1)); print("used", sb.used, "remaining", sb.remaining)
