"""Pass 5: generate with probe A (four copy-promoting heads ablated), standard decoding, same prompts and seed."""
import json, sys
sys.path.insert(0, "passes")
from common import *
sb = sandbox()
out = sb.generate(WS / "edited" / "probe_copyheads_ablate", GEN_PROMPTS, seeds=(0,), decode=SAMPLE,
    purpose="Causal test of the hypothesis 'heads L4H5, L4H4, L5H7, L5H6 drive repetition'. Their direct log-odds push "
            "toward 4-gram-completing tokens is small (about 0.27 logits summed) and later MLPs push the other way, so my "
            "prediction is that even removing them entirely leaves the loop rate near the baseline 0.234 (above 0.19), "
            "i.e. repetition is not localised in these heads. A large drop would refute that and justify a head edit.")
print(json.dumps(out, indent=1)); print("used", sb.used, "remaining", sb.remaining)
