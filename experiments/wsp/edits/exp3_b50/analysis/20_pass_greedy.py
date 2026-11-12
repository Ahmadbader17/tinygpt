"""PASSES 39-40: greedy decoding, unedited vs v2, prompts 0-63."""
import sys
sys.path.insert(0, "analysis")
from common import *

sb = sandbox()
P = prompts()[:64]
pur = ("Hypothesis: the copy heads also drive loops under greedy decoding. Prediction: v2 (eight copy heads x0.8) lowers greedy "
       "loop rate on prompts 0-63 by 0-10 points and rep-4 by 10-25% relative to the unedited model.")
for name, ck in (("base", WS / "checkpoint"), ("v2", WS / "edited" / "copy8_x080")):
    o = sb.generate(ck, P, seeds=None, decode=GREEDY, purpose=pur + f" [{name}, greedy]")
    log(f"{name} greedy", o)
print("used", sb.used, "remaining", sb.remaining)
