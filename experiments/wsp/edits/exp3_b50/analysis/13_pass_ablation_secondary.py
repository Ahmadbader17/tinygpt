"""PASSES 18-20. Causal test: are the weaker copy-signature heads L4H6, L4H4, L3H0, L5H4 also drivers?"""
import sys
sys.path.insert(0, "analysis")
from common import *
from headscale import build

sb = sandbox()
ck = build("test_ablate_secondary4", [(4, 6), (4, 4), (3, 0), (5, 4)], 0.0, __file__)
P = prompts()
pur = ("Hypothesis: heads L4H6, L4H4, L3H0, L5H4 (weaker copy signatures in weights and in pass 2) also carry in-context "
       "copying. Prediction: zeroing them lowers rep-4 on prompts 0-127 seed 0 by 0.003-0.008 from the unedited 0.0279 and "
       "raises loss on dev stories 1-64 by 0.03-0.08.")
for k in (0, 1):
    o = sb.generate(ck, P[64 * k : 64 * k + 64], seeds=(0,), decode=SAMPLED, purpose=pur + f" [generation, prompts {64*k}-{64*k+63}]")
    log(f"secondary ablated g{k}", o)
o = sb.read(ck, story_rows(1, 64), purpose=pur + " [loss, dev stories 1-64]")
log("secondary ablated l0", o); print("  delta vs base (1.64999)", o["mean_nll"] - 1.6499893557179008)
print("used", sb.used, "remaining", sb.remaining)
