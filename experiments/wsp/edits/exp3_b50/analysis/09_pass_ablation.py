"""PASSES 3-7. Causal test of the copy-head hypothesis by zero-ablation (a test, not a candidate edit).

core4    = L5H6, L5H7, L5H1, L4H5: the heads that pass 2 singled out (DLA on repeated n-grams, attention
           to the token after the previous occurrence) and whose weights show copy OV / induction QK.
control4 = L5H0, L5H2, L5H3, L4H1: same layers, same count, none of the three signatures.
Same prompts and seed as pass 1 (common random numbers), and dev stories 1-64 for the loss.
"""
import sys
sys.path.insert(0, "analysis")
from common import *
from headscale import build

sb = sandbox()
core = build("test_ablate_core4", [(5, 6), (5, 7), (5, 1), (4, 5)], 0.0, __file__)
ctrl = build("test_ablate_control4", [(5, 0), (5, 2), (5, 3), (4, 1)], 0.0, __file__)
rows = story_rows(1, 64)
P = prompts()[:64]
o = sb.read(WS / "checkpoint", rows, purpose="Baseline loss on dev stories 1-64 (<eot>+story, cut at 256 tokens), the reference "
            "for every paired loss comparison of an ablation or an edit.")
log("pass3 base loss", o); base = o
o = sb.generate(core, P, seeds=(0,), decode=SAMPLED, purpose="Hypothesis: heads L5H6, L5H7, L5H1, L4H5 carry the in-context copying "
            "that produces repeated 4-grams. Prediction: with their output zeroed, loop rate and rep-4 on prompts 0-63 seed 0 "
            "fall far below the unedited model's 0.234 / 0.0255 (pass 1).")
log("pass4 core4 ablated gen", o)
o = sb.read(core, rows, purpose="Same hypothesis, cost side: how much of the loss on real stories depends on the four copy heads. "
            "Prediction: zeroing them raises loss well above the 0.02 tolerance, i.e. copying is useful and cannot simply be removed.")
log("pass5 core4 ablated loss", o); print("  delta vs base", o["mean_nll"] - base["mean_nll"])
o = sb.generate(ctrl, P, seeds=(0,), decode=SAMPLED, purpose="Control for the copy-head hypothesis: zeroing four heads of the same layers "
            "without copy signatures (L5H0, L5H2, L5H3, L4H1). Prediction: loop rate and rep-4 do not fall as they do for the copy heads.")
log("pass6 control4 ablated gen", o)
o = sb.read(ctrl, rows, purpose="Control, cost side: loss change from zeroing the four non-copy heads, to compare with the copy heads' cost.")
log("pass7 control4 ablated loss", o); print("  delta vs base", o["mean_nll"] - base["mean_nll"])
print("used", sb.used, "remaining", sb.remaining)
