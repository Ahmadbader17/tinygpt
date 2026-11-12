"""Edit A2 (revision 1 of A1): scale the output of heads L4H5, L4H4, L5H7, L5H6 by 0.75.

Which heads: the four whose output raises the log-odds of 4-gram-completing tokens on both model-written and real
text (analysis/09), confirmed causally by probe A (zeroing them halves loop rate and rep-4, pass 5).
How much: 0.75. A1 (factor 0.5, edits/copyheads_half.py) cost +0.0225 nats on real stories, over the 0.02 limit, so the
step is halved: the loss cost of scaling is second order in (1 - factor), so it should fall to about a quarter,
while the effect on repetition is first order. (Ablation, factor 0, removes legitimate entity copying
and overshoots the real-story repetition level.) The factor is "half of the A1 step", chosen by this reasoning;
it is not solved from the measured loss and not computed from activations.
Implementation: head h owns input columns 32h..32h+31 of blocks.L.attn.proj.weight; scaling them scales everything
the head writes (the shared proj bias is untouched).
Run from the workspace root: python edits/copyheads_075.py
"""
from pathlib import Path
from tinygpt.wsp.edits import load_base_params, write_edit

HEADS = [(4, 5), (4, 4), (5, 7), (5, 6)]
FACTOR = 0.75
NAME = "copyheads_075"
ws = Path(".")
params = load_base_params(ws / "checkpoint")
new = {name: value.clone() for name, value in params.items()}
for layer, head in HEADS:
    new[f"blocks.{layer}.attn.proj.weight"][:, 32 * head : 32 * head + 32] *= FACTOR
summary = write_edit(ws / "checkpoint", new, ws / "edited" / NAME, NAME, Path(__file__))
print({k: summary[k] for k in ("edit", "tensors_touched", "weights_changed", "relative_l2", "edited_sha256")})
