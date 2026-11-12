"""Edit A1: scale the output of heads L4H5, L4H4, L5H7, L5H6 by 0.5.

Which heads: the four whose output raises the log-odds of 4-gram-completing tokens on both model-written and real
text (analysis/09), confirmed causally by probe A (zeroing them halves loop rate and rep-4, pass 5).
How much: 0.5, the midpoint between the unedited model and the ablation. Ablation removes legitimate entity copying
and overshoots the real-story repetition level; the loss cost of scaling is second order in (1 - factor) while the
effect on repetition is first order. The factor is chosen by this reasoning, not computed from activations/outputs.
Implementation: head h owns input columns 32h..32h+31 of blocks.L.attn.proj.weight; scaling them scales everything
the head writes (the shared proj bias is untouched).
Run from the workspace root: python edits/copyheads_half.py
"""
from pathlib import Path
from tinygpt.wsp.edits import load_base_params, write_edit

HEADS = [(4, 5), (4, 4), (5, 7), (5, 6)]
FACTOR = 0.5
NAME = "copyheads_half"
ws = Path(".")
params = load_base_params(ws / "checkpoint")
new = {name: value.clone() for name, value in params.items()}
for layer, head in HEADS:
    new[f"blocks.{layer}.attn.proj.weight"][:, 32 * head : 32 * head + 32] *= FACTOR
summary = write_edit(ws / "checkpoint", new, ws / "edited" / NAME, NAME, Path(__file__))
print({k: summary[k] for k in ("edit", "tensors_touched", "weights_changed", "relative_l2", "edited_sha256")})
