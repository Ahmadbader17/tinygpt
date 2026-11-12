"""Edit v2 (revision 1): multiply the output of all eight copy heads by 0.8.

Heads: L5H6, L5H7, L5H1, L4H5 (as in v1) and L4H6, L4H4, L3H0, L5H4. All eight have a copy-shaped OV
circuit and/or an induction-shaped QK circuit in the weights, attend to the token that followed the
previous occurrence of the current token (pass 2), and lower repetition when zeroed (passes 4-5 for
the first four, 18-20 for the second four); four other heads of the same layers do not (passes 6-7).

Factor 0.8 is the evaluation's decoding temperature: sampling at T=0.8 multiplies each logit
contribution by 1.25, and 0.8 * 1.25 = 1 restores the copy heads' contribution to its trained,
calibrated size. Head h of layer L owns columns 32h..32h+31 of blocks.L.attn.proj.weight.

Run from the workspace root: python edits/copy8_x080.py
"""
from pathlib import Path

from tinygpt.wsp.edits import load_base_params, write_edit

NAME = "copy8_x080"
HEADS = [(5, 6), (5, 7), (5, 1), (4, 5), (4, 6), (4, 4), (3, 0), (5, 4)]
FACTOR = 0.8

ws = Path(".")
params = load_base_params(ws / "checkpoint")
new = {k: v.clone() for k, v in params.items()}
for layer, head in HEADS:
    new[f"blocks.{layer}.attn.proj.weight"][:, 32 * head : 32 * head + 32] *= FACTOR
summary = write_edit(ws / "checkpoint", new, ws / "edited" / NAME, NAME, Path(__file__))
print(NAME, "tensors touched", summary["tensors_touched"], "weights changed", summary["weights_changed"],
      "relative L2", round(summary["relative_l2"], 5), "sha", summary["edited_sha256"][:12])
