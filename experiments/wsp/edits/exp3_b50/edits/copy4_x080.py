"""Edit v1: multiply the output of the four copy heads by 0.8.

Heads: L5H6, L5H7, L5H1, L4H5. They were identified in three independent ways (see diagnosis.md):
weights (copy-shaped OV, induction-shaped QK through the layer-0 local-context heads), pass 2
(attention to the token that followed the previous occurrence; direct logit contribution concentrated
on repeated n-grams) and passes 4-7 (zeroing them halves loop rate, zeroing four other heads does not).

Factor 0.8 is the evaluation's decoding temperature: sampling at T=0.8 multiplies each logit
contribution by 1.25, and 0.8 * 1.25 = 1 restores the copy heads' contribution to its trained,
calibrated size. Head h of layer L owns columns 32h..32h+31 of blocks.L.attn.proj.weight.

Run from the workspace root: python edits/copy4_x080.py
"""
from pathlib import Path

from tinygpt.wsp.edits import load_base_params, write_edit

NAME = "copy4_x080"
HEADS = [(5, 6), (5, 7), (5, 1), (4, 5)]
FACTOR = 0.8

ws = Path(".")
params = load_base_params(ws / "checkpoint")
new = {k: v.clone() for k, v in params.items()}
for layer, head in HEADS:
    new[f"blocks.{layer}.attn.proj.weight"][:, 32 * head : 32 * head + 32] *= FACTOR
summary = write_edit(ws / "checkpoint", new, ws / "edited" / NAME, NAME, Path(__file__))
print(NAME, "tensors touched", summary["tensors_touched"], "weights changed", summary["weights_changed"],
      "relative L2", round(summary["relative_l2"], 5), "sha", summary["edited_sha256"][:12])
