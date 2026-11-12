"""Probe A (a diagnostic edit, not a candidate for the final version): zero the output of heads L4H5, L4H4, L5H7, L5H6.

These four heads are the only ones whose direct output raises the log-odds of the token that would complete an
already-seen 4-gram, on both the model's own text and real text (analysis/09_second_order.py), and they attend about
twice as much as other heads to the position after an earlier occurrence of the current token. Zeroing their columns
of attn.proj.weight (head h owns input columns 32h..32h+31) removes everything they write. It is the upper bound of
any "scale the copy heads down" edit: if even this does not lower the loop rate, no factor between 0 and 1 will.
The factor 0 is an ablation; no number comes from activations.
Run from the workspace root: python edits/probe_copyheads_ablate.py
"""
from pathlib import Path
from tinygpt.wsp.edits import load_base_params, write_edit

HEADS = [(4, 5), (4, 4), (5, 7), (5, 6)]
NAME = "probe_copyheads_ablate"
ws = Path(".")
params = load_base_params(ws / "checkpoint")
new = {name: value.clone() for name, value in params.items()}
for layer, head in HEADS:
    new[f"blocks.{layer}.attn.proj.weight"][:, 32 * head : 32 * head + 32] = 0.0
summary = write_edit(ws / "checkpoint", new, ws / "edited" / NAME, NAME, Path(__file__))
print({k: summary[k] for k in ("edit", "tensors_touched", "weights_changed", "relative_l2", "edited_sha256")})
