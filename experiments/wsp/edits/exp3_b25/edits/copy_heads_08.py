"""Edit v1: scale the output of the seven copy heads by 0.8 (the evaluation's sampling temperature).

Heads (layer, head): 3.0, 4.5, 4.6, 5.1, 5.4, 5.6, 5.7. Chosen because (a) their OV circuit, computed from
the weights alone, maps an attended token to its own unembedding direction (analysis/06_copy_extended.py)
and (b) a capture pass showed that they carry the extra logit of tokens completing a repeated 4-gram.
The factor 0.8 is a chosen scalar: the decoding temperature (see journal.md, "Edit v1").
Run from the workspace root:  python edits/copy_heads_08.py
"""
from pathlib import Path
from tinygpt.wsp.edits import load_base_params, write_edit

NAME = "copy_heads_08"
HEADS = [(3, 0), (4, 5), (4, 6), (5, 1), (5, 4), (5, 6), (5, 7)]
FACTOR = 0.8
HEAD_DIM = 32

ws = Path(".")
params = load_base_params(ws / "checkpoint")
new = {name: value.clone() for name, value in params.items()}
for layer, head in HEADS:
    cols = slice(HEAD_DIM * head, HEAD_DIM * head + HEAD_DIM)
    new[f"blocks.{layer}.attn.proj.weight"][:, cols] *= FACTOR
summary = write_edit(ws / "checkpoint", new, ws / "edited" / NAME, NAME, Path(__file__))
print({k: summary[k] for k in ("tensors_touched", "weights_changed", "fraction_changed", "relative_l2", "edited_sha256")})
