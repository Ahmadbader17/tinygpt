"""Final repair. Run from the workspace root: rebuilds repaired/ from checkpoint/ alone, deterministically.

Fault: blocks.5.attn.proj.weight columns 192..223 (the write matrix of layer-5 head 6) are exactly zero.
The original values are not recoverable from the remaining weights. Partial repair: give head 6 the mean
OV circuit of its two nearest neighbours in the same layer (heads 1 and 7, nearest by QK-matrix cosine and
by value-row-space overlap), routed through head 6's own intact value rows:

    P6 = 0.5 * (P1 V1 + P7 V7) pinv(V6)

All numbers come from the checkpoint's weights; 0.5 is the plain average of two neighbours.
"""
from pathlib import Path

import torch

from tinygpt.wsp.edits import load_base_params, write_edit

LAYER, HEAD, NEIGHBOURS = 5, 6, (1, 7)

ws = Path(".")
params = load_base_params(ws / "checkpoint")
new = {name: value.clone() for name, value in params.items()}
qkv = params[f"blocks.{LAYER}.attn.qkv.weight"].double()
proj = params[f"blocks.{LAYER}.attn.proj.weight"].double()


def value_rows(h: int) -> torch.Tensor:  # (32, 256)
    return qkv[512 + 32 * h : 512 + 32 * h + 32]


def write_cols(h: int) -> torch.Tensor:  # (256, 32)
    return proj[:, 32 * h : 32 * h + 32]


assert float(write_cols(HEAD).abs().max()) == 0.0, "expected the damaged (zeroed) head"
target = sum(write_cols(h) @ value_rows(h) for h in NEIGHBOURS) / len(NEIGHBOURS)  # (256, 256)
p6 = target @ torch.linalg.pinv(value_rows(HEAD))  # (256, 32)
new[f"blocks.{LAYER}.attn.proj.weight"][:, 32 * HEAD : 32 * HEAD + 32] = p6.float()

summary = write_edit(ws / "checkpoint", new, ws / "repaired", "repair_l5h6_neighbour_ov", Path(__file__))
print({k: summary[k] for k in ("tensors_touched", "weights_changed", "relative_l2", "edited_sha256")})
