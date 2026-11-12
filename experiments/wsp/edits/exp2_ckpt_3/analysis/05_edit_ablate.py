"""Diagnostic edit (not the repair): zero blocks.3.attn.proj.weight, i.e. remove layer-3 attention's write
(bias kept). Tests whether the isotropic noise found in that tensor is what raises the loss."""
from pathlib import Path
from tinygpt.wsp.edits import load_base_params, write_edit
ws = Path(".")
params = load_base_params(ws / "checkpoint")
new = {k: v.clone() for k, v in params.items()}
new["blocks.3.attn.proj.weight"].zero_()
s = write_edit(ws / "checkpoint", new, ws / "edited" / "ablate_l3_attn_proj", "ablate_l3_attn_proj", Path(__file__))
print(s["tensors_touched"], s["weights_changed"], s["relative_l2"])
