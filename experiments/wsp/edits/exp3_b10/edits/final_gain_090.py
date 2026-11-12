"""Edit B: multiply the final LayerNorm's gain and bias by 0.9 (logits scale by exactly 0.9).

Reasoning (see diagnosis.md): logits = E @ (ln_f.weight * normalise(x) + ln_f.bias) with no head bias, so scaling
ln_f.weight and ln_f.bias by s multiplies every logit by s -- the model's own temperature becomes 1/s. The diagnosis
found that the excess repetition is produced by the sampler's sharpening (T=0.8), not by a copy circuit: at T=1.0 the
unedited model repeats like real stories. s = 0.9 undoes half of that sharpening (effective T = 0.8/0.9 = 0.89).
The scalar 0.9 is chosen by this reasoning; it is not computed from activations or outputs.
Run from the workspace root: python edits/final_gain_090.py
"""
from pathlib import Path
from tinygpt.wsp.edits import load_base_params, write_edit

S = 0.9
NAME = "final_gain_090"
ws = Path(".")
params = load_base_params(ws / "checkpoint")
new = {name: value.clone() for name, value in params.items()}
new["ln_f.weight"] = params["ln_f.weight"] * S
new["ln_f.bias"] = params["ln_f.bias"] * S
summary = write_edit(ws / "checkpoint", new, ws / "edited" / NAME, NAME, Path(__file__))
print({k: summary[k] for k in ("edit", "tensors_touched", "weights_changed", "relative_l2", "edited_sha256")})
