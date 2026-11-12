"""Build a checkpoint in which the output of the named heads is multiplied by a factor.

Head h of layer L writes through columns 32h..32h+31 of blocks.L.attn.proj.weight; scaling those
columns scales exactly what the head adds to the residual stream and nothing else.
"""
from pathlib import Path
from tinygpt.wsp.edits import load_base_params, write_edit


def build(name, heads, factor, script):
    ws = Path(".")
    params = load_base_params(ws / "checkpoint")
    new = {k: v.clone() for k, v in params.items()}
    for L, h in heads:
        new[f"blocks.{L}.attn.proj.weight"][:, 32 * h : 32 * h + 32] *= factor
    s = write_edit(ws / "checkpoint", new, ws / "edited" / name, name, Path(script))
    print(name, "tensors touched", s["tensors_touched"], "weights changed", s["weights_changed"], "relative L2", round(s["relative_l2"], 5))
    return ws / "edited" / name
