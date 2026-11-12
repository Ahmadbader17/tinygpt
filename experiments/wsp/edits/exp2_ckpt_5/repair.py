"""Final repair: undo the gain on head 5 of block 5's attention output projection.

Diagnosis (weights only, see journal.md): the 32 input columns of blocks.5.attn.proj.weight
owned by head 5 (columns 160..191) have Frobenius norm 22.43, while every other head in the
model has |O_h| / |V_h| close to 1 (layer 5: 1.001..1.047), where V_h is the head's value
rows of qkv.weight. Head 5's ratio is 5.66 and it is uniform over its 32 columns, i.e. the
block was multiplied by one scalar.

The divisor is derived from the weights alone: head 5's |O|/|V| ratio divided by the median of
that ratio over the other seven heads of the same layer, so that after the edit head 5 has the
layer's typical output/value balance. No activations or model outputs enter this number.
"""
from pathlib import Path

import torch

from tinygpt.wsp.edits import load_base_params, write_edit

LAYER, HEAD, D = 5, 5, 32


def main() -> None:
    ws = Path(".")
    params = load_base_params(ws / "checkpoint")
    new = {name: value.clone() for name, value in params.items()}
    proj = params[f"blocks.{LAYER}.attn.proj.weight"].double()
    qkv = params[f"blocks.{LAYER}.attn.qkv.weight"].double()
    ratios = []
    for h in range(8):
        o = proj[:, D * h: D * h + D].norm()
        v = qkv[512 + D * h: 512 + D * h + D].norm()
        ratios.append(o / v)
    others = torch.stack([r for h, r in enumerate(ratios) if h != HEAD])
    divisor = float(ratios[HEAD] / others.median())
    w = new[f"blocks.{LAYER}.attn.proj.weight"]
    w[:, D * HEAD: D * HEAD + D] = (proj[:, D * HEAD: D * HEAD + D] / divisor).float()
    summary = write_edit(ws / "checkpoint", new, ws / "repaired", "repair", Path(__file__))
    print(f"divisor {divisor:.4f}; tensors touched {summary['tensors_touched']}, "
          f"weights changed {summary['weights_changed']}, relative L2 {summary['relative_l2']:.4f}")


if __name__ == "__main__":
    main()
