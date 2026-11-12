"""Repair: undo the scaling of 64 gains in blocks.5.ln2.weight.

Reasoning (weights only, see journal.md): 64 of the 256 entries of blocks.5.ln2.weight sit in a
separate cluster around 5.4 while the other 192 sit around 1.22, which is where the depth trend
of every other LayerNorm gain puts them. The cluster keeps the fine structure of a healthy gain
vector (it correlates 0.93 with the column norms of blocks.5.mlp.fc.weight, as gains do in every
layer), so the entries were multiplied by a common factor, not replaced. Three weights-only
estimates of the factor give 4.45 to 4.51; I take the round value 4.5 and divide it out.

The threshold 2.0 lies in the empty gap between the clusters (largest healthy-looking gain
1.352, smallest scaled gain 3.862). Run from the workspace root.
"""
from pathlib import Path

from tinygpt.wsp.edits import load_base_params, write_edit

TENSOR = "blocks.5.ln2.weight"
THRESHOLD = 2.0  # in the gap between the two clusters of gains
FACTOR = 4.5  # chosen scalar: the common factor the 64 gains were multiplied by


def main() -> None:
    ws = Path(".")
    params = load_base_params(ws / "checkpoint")
    new = {name: value.clone() for name, value in params.items()}
    gain = new[TENSOR]
    scaled = gain > THRESHOLD
    assert int(scaled.sum()) == 64, f"expected 64 scaled gains, found {int(scaled.sum())}"
    gain[scaled] = gain[scaled] / FACTOR
    summary = write_edit(ws / "checkpoint", new, ws / "repaired", "repair", Path(__file__))
    print("tensors touched:", summary["tensors_touched"], "weights changed:", summary["weights_changed"],
          "relative L2:", round(summary["relative_l2"], 5))


if __name__ == "__main__":
    main()
