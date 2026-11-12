"""Repair: remove the foreign rank-1 component from blocks.2.mlp.proj.weight.

Reasoning (weights only, see journal.md): this tensor has one singular value of 36.2 while its
next ones are 3.10, 3.05, 2.82 and every other mlp.proj in the model tops out at 2.4 to 4.8.
The output vector u of that direction is read by later layers exactly as a random direction
would be (norms 1.62 / 1.27 against 2.2 to 3.4 for the learned directions), and its input
vector v is unrelated to the fc row norms, so it is not a learned direction that was scaled up
but a component that does not belong to the matrix. The edit therefore subtracts
sigma_1 * u_1 * v_1^T, the top term of the tensor's own SVD. No number comes from activations.
"""
from pathlib import Path

import torch

from tinygpt.wsp.edits import load_base_params, write_edit

TENSOR = "blocks.2.mlp.proj.weight"


def main() -> None:
    ws = Path(".")
    params = load_base_params(ws / "checkpoint")
    new = {name: value.clone() for name, value in params.items()}
    w = params[TENSOR].double()
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    fixed = w - s[0] * torch.outer(u[:, 0], vh[0])
    new[TENSOR] = fixed.to(torch.float32)
    summary = write_edit(ws / "checkpoint", new, ws / "repaired", "remove_rank1_l2_mlp_proj", Path(__file__))
    print("removed sigma_1 =", round(s[0].item(), 4), "next:", [round(x, 3) for x in s[1:4].tolist()])
    print("tensors touched", summary["tensors_touched"], "relative_l2", summary["relative_l2"],
          "tensor rel l2", summary["per_tensor"][TENSOR]["relative_l2"])


if __name__ == "__main__":
    main()
