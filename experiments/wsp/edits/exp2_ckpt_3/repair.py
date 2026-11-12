"""Repair of blocks.3.attn.proj.weight: remove as much as possible of the iid Gaussian noise added to it.

Reasoning (weights only, see journal.md):
* In every healthy attention layer of this checkpoint the output projection and the value map are balanced
  per head: O_h^T O_h ~= V_h V_h^T =: G_h (O_h = proj.weight[:, 32h:32h+32], V_h = value rows of qkv.weight).
* In layer 3, O_h^T O_h ~= G_h + c*I with the same c ~ 0.95 for all 8 heads: the signature of
  proj = W0 + N, N iid with variance sigma^2, c = 256*sigma^2.
* sigma^2 is taken from that balance over the whole tensor: (||proj||_F^2 - ||V||_F^2) / 65536.
* With row prior covariance G_h/256 for the clean O_h and noise covariance sigma^2*I, the posterior mean of
  the clean block is the Wiener filter  O_h <- O_h G_h (G_h + 256 sigma^2 I)^-1 .
No activations, model outputs or text are used; every number comes from the checkpoint's own weights.
"""
from pathlib import Path

import torch

from tinygpt.wsp.edits import load_base_params, write_edit

NAME = "blocks.3.attn.proj.weight"
QKV = "blocks.3.attn.qkv.weight"
N_HEAD, HEAD_DIM, D = 8, 32, 256


def repaired_proj(params):
    W = params[NAME].double()
    V = params[QKV].double()[2 * D:]  # value rows
    noise_var = (W.norm() ** 2 - V.norm() ** 2) / W.numel()  # sigma^2 from the O/V balance
    c = D * noise_var  # noise power per input column
    out = W.clone()
    eye = torch.eye(HEAD_DIM, dtype=torch.float64)
    for h in range(N_HEAD):
        sl = slice(HEAD_DIM * h, HEAD_DIM * (h + 1))
        G = V[sl] @ V[sl].T  # (32, 32) clean second moment of this head's input columns
        gain = G @ torch.linalg.inv(G + c * eye)  # Wiener gain, symmetric
        out[:, sl] = W[:, sl] @ gain
    return out.float(), float(noise_var.sqrt())


def main():
    ws = Path(".")
    params = load_base_params(ws / "checkpoint")
    new = {k: v.clone() for k, v in params.items()}
    new[NAME], sigma = repaired_proj(params)
    s = write_edit(ws / "checkpoint", new, ws / "repaired", "wiener_l3_attn_proj", Path(__file__))
    print(f"sigma_hat={sigma:.5f} touched={s['tensors_touched']} changed={s['weights_changed']} "
          f"rel_l2={s['per_tensor'][NAME]['relative_l2']:.4f} new_fro={new[NAME].norm():.3f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
