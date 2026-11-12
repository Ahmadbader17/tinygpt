"""Candidate edit (hypothesis test, not the declared repair unless it passes): fill the zeroed write matrix of
L5 head 6 with the mean OV circuit of its two weight-neighbours (h1, h7), routed through head 6's own value rows:
    P6 = 0.5 * (P1 V1 + P7 V7) V6^+
Every number comes from the checkpoint's weights; 0.5 is the plain average of two neighbours."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params, write_edit

ws = Path(".")
params = load_base_params(ws / "checkpoint")
new = {k: v.clone() for k, v in params.items()}
qkv = params["blocks.5.attn.qkv.weight"].double(); pr = params["blocks.5.attn.proj.weight"].double()
V = lambda h: qkv[512 + 32*h: 512 + 32*h + 32]
P = lambda h: pr[:, 32*h: 32*h + 32]
target = 0.5 * (P(1) @ V(1) + P(7) @ V(7))          # 256 x 256
P6 = target @ torch.linalg.pinv(V(6))                # 256 x 32
print("surrogate P6 fro", float(P6.norm()), " |V6|", float(V(6).norm()), " |P1|,|P7|", float(P(1).norm()), float(P(7).norm()))
new["blocks.5.attn.proj.weight"][:, 192:224] = P6.float()
s = write_edit(ws / "checkpoint", new, ws / "edited" / "surrogate_h1h7", "surrogate_h1h7", Path(__file__))
print({k: s[k] for k in ("tensors_touched", "weights_changed", "relative_l2")})
