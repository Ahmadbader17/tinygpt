"""A forward pass that also returns what happened inside: attention patterns and activations.

Used by the budgeted `read` call (diagnosis) and by the scripted verification of Exp 1. The
attention is written as an explicit softmax on purpose, so the pattern is visible. Tests pin
the logits to `GPT.forward` and check that the captured pieces add up to the residual stream.
"""

from __future__ import annotations

import math
from collections.abc import Collection

import torch
from torch import Tensor

from tinygpt.model import GPT, Block

# what can be captured per layer; the result key is the kind followed by the layer number
CAPTURE_KINDS = (
    "attn",  # (B, H, T, T) attention probabilities; row t is where position t looks
    "head_out",  # (B, H, T, C) what each head writes into the residual stream (no bias)
    "attn_out",  # (B, T, C) the attention sublayer's output, bias included
    "mlp_hidden",  # (B, T, 4C) MLP hidden activations after GELU
    "mlp_out",  # (B, T, C) the MLP sublayer's output
    "resid_pre",  # (B, T, C) residual stream entering the block
    "resid_mid",  # (B, T, C) after attention, before the MLP
    "resid_post",  # (B, T, C) leaving the block
)


@torch.inference_mode()
def forward_capture(
    model: GPT, idx: Tensor, want: Collection[str], layers: Collection[int]
) -> tuple[Tensor, dict[str, Tensor]]:
    """Logits (B, T, vocab) and the requested activations, keyed like "attn3" or "resid_mid0"."""
    if model.training:
        raise ValueError("put the model in eval mode first")
    unknown = set(want) - set(CAPTURE_KINDS)
    if unknown:
        raise ValueError(f"unknown capture kinds {sorted(unknown)}; choose from {CAPTURE_KINDS}")
    b, t = idx.shape
    pos = torch.arange(t, device=idx.device)
    x = model.wte(idx) + model.wpe(pos)  # (B, T, C)
    out: dict[str, Tensor] = {}

    def keep(kind: str, layer: int, value: Tensor) -> None:
        if kind in want and layer in layers:
            out[f"{kind}{layer}"] = value

    for i, block in enumerate(model.blocks):
        assert isinstance(block, Block)
        attn = block.attn
        c, h, d = x.shape[-1], attn.n_head, attn.head_dim
        keep("resid_pre", i, x)
        q, k, v = attn.qkv(block.ln1(x)).split(c, dim=2)
        q = q.view(b, t, h, d).transpose(1, 2)  # (B, T, C) -> (B, H, T, D)
        k = k.view(b, t, h, d).transpose(1, 2)
        v = v.view(b, t, h, d).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d)  # (B, H, T, T)
        scores = scores.masked_fill(~attn.mask[:t, :t], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        keep("attn", i, probs)
        y = probs @ v  # (B, H, T, D): each head's mixed values
        if "head_out" in want and i in layers:
            # proj.weight is (C, H*D): head n owns input columns n*D .. (n+1)*D
            w = attn.proj.weight.view(c, h, d)
            out[f"head_out{i}"] = torch.einsum("bhtd,chd->bhtc", y, w)
        a = attn.proj(y.transpose(1, 2).contiguous().view(b, t, c))  # (B, T, C)
        keep("attn_out", i, a)
        x = x + a
        keep("resid_mid", i, x)
        hidden = block.mlp.act(block.mlp.fc(block.ln2(x)))  # (B, T, 4C)
        keep("mlp_hidden", i, hidden)
        m = block.mlp.proj(hidden)
        keep("mlp_out", i, m)
        x = x + m
        keep("resid_post", i, x)
    logits: Tensor = model.lm_head(model.ln_f(x))
    return logits, out


def forward_with_attention(
    model: GPT, idx: Tensor, layers: Collection[int]
) -> tuple[Tensor, dict[int, Tensor]]:
    """Logits and, for each requested layer, attention probabilities (B, H, T, T)."""
    logits, out = forward_capture(model, idx, ("attn",), layers)
    return logits, {layer: out[f"attn{layer}"] for layer in layers}
