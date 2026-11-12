"""Attention patterns of a forward pass, for reading heads (Exp 1) and diagnosis (Exp 2).

Written as an explicit softmax on purpose, so the pattern is visible. A test pins its logits
to `GPT.forward`, so the patterns describe the same computation the model does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from tinygpt.model import GPT, Block


@torch.inference_mode()
def forward_with_attention(
    model: GPT, idx: Tensor, layers: Sequence[int]
) -> tuple[Tensor, dict[int, Tensor]]:
    """Logits (B, T, vocab) and, for each requested layer, attention probabilities (B, H, T, T)."""
    if model.training:
        raise ValueError("put the model in eval mode first")
    b, t = idx.shape
    pos = torch.arange(t, device=idx.device)
    x = model.wte(idx) + model.wpe(pos)  # (B, T, C)
    captured: dict[int, Tensor] = {}
    for i, block in enumerate(model.blocks):
        assert isinstance(block, Block)
        attn = block.attn
        c = x.shape[-1]
        q, k, v = attn.qkv(block.ln1(x)).split(c, dim=2)
        q = q.view(b, t, attn.n_head, attn.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(b, t, attn.n_head, attn.head_dim).transpose(1, 2)
        v = v.view(b, t, attn.n_head, attn.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(attn.head_dim)  # (B, H, T, T)
        scores = scores.masked_fill(~attn.mask[:t, :t], float("-inf"))
        probs = torch.softmax(scores, dim=-1)  # row t: where position t looks
        if i in layers:
            captured[i] = probs
        y = (probs @ v).transpose(1, 2).contiguous().view(b, t, c)
        x = x + attn.proj(y)
        x = x + block.mlp(block.ln2(x))
    logits: Tensor = model.lm_head(model.ln_f(x))
    return logits, captured
