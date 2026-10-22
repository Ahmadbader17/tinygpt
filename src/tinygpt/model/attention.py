"""Causal multi-head self attention with a readable path and a fast path."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinygpt.config import ModelConfig


# LEARN: rewrite this yourself, see docs/LEARNING_PLAN.md item 3
class CausalSelfAttention(nn.Module):
    """Multi-head attention where position t may only look at positions <= t.

    "naive" spells out softmax(Q K^T / sqrt(d)) V; "sdpa" calls the fused PyTorch kernel.
    Both share weights and must agree, so the naive path doubles as the reference.
    """

    mask: Tensor  # registered buffer, typed so mypy does not see Tensor | Module

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.impl = cfg.attention_impl
        self.dropout = cfg.dropout
        # one matmul produces Q, K and V for all heads: (B, T, C) -> (B, T, 3C)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # lower triangle of ones: row t may see columns 0..t. Not saved in checkpoints.
        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x.shape  # (B, T, C)
        q, k, v = self.qkv(x).split(c, dim=2)  # 3 x (B, T, C)
        # (B, T, C) -> (B, n_head, T, head_dim): each head attends independently
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        if self.impl == "sdpa":
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )  # (B, n_head, T, head_dim)
        else:
            # scale by sqrt(head_dim) so score variance stays ~1 and softmax is not saturated
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, n_head, T, T)
            scores = scores.masked_fill(~self.mask[:t, :t], float("-inf"))  # hide the future
            att = self.attn_drop(F.softmax(scores, dim=-1))  # rows sum to 1
            y = att @ v  # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)

        # (B, n_head, T, head_dim) -> (B, T, n_head, head_dim) -> (B, T, C): concatenate heads
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        out: Tensor = self.resid_drop(self.proj(y))  # (B, T, C)
        return out
