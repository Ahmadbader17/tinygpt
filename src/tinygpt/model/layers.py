"""MLP and the pre-norm transformer block."""

from __future__ import annotations

from torch import Tensor, nn

from tinygpt.config import ModelConfig
from tinygpt.model.attention import CausalSelfAttention


# LEARN: rewrite this yourself, see docs/LEARNING_PLAN.md item 4
class MLP(nn.Module):
    """Per-token feed forward network: C -> 4C -> C. Mixes channels, never positions."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.act = nn.GELU()
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.fc(x))  # (B, T, C) -> (B, T, 4C)
        out: Tensor = self.drop(self.proj(h))  # (B, T, 4C) -> (B, T, C)
        return out


class Block(nn.Module):
    """x = x + attn(ln1(x)); x = x + mlp(ln2(x)).

    Pre-norm keeps the residual stream un-normalised, so gradients reach early layers
    through the identity path; post-norm is harder to train without careful warmup.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))  # (B, T, C)
        out: Tensor = x + self.mlp(self.ln2(x))  # (B, T, C)
        return out
