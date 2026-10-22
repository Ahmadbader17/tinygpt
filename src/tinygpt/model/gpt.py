"""The GPT model: embeddings -> blocks -> final norm -> tied output head."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_model, save_model
from torch import Tensor, nn

from tinygpt.config import ModelConfig
from tinygpt.model.layers import Block

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "model_config.json"


# LEARN: rewrite this yourself, see docs/LEARNING_PLAN.md item 4
class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)  # token id -> vector
        self.wpe = nn.Embedding(cfg.context_length, cfg.d_model)  # position -> vector (learned)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # weight tying: one (V, C) matrix reads tokens in and scores tokens out. This saves
        # V*C parameters (a third of the small model) and ties input and output meaning.
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        # residual output projections feed a sum of 2*n_layer branches; shrink them so the
        # residual stream's variance does not grow with depth
        for name, p in self.named_parameters():
            if name.endswith(("attn.proj.weight", "mlp.proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        b, t = idx.shape  # (B, T) int64 token ids
        if t > self.cfg.context_length:
            raise ValueError(
                f"sequence length {t} exceeds context_length {self.cfg.context_length}"
            )
        pos = torch.arange(t, device=idx.device)  # (T,)
        x = self.drop(self.wte(idx) + self.wpe(pos))  # (B, T, C) + (T, C) broadcasts over B
        for block in self.blocks:
            x = block(x)  # (B, T, C)
        logits = self.lm_head(self.ln_f(x))  # (B, T, C) -> (B, T, vocab)
        loss = None
        if targets is not None:
            # cross_entropy wants (N, vocab) and (N,): fold batch and time together
            loss = F.cross_entropy(logits.view(b * t, -1), targets.reshape(b * t))
        return logits, loss

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Parameter count; the tied head is counted once (parameters() de-duplicates)."""
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.wte.weight.numel() + self.wpe.weight.numel()
        return n

    def save_pretrained(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # save_model stores the tied tensor once, plain save_file would refuse shared memory
        save_model(self, str(out_dir / WEIGHTS_FILE))
        (out_dir / CONFIG_FILE).write_text(json.dumps(asdict(self.cfg), indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, model_dir: Path) -> GPT:
        cfg = ModelConfig(**json.loads((model_dir / CONFIG_FILE).read_text(encoding="utf-8")))
        model = cls(cfg)
        load_model(model, str(model_dir / WEIGHTS_FILE))
        return model
