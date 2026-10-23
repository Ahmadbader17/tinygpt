"""Autoregressive generation: temperature, top-k, seed, stop at <|endoftext|>."""

from __future__ import annotations

import torch
from torch import Tensor

from tinygpt.config import SampleConfig
from tinygpt.model import GPT


# LEARN: rewrite this yourself, see docs/LEARNING_PLAN.md item 6
@torch.inference_mode()  # no autograd bookkeeping: less memory and faster than no_grad
def generate(model: GPT, idx: Tensor, cfg: SampleConfig, eot_id: int | None) -> Tensor:
    """Extend `idx` (B, T) by up to cfg.max_new_tokens tokens; returns (B, T + new)."""
    was_training = model.training
    model.eval()
    gen = torch.Generator(device=idx.device)
    if cfg.seed is None:
        gen.seed()
    else:
        gen.manual_seed(cfg.seed)
    for _ in range(cfg.max_new_tokens):
        ctx = idx[:, -model.cfg.context_length :]  # the model only sees its last T tokens
        logits, _ = model(ctx)
        logits = logits[:, -1, :]  # (B, T, V) -> (B, V): scores for the next token only
        if cfg.temperature == 0:
            nxt = logits.argmax(dim=-1, keepdim=True)  # (B, 1) greedy
        else:
            logits = logits / cfg.temperature
            if cfg.top_k is not None:
                kth = torch.topk(logits, min(cfg.top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))  # drop the long tail
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=gen)  # (B, 1)
        idx = torch.cat([idx, nxt], dim=1)
        if eot_id is not None and bool((nxt == eot_id).all()):
            break
    model.train(was_training)
    return idx
