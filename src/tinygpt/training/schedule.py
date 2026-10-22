"""Learning rate schedule: linear warmup, then cosine decay to a floor."""

from __future__ import annotations

import math

from tinygpt.config import TrainConfig


# LEARN: rewrite this yourself, see docs/LEARNING_PLAN.md item 5
def lr_at(step: int, cfg: TrainConfig) -> float:
    """Learning rate to use for 0-indexed optimiser step `step`.

    Warmup avoids huge early updates while Adam's moment estimates are still noisy;
    cosine decay gives big steps early and fine adjustments at the end.
    """
    if step < cfg.warmup_steps:
        return (
            cfg.lr_peak * (step + 1) / cfg.warmup_steps
        )  # reaches lr_peak on the last warmup step
    if step >= cfg.max_steps:
        return cfg.lr_min
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)  # 0 -> 1
    return cfg.lr_min + 0.5 * (cfg.lr_peak - cfg.lr_min) * (1 + math.cos(math.pi * progress))
