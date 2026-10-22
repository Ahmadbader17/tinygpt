"""Checkpoints: safetensors weights plus everything needed to resume exactly.

Layout of one checkpoint directory:
    model.safetensors, model_config.json   weights (see GPT.save_pretrained)
    trainer_state.pt                       optimiser state, step, RNG states
    state.json                             step / best_val / tokens_seen, human readable
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_model
from torch import optim

from tinygpt.model import GPT
from tinygpt.model.gpt import WEIGHTS_FILE

STATE_FILE = "trainer_state.pt"


@dataclass
class TrainState:
    step: int = 0
    tokens_seen: int = 0
    best_val: float = float("inf")
    elapsed: float = 0.0  # training seconds across all sessions
    sampler_rng: dict[str, Any] = field(default_factory=dict)  # numpy bit generator state
    torch_rng: torch.Tensor | None = None


def save_checkpoint(
    out_dir: Path, model: GPT, optimizer: optim.Optimizer, state: TrainState
) -> None:
    """Write to a sibling directory first, then swap, so a crash never leaves half a checkpoint."""
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    old = out_dir.with_name(out_dir.name + ".old")
    for d in (tmp, old):
        shutil.rmtree(d, ignore_errors=True)
    model.save_pretrained(tmp)
    torch.save({"optimizer": optimizer.state_dict(), "state": asdict(state)}, tmp / STATE_FILE)
    summary = {k: v for k, v in asdict(state).items() if k not in ("sampler_rng", "torch_rng")}
    (tmp / "state.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if out_dir.exists():
        out_dir.rename(old)
    tmp.rename(out_dir)
    shutil.rmtree(old, ignore_errors=True)


def load_checkpoint(in_dir: Path, model: GPT, optimizer: optim.Optimizer) -> TrainState:
    """Load weights and optimiser state in place; return the trainer state."""
    load_model(model, str(in_dir / WEIGHTS_FILE))
    # our own file, so full unpickling is fine (it holds numpy RNG state, not just tensors)
    blob = torch.load(in_dir / STATE_FILE, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(blob["optimizer"])
    return TrainState(**blob["state"])
