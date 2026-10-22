"""Frozen dataclass configs loaded from TOML.

Frozen so a run cannot mutate its own hyperparameters halfway through; the resolved
config is written next to every run and is the single source of truth for it.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    n_layer: int
    n_head: int
    dropout: float = 0.0
    attention_impl: Literal["naive", "sdpa"] = "sdpa"

    def __post_init__(self) -> None:
        # Heads split d_model evenly: (B, T, C) -> (B, n_head, T, C // n_head).
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model={self.d_model} not divisible by n_head={self.n_head}")
        if self.attention_impl not in ("naive", "sdpa"):
            raise ValueError(f"unknown attention_impl: {self.attention_impl!r}")


@dataclass(frozen=True)
class DataConfig:
    source: Literal["tinystories", "gutenberg"] = "tinystories"
    train_stories: int = 300_000
    val_stories: int = 5_000
    tokenizer_vocab_size: int = 4096


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_steps: int = 4500
    warmup_steps: int = 200
    lr_peak: float = 1e-3
    lr_min: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 20
    num_threads: int = 4


@dataclass(frozen=True)
class SampleConfig:
    max_new_tokens: int = 120
    temperature: float = 0.8
    top_k: int | None = 40
    seed: int | None = None


@dataclass(frozen=True)
class Config:
    name: str
    model: ModelConfig
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 1337


def _build[T](cls: type[T], raw: dict[str, Any], section: str) -> T:
    """Construct a dataclass from a TOML table, rejecting unknown keys (typos)."""
    known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown keys in [{section}]: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: Path | str) -> Config:
    """Load a `Config` from a TOML file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Config(
        name=raw["name"],
        seed=raw.get("seed", 1337),
        model=_build(ModelConfig, raw["model"], "model"),
        data=_build(DataConfig, raw.get("data", {}), "data"),
        train=_build(TrainConfig, raw.get("train", {}), "train"),
    )
