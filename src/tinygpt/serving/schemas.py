"""Request and response models; the validation limits live here."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from tinygpt.config import SampleConfig


class CompletionIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    max_new_tokens: int = Field(default=SampleConfig.max_new_tokens, ge=1, le=256)
    temperature: float = Field(default=SampleConfig.temperature, ge=0.0, le=2.0)
    # the upper bound (vocab size) depends on the loaded model, so the route checks it
    top_k: int | None = Field(default=SampleConfig.top_k, ge=1)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


class CompletionOut(BaseModel):
    prompt: str
    completion: str
    tokens_generated: int
    stop_reason: Literal["eot", "length"]
    seed: int  # pass it back as `seed` to reproduce this exact completion
    latency_ms: int


class ModelInfo(BaseModel):
    name: str
    version: str
    parameters: int
    config: dict[str, Any]
    training: dict[str, Any]  # tokens, steps, validation loss and perplexity (when exported)
