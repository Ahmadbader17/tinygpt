"""Server settings from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path("export")
    max_concurrency: int = 2
    queue_timeout: float = 2.0  # seconds a request waits for a free slot before getting a 429
    num_threads: int | None = None  # None keeps PyTorch's default
    hf_repo: str | None = (
        None  # e.g. "user/tiny-gpt": download the model bundle from the Hub at startup
    )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        threads = e.get("TINYGPT_NUM_THREADS")
        return cls(
            model_dir=Path(e.get("TINYGPT_MODEL_DIR", str(cls.model_dir))),
            max_concurrency=int(e.get("TINYGPT_MAX_CONCURRENCY", cls.max_concurrency)),
            queue_timeout=float(e.get("TINYGPT_QUEUE_TIMEOUT", cls.queue_timeout)),
            num_threads=int(threads) if threads else None,
            hf_repo=e.get("TINYGPT_HF_REPO") or None,
        )
