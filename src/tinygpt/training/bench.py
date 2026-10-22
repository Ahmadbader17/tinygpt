"""Throughput benchmark: tokens per second per thread count, with a time projection."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from tinygpt.config import Config
from tinygpt.data.dataset import MemmapTokenSampler
from tinygpt.model import GPT
from tinygpt.training.trainer import Trainer


@dataclass(frozen=True)
class BenchResult:
    threads: int
    tokens_per_sec: float
    hours_for_budget: float


def bench_threads(
    cfg: Config, thread_counts: list[int], steps: int, token_budget: int
) -> list[BenchResult]:
    """Time real training steps (forward, backward, clip, AdamW) on random tokens.

    Throughput does not depend on token values, so no dataset is needed.
    """
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "random.bin"
        rng = np.random.default_rng(0)
        rng.integers(0, cfg.model.vocab_size, 200_000, dtype=np.uint16).tofile(path)
        data = MemmapTokenSampler(path, cfg.model.context_length)
        for n in thread_counts:
            torch.set_num_threads(n)
            torch.manual_seed(0)
            trainer = Trainer(GPT(cfg.model), replace(cfg), data, data, Path(tmp) / f"run{n}")
            for _ in range(2):  # warm-up: allocator and kernel caches
                trainer.train_step()
            t0 = time.perf_counter()
            for _ in range(steps):
                trainer.train_step()
            tps = steps * trainer.tokens_per_step / (time.perf_counter() - t0)
            results.append(BenchResult(n, tps, token_budget / tps / 3600))
    return results
