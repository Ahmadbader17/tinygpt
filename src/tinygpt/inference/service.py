"""CompletionService: the one entry point the CLI and the API both call."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from tinygpt.config import SampleConfig
from tinygpt.inference.sampler import generate
from tinygpt.model import GPT
from tinygpt.tokenizer import BPETokenizer, Tokenizer


@dataclass(frozen=True)
class CompletionRequest:
    prompt: str
    max_new_tokens: int = 120
    temperature: float = 0.8
    top_k: int | None = 40
    seed: int | None = None


@dataclass(frozen=True)
class CompletionResult:
    text: str  # the completion only, without the prompt
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop" (model ended the story) or "length" (hit max_new_tokens)
    seed: int  # the seed actually used, so any sample can be reproduced
    latency_s: float


class CompletionService:
    def __init__(self, model: GPT, tokenizer: Tokenizer, max_concurrency: int = 1) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.model = model.eval()
        self.tokenizer = tokenizer
        # a CPU forward pass already uses many threads; capping concurrent requests keeps
        # each one fast instead of letting them fight over the same cores
        self._slots = threading.BoundedSemaphore(max_concurrency)

    @classmethod
    def from_run(
        cls,
        run_dir: Path,
        tokenizer_path: Path,
        checkpoint: str = "best",
        max_concurrency: int = 1,
    ) -> CompletionService:
        model = GPT.from_pretrained(run_dir / "checkpoints" / checkpoint)
        tokenizer = BPETokenizer.load(tokenizer_path)
        if tokenizer.vocab_size != model.cfg.vocab_size:
            raise ValueError(
                f"tokenizer vocab {tokenizer.vocab_size} != model vocab {model.cfg.vocab_size}"
            )
        return cls(model, tokenizer, max_concurrency)

    def complete(self, req: CompletionRequest) -> CompletionResult:
        # an empty prompt starts from <|endoftext|>, the same context every training story had
        ids = self.tokenizer.encode(req.prompt) or [self.tokenizer.eot_id]
        seed = req.seed if req.seed is not None else secrets.randbits(31)
        cfg = SampleConfig(req.max_new_tokens, req.temperature, req.top_k, seed)
        with self._slots:
            start = time.perf_counter()
            idx = torch.tensor([ids], dtype=torch.long)
            out = generate(self.model, idx, cfg, self.tokenizer.eot_id)[0].tolist()
            latency = time.perf_counter() - start
        new = out[len(ids) :]
        stopped = bool(new) and new[-1] == self.tokenizer.eot_id
        if stopped:
            new = new[:-1]  # the stop marker is not part of the story
        return CompletionResult(
            text=self.tokenizer.decode(new),
            prompt_tokens=len(ids),
            completion_tokens=len(new),
            finish_reason="stop" if stopped else "length",
            seed=seed,
            latency_s=latency,
        )
