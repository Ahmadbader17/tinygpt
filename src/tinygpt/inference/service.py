"""CompletionService: the one entry point the CLI and the API both call."""

from __future__ import annotations

import json
import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from tinygpt.config import SampleConfig
from tinygpt.inference.sampler import generate
from tinygpt.model import GPT
from tinygpt.model.gpt import CONFIG_FILE, WEIGHTS_FILE
from tinygpt.tokenizer import BPETokenizer, Tokenizer

TOKENIZER_FILE = "tokenizer.json"
INFO_FILE = "model_info.json"


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


class ServiceBusy(RuntimeError):
    """All generation slots stayed taken for longer than the queue timeout."""


class CompletionService:
    def __init__(
        self,
        model: GPT,
        tokenizer: Tokenizer,
        max_concurrency: int = 1,
        queue_timeout: float | None = None,
    ) -> None:
        """`queue_timeout` seconds to wait for a free slot; None waits as long as it takes."""
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.queue_timeout = queue_timeout
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

    @classmethod
    def from_bundle(
        cls, model_dir: Path, max_concurrency: int = 1, queue_timeout: float | None = None
    ) -> CompletionService:
        """Load a self-contained model directory (weights, config, tokenizer.json)."""
        model = GPT.from_pretrained(model_dir)
        tokenizer = BPETokenizer.load(model_dir / TOKENIZER_FILE)
        return cls(model, tokenizer, max_concurrency, queue_timeout)

    def complete(self, req: CompletionRequest) -> CompletionResult:
        # an empty prompt starts from <|endoftext|>, the same context every training story had
        ids = self.tokenizer.encode(req.prompt) or [self.tokenizer.eot_id]
        seed = req.seed if req.seed is not None else secrets.randbits(31)
        cfg = SampleConfig(req.max_new_tokens, req.temperature, req.top_k, seed)
        if not self._slots.acquire(timeout=self.queue_timeout):
            raise ServiceBusy("all generation slots are busy")
        try:
            start = time.perf_counter()
            idx = torch.tensor([ids], dtype=torch.long)
            out = generate(self.model, idx, cfg, self.tokenizer.eot_id)[0].tolist()
            latency = time.perf_counter() - start
        finally:
            self._slots.release()
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


def export_bundle(
    run_dir: Path,
    tokenizer_path: Path,
    out_dir: Path,
    info: dict[str, object],
    checkpoint: str = "best",
) -> None:
    """Copy weights, config and tokenizer into one directory a server can load by itself."""
    src = run_dir / "checkpoints" / checkpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in (WEIGHTS_FILE, CONFIG_FILE):
        shutil.copy2(src / name, out_dir / name)
    shutil.copy2(tokenizer_path, out_dir / TOKENIZER_FILE)
    (out_dir / INFO_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
