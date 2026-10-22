"""Trainer: one optimiser step, evaluation, the fit loop, checkpoint and resume."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType

import numpy as np
import torch
from torch import optim

from tinygpt.config import Config, TrainConfig, dump_config
from tinygpt.data.dataset import MemmapTokenSampler
from tinygpt.model import GPT
from tinygpt.training.checkpoint import TrainState, load_checkpoint, save_checkpoint
from tinygpt.training.metrics import MetricsLogger, plot_loss
from tinygpt.training.schedule import lr_at
from tinygpt.utils.env import probe_environment

log = logging.getLogger(__name__)

SampleFn = Callable[[GPT], str]
VAL_SEED = 12345  # every evaluation sees the same validation windows, so curves are comparable


def build_optimizer(model: GPT, cfg: TrainConfig) -> optim.AdamW:
    """AdamW with weight decay on matrices only; biases and LayerNorm gains are not decayed."""
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr_peak,
        betas=(cfg.beta1, cfg.beta2),
    )


@torch.no_grad()
def estimate_loss(
    model: GPT, data: MemmapTokenSampler, batches: int, batch_size: int, device: torch.device
) -> float:
    """Mean loss over `batches` fixed random validation batches."""
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(VAL_SEED)
    total = 0.0
    for _ in range(batches):
        x, y = data.batch(batch_size, rng)
        _, loss = model(x.to(device), y.to(device))
        assert loss is not None
        total += loss.item()
    model.train(was_training)
    return total / batches


def write_run_metadata(run_dir: Path, cfg: Config) -> None:
    """config.toml, environment.json and git_commit.txt, written once per run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "config.toml").exists():
        return
    (run_dir / "config.toml").write_text(dump_config(cfg), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(probe_environment(), indent=2))
    commit = "unknown"
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    (run_dir / "git_commit.txt").write_text(commit + "\n")


# LEARN: rewrite train_step yourself, see docs/LEARNING_PLAN.md item 5
class Trainer:
    """Everything is injected, so tests can pass the tiny model and a fixture token file."""

    def __init__(
        self,
        model: GPT,
        cfg: Config,
        train_data: MemmapTokenSampler,
        val_data: MemmapTokenSampler,
        run_dir: Path,
        device: torch.device | None = None,
        sample_fn: SampleFn | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.optimizer = build_optimizer(self.model, cfg.train)
        self.train_data, self.val_data = train_data, val_data
        self.run_dir = run_dir
        self.sample_fn = sample_fn
        self.rng = np.random.default_rng(cfg.seed)  # picks training windows; saved for resume
        self.state = TrainState()
        self.metrics = MetricsLogger(run_dir / "metrics.csv")
        self._stop = False

    @property
    def tokens_per_step(self) -> int:
        t = self.cfg.train
        return t.batch_size * t.grad_accum_steps * self.cfg.model.context_length

    def train_step(self) -> float:
        """One optimiser update. Returns the mean loss over the accumulated micro batches."""
        tc = self.cfg.train
        self.model.train()
        lr = lr_at(self.state.step, tc)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(tc.grad_accum_steps):
            x, y = self.train_data.batch(tc.batch_size, self.rng)  # (B, T) each
            _, loss = self.model(x.to(self.device), y.to(self.device))
            assert loss is not None
            # gradients add up across micro batches, so divide to get the mean
            (loss / tc.grad_accum_steps).backward()
            total += loss.item() / tc.grad_accum_steps
        # clip the global gradient norm: one bad batch cannot throw the weights far away
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
        self.optimizer.step()
        self.state.step += 1
        self.state.tokens_seen += self.tokens_per_step
        return total

    def evaluate(self) -> float:
        tc = self.cfg.train
        return estimate_loss(self.model, self.val_data, tc.eval_batches, tc.batch_size, self.device)

    def save(self, name: str) -> None:
        self.state.sampler_rng = dict(self.rng.bit_generator.state)
        self.state.torch_rng = torch.get_rng_state()
        save_checkpoint(self.run_dir / "checkpoints" / name, self.model, self.optimizer, self.state)

    def resume(self) -> bool:
        """Continue from checkpoints/last if it exists. Returns whether it did."""
        ckpt = self.run_dir / "checkpoints" / "last"
        if not ckpt.exists():
            return False
        self.state = load_checkpoint(ckpt, self.model, self.optimizer)
        self.rng.bit_generator.state = self.state.sampler_rng
        assert self.state.torch_rng is not None
        torch.set_rng_state(self.state.torch_rng)
        log.info("resumed from %s at step %d", ckpt, self.state.step)
        return True

    def _request_stop(self, signum: int, frame: FrameType | None) -> None:
        log.warning("signal %d received: finishing the step, saving, exiting", signum)
        self._stop = True

    def fit(self, max_hours: float | None = None) -> None:
        """Train to cfg.train.max_steps (or the time limit), checkpointing as we go."""
        tc = self.cfg.train
        write_run_metadata(self.run_dir, self.cfg)
        old_handlers = {
            s: signal.signal(s, self._request_stop) for s in (signal.SIGINT, signal.SIGTERM)
        }
        session_start = time.perf_counter()
        base_elapsed = self.state.elapsed
        window_start, window_loss, window_steps = session_start, 0.0, 0
        try:
            while self.state.step < tc.max_steps and not self._stop:
                window_loss += self.train_step()
                window_steps += 1
                st = self.state
                now = time.perf_counter()
                st.elapsed = base_elapsed + now - session_start
                at_end = st.step == tc.max_steps
                do_eval = st.step % tc.eval_interval == 0 or at_end
                if st.step % tc.log_interval == 0 or do_eval:
                    train_loss = window_loss / window_steps
                    tps = window_steps * self.tokens_per_step / max(now - window_start, 1e-9)
                    val = self.evaluate() if do_eval else None
                    lr = lr_at(st.step - 1, tc)
                    self.metrics.log(
                        step=st.step,
                        train_loss=train_loss,
                        val_loss=val,
                        lr=lr,
                        tokens_seen=st.tokens_seen,
                        tokens_per_sec=round(tps),
                        elapsed=round(st.elapsed),
                    )
                    eta_h = (tc.max_steps - st.step) * self.tokens_per_step / max(tps, 1) / 3600
                    log.info(
                        "step %d/%d loss %.4f%s lr %.2e %.0f tok/s eta %.2fh",
                        st.step,
                        tc.max_steps,
                        train_loss,
                        f" val {val:.4f}" if val is not None else "",
                        lr,
                        tps,
                        eta_h,
                    )
                    window_start, window_loss, window_steps = time.perf_counter(), 0.0, 0
                    if val is not None:
                        self._after_eval(val)
                if max_hours is not None and st.elapsed - base_elapsed > max_hours * 3600:
                    log.warning("time limit of %.2fh reached at step %d", max_hours, st.step)
                    break
        finally:
            for s, h in old_handlers.items():
                signal.signal(s, h)
            self.save("last")  # always leave a resumable checkpoint, even on Ctrl-C or error
            plot_loss(self.metrics.path, self.run_dir / "loss.png")
            log.info("stopped at step %d; best val %.4f", self.state.step, self.state.best_val)

    def _after_eval(self, val: float) -> None:
        if not math.isfinite(val):
            raise RuntimeError("validation loss is not finite; training diverged")
        if val < self.state.best_val:
            self.state.best_val = val
            self.save("best")
        self.save("last")
        if self.sample_fn is not None:
            with open(self.run_dir / "samples.md", "a", encoding="utf-8") as f:
                f.write(
                    f"## step {self.state.step} (val {val:.4f})\n\n{self.sample_fn(self.model)}\n\n"
                )
        plot_loss(self.metrics.path, self.run_dir / "loss.png")
