import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from tinygpt.config import Config, TrainConfig, dump_config, load_config
from tinygpt.data.dataset import MemmapTokenSampler
from tinygpt.model import GPT
from tinygpt.training import Trainer, estimate_loss

BASE = load_config(Path(__file__).parent.parent / "configs" / "tiny.toml")


def make_cfg(**train: float) -> Config:
    defaults = {"batch_size": 4, "eval_interval": 5, "eval_batches": 2, "log_interval": 1}
    tc = dataclasses.replace(BASE.train, **{**defaults, **train})
    return dataclasses.replace(BASE, train=tc)


def make_trainer(cfg: Config, tokens: Path, run_dir: Path, seed: int = 0) -> Trainer:
    torch.manual_seed(seed)
    data = MemmapTokenSampler(tokens, cfg.model.context_length)
    return Trainer(GPT(cfg.model), cfg, data, data, run_dir)


def test_overfits_one_fixed_batch(tmp_path: Path) -> None:
    """A file of exactly context_length + 1 tokens has one possible window: a fixed batch."""
    t = BASE.model.context_length
    path = tmp_path / "one.bin"
    np.random.default_rng(1).integers(0, 512, t + 1, dtype=np.uint16).tofile(path)
    cfg = make_cfg(lr_peak=3e-3, lr_min=3e-3, warmup_steps=10, max_steps=300)
    trainer = make_trainer(cfg, path, tmp_path / "run")
    first = trainer.train_step()
    last = first
    for _ in range(299):
        last = trainer.train_step()
    assert first > 5.5  # starts near ln(512) = 6.24
    assert last < 0.1  # gradients reach every parameter and the optimiser works


def test_resume_continues_with_identical_losses(token_file: Path, tmp_path: Path) -> None:
    cfg = make_cfg(max_steps=10, warmup_steps=2)
    a = make_trainer(cfg, token_file, tmp_path / "a")
    for _ in range(5):
        a.train_step()
    a.save("last")
    expected = [a.train_step() for _ in range(5)]

    # a fresh process: different init, fresh optimiser, then resume from the checkpoint
    b = make_trainer(cfg, token_file, tmp_path / "a", seed=99)
    assert b.resume()
    assert b.state.step == 5
    got = [b.train_step() for _ in range(5)]
    assert got == pytest.approx(expected, abs=0, rel=0)


def test_resume_without_checkpoint_returns_false(token_file: Path, tmp_path: Path) -> None:
    assert not make_trainer(make_cfg(), token_file, tmp_path / "x").resume()


def test_fit_writes_run_outputs_and_best_checkpoint(token_file: Path, tmp_path: Path) -> None:
    cfg = make_cfg(max_steps=10, warmup_steps=2)
    run = tmp_path / "run"
    trainer = make_trainer(cfg, token_file, run)
    trainer.sample_fn = lambda model: "sample text"
    trainer.fit()
    for name in ("config.toml", "environment.json", "git_commit.txt", "metrics.csv", "loss.png"):
        assert (run / name).exists(), name
    assert (run / "checkpoints" / "last" / "model.safetensors").exists()
    assert (run / "checkpoints" / "best" / "model.safetensors").exists()
    assert (run / "samples.md").read_text().count("## step") == 2  # steps 5 and 10
    assert load_config(run / "config.toml") == cfg  # the saved config reproduces the run
    assert trainer.state.step == 10


def test_fit_stops_on_time_limit_and_can_resume(token_file: Path, tmp_path: Path) -> None:
    cfg = make_cfg(max_steps=1000, warmup_steps=2)
    run = tmp_path / "run"
    first = make_trainer(cfg, token_file, run)
    first.fit(max_hours=1e-9)  # stops after the first step
    assert 1 <= first.state.step < 1000
    second = make_trainer(cfg, token_file, run)
    assert second.resume()
    assert second.state.step == first.state.step


def test_estimate_loss_is_deterministic_and_restores_mode(token_file: Path) -> None:
    torch.manual_seed(0)
    model = GPT(BASE.model)
    data = MemmapTokenSampler(token_file, BASE.model.context_length)
    model.train()
    a = estimate_loss(model, data, 3, 4, torch.device("cpu"))
    b = estimate_loss(model, data, 3, 4, torch.device("cpu"))
    assert a == b
    assert model.training


def test_loss_decreases_on_learnable_data(tmp_path: Path) -> None:
    """Repeating pattern: a few steps of training must beat the untrained loss clearly."""
    path = tmp_path / "pattern.bin"
    (np.arange(5000) % 17).astype(np.uint16).tofile(path)
    cfg = make_cfg(lr_peak=3e-3, lr_min=3e-3, warmup_steps=5, max_steps=60)
    trainer = make_trainer(cfg, path, tmp_path / "run")
    losses = [trainer.train_step() for _ in range(60)]
    assert losses[-1] < losses[0] - 3


def test_dump_config_roundtrip(tmp_path: Path) -> None:
    cfg: Config = load_config(Path(__file__).parent.parent / "configs" / "small.toml")
    assert isinstance(cfg.train, TrainConfig)
    (tmp_path / "c.toml").write_text(dump_config(cfg))
    assert load_config(tmp_path / "c.toml") == cfg
