from pathlib import Path

import torch

from tinygpt.config import load_config
from tinygpt.model import GPT
from tinygpt.training.checkpoint import TrainState, load_checkpoint, save_checkpoint
from tinygpt.training.trainer import build_optimizer

CFG = load_config(Path(__file__).parent.parent / "configs" / "tiny.toml")


def test_roundtrip_restores_weights_optimiser_and_state(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = GPT(CFG.model)
    opt = build_optimizer(model, CFG.train)
    idx = torch.randint(0, CFG.model.vocab_size, (2, 16))
    _, loss = model(idx, idx.roll(-1, 1))
    assert loss is not None
    loss.backward()
    opt.step()  # populate Adam moment estimates
    state = TrainState(step=7, tokens_seen=123, best_val=1.5, torch_rng=torch.get_rng_state())
    save_checkpoint(tmp_path / "last", model, opt, state)

    model2 = GPT(CFG.model)
    opt2 = build_optimizer(model2, CFG.train)
    loaded = load_checkpoint(tmp_path / "last", model2, opt2)

    assert (loaded.step, loaded.tokens_seen, loaded.best_val) == (7, 123, 1.5)
    model.eval()
    model2.eval()
    torch.testing.assert_close(model(idx)[0], model2(idx)[0], atol=0, rtol=0)
    p = next(iter(model.parameters()))
    assert torch.equal(
        opt.state[p]["exp_avg"], opt2.state[next(iter(model2.parameters()))]["exp_avg"]
    )


def test_saving_twice_replaces_cleanly(tmp_path: Path) -> None:
    model = GPT(CFG.model)
    opt = build_optimizer(model, CFG.train)
    for step in (1, 2):
        save_checkpoint(tmp_path / "last", model, opt, TrainState(step=step))
    assert load_checkpoint(tmp_path / "last", model, opt).step == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["last"]  # no .tmp / .old left behind
