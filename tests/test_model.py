import math
from pathlib import Path

import pytest
import torch

from tinygpt.config import ModelConfig, load_config
from tinygpt.model import GPT

CONFIGS = Path(__file__).parent.parent / "configs"
TINY = load_config(CONFIGS / "tiny.toml").model
SMALL = load_config(CONFIGS / "small.toml").model


def analytic_params(c: ModelConfig) -> int:
    """Written out by hand from the design: every term is one named tensor."""
    d = c.d_model
    embeddings = c.vocab_size * d + c.context_length * d  # wte (tied head adds nothing) + wpe
    attn = (d * 3 * d + 3 * d) + (d * d + d)  # qkv + proj, with biases
    mlp = (d * 4 * d + 4 * d) + (4 * d * d + d)  # fc + proj, with biases
    norms = 2 * (2 * d)  # ln1 + ln2, each weight + bias
    return embeddings + c.n_layer * (attn + mlp + norms) + 2 * d  # + final LayerNorm


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(TINY)


def test_logits_shape(model: GPT) -> None:
    idx = torch.randint(0, TINY.vocab_size, (3, 20))
    logits, loss = model(idx)
    assert logits.shape == (3, 20, TINY.vocab_size)
    assert loss is None


def test_parameter_count_matches_formula() -> None:
    assert GPT(TINY).num_parameters() == analytic_params(TINY)
    assert GPT(SMALL).num_parameters() == analytic_params(SMALL)
    assert 5.7e6 < analytic_params(SMALL) < 5.9e6  # the "5.8M" in DESIGN.md


def test_exclude_embeddings(model: GPT) -> None:
    emb = (TINY.vocab_size + TINY.context_length) * TINY.d_model
    assert model.num_parameters(exclude_embeddings=True) == model.num_parameters() - emb


def test_output_head_is_tied_to_token_embedding(model: GPT) -> None:
    assert model.lm_head.weight is model.wte.weight


def test_initial_loss_is_near_log_vocab(model: GPT) -> None:
    idx = torch.randint(0, TINY.vocab_size, (8, 32))
    _, loss = model(idx, idx.roll(-1, dims=1))
    assert loss is not None
    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.3


def test_future_token_does_not_change_earlier_logits(model: GPT) -> None:
    model.eval()
    idx = torch.randint(0, TINY.vocab_size, (1, 12))
    idx2 = idx.clone()
    idx2[0, 8] = (idx[0, 8] + 1) % TINY.vocab_size
    a, _ = model(idx)
    b, _ = model(idx2)
    torch.testing.assert_close(a[:, :8], b[:, :8])


def test_rejects_sequence_longer_than_context(model: GPT) -> None:
    with pytest.raises(ValueError, match="context_length"):
        model(torch.zeros(1, TINY.context_length + 1, dtype=torch.long))


def test_gradients_reach_every_parameter(model: GPT) -> None:
    idx = torch.randint(0, TINY.vocab_size, (2, 16))
    _, loss = model(idx, idx.roll(-1, dims=1))
    assert loss is not None
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_save_and_load_gives_identical_logits(model: GPT, tmp_path: Path) -> None:
    model.eval()
    model.save_pretrained(tmp_path)
    loaded = GPT.from_pretrained(tmp_path).eval()
    idx = torch.randint(0, TINY.vocab_size, (2, 16))
    torch.testing.assert_close(model(idx)[0], loaded(idx)[0], atol=0, rtol=0)
    assert loaded.lm_head.weight is loaded.wte.weight  # tying survives a round trip
