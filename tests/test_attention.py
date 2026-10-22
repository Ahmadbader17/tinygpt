from dataclasses import replace

import pytest
import torch

from tinygpt.config import ModelConfig
from tinygpt.model import CausalSelfAttention

CFG = ModelConfig(vocab_size=512, context_length=16, d_model=32, n_layer=2, n_head=4)


def make_pair() -> tuple[CausalSelfAttention, CausalSelfAttention]:
    torch.manual_seed(0)
    naive = CausalSelfAttention(replace(CFG, attention_impl="naive")).eval()
    sdpa = CausalSelfAttention(replace(CFG, attention_impl="sdpa")).eval()
    sdpa.load_state_dict(naive.state_dict())  # same weights, different code path
    return naive, sdpa


def test_naive_and_sdpa_match() -> None:
    naive, sdpa = make_pair()
    x = torch.randn(3, 10, CFG.d_model)
    torch.testing.assert_close(naive(x), sdpa(x), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("impl", ["naive", "sdpa"])
def test_future_tokens_do_not_change_past_outputs(impl: str) -> None:
    torch.manual_seed(0)
    attn = CausalSelfAttention(replace(CFG, attention_impl=impl)).eval()  # type: ignore[arg-type]
    x = torch.randn(2, 10, CFG.d_model)
    x2 = x.clone()
    x2[:, 6:] = torch.randn(2, 4, CFG.d_model)  # rewrite positions 6..9 only
    y, y2 = attn(x), attn(x2)
    torch.testing.assert_close(y[:, :6], y2[:, :6])  # positions 0..5 cannot see 6..9
    assert not torch.allclose(y[:, 6:], y2[:, 6:])  # sanity: the change is visible later


def test_output_shape_matches_input() -> None:
    naive, _ = make_pair()
    assert naive(torch.randn(2, 7, CFG.d_model)).shape == (2, 7, CFG.d_model)
