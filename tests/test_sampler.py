from dataclasses import replace
from pathlib import Path

import torch

from tinygpt.config import SampleConfig, load_config
from tinygpt.inference import generate
from tinygpt.model import GPT

CFG = load_config(Path(__file__).parent.parent / "configs" / "tiny.toml").model
PROMPT = torch.tensor([[5, 9, 12]])


def make_model() -> GPT:
    torch.manual_seed(0)
    return GPT(CFG).eval()


def test_temperature_zero_is_argmax() -> None:
    model = make_model()
    out = generate(model, PROMPT, SampleConfig(max_new_tokens=1, temperature=0), eot_id=None)
    expected = model(PROMPT)[0][0, -1].argmax()
    assert out[0, -1] == expected
    assert out.shape == (1, 4)


def test_same_seed_same_tokens_different_seed_differs() -> None:
    model = make_model()
    cfg = SampleConfig(max_new_tokens=20, temperature=1.0, top_k=None, seed=3)
    a = generate(model, PROMPT, cfg, None)
    b = generate(model, PROMPT, cfg, None)
    c = generate(model, PROMPT, replace(cfg, seed=4), None)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_stops_at_eot() -> None:
    model = make_model()
    greedy = SampleConfig(max_new_tokens=30, temperature=0)
    first = int(generate(model, PROMPT, replace(greedy, max_new_tokens=1), None)[0, -1])
    out = generate(model, PROMPT, greedy, eot_id=first)  # the first greedy token is the stop token
    assert out.shape[1] == PROMPT.shape[1] + 1


def test_top_k_one_equals_greedy() -> None:
    model = make_model()
    greedy = generate(model, PROMPT, SampleConfig(max_new_tokens=8, temperature=0), None)
    top1 = generate(
        model, PROMPT, SampleConfig(max_new_tokens=8, temperature=1.0, top_k=1, seed=0), None
    )
    assert torch.equal(greedy, top1)


def test_context_longer_than_window_is_cropped() -> None:
    model = make_model()
    long_prompt = torch.randint(0, CFG.vocab_size, (1, CFG.context_length + 10))
    out = generate(model, long_prompt, SampleConfig(max_new_tokens=2, temperature=0), None)
    assert out.shape[1] == long_prompt.shape[1] + 2
