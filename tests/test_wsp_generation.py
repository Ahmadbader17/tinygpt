from pathlib import Path

import numpy as np
import torch

from tinygpt.config import SampleConfig, load_config
from tinygpt.inference import generate
from tinygpt.model import GPT
from tinygpt.wsp.generation import (
    DecodeConfig,
    _advance,
    _Cache,
    generate_all,
    generate_batch,
    next_token,
    uniforms_for,
)

CFG = load_config(Path(__file__).parent.parent / "configs" / "tiny.toml").model
NO_EOT = -1  # an id the model can never emit, so generation always runs to full length


def make_model() -> GPT:
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    # default init gives nearly flat logits; scale up so argmax and sampling are well separated
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(8.0)
    return model


def test_cached_logits_match_the_full_forward_pass() -> None:
    model = make_model()
    idx = torch.randint(0, CFG.vocab_size, (3, 20), generator=torch.Generator().manual_seed(1))
    full, _ = model(idx)  # (B, T, vocab)
    cache = _Cache(len(model.blocks))
    with torch.inference_mode():
        logits = _advance(model, idx[:, :5], cache)  # prompt of 5 tokens
        # logits here reach about 45 in size, so 1e-4 is float32 rounding, not a real difference
        assert torch.allclose(logits, full[:, 4], atol=1e-4)
        for t in range(5, 20):
            logits = _advance(model, idx[:, t : t + 1], cache)  # then one token at a time
            assert torch.allclose(logits, full[:, t], atol=1e-4)


def test_greedy_matches_the_repo_sampler() -> None:
    model = make_model()
    prompt = torch.tensor([[5, 9, 12, 40]])
    cfg = DecodeConfig(max_new_tokens=30, temperature=0)
    ours = generate_batch(model, prompt, cfg, None, eot_id=NO_EOT)[0]
    ref = generate(model, prompt, SampleConfig(max_new_tokens=30, temperature=0), eot_id=None)
    assert ours.tokens == ref[0, 4:].tolist()
    assert not ours.hit_eot


def test_generation_stops_at_eot_and_drops_it() -> None:
    model = make_model()
    prompt = torch.tensor([[5, 9, 12, 40]])
    cfg = DecodeConfig(max_new_tokens=30, temperature=0)
    free = generate_batch(model, prompt, cfg, None, eot_id=NO_EOT)[0].tokens
    stop = free[3]
    cut = free.index(stop)  # the stop token may already occur before position 3
    gen = generate_batch(model, prompt, cfg, None, eot_id=stop)[0]
    assert gen.hit_eot and gen.tokens == free[:cut]


def test_sampling_does_not_depend_on_batching_or_order() -> None:
    model = make_model()
    prompts = [[5, 9, 12], [7, 7], [1, 2, 3], [30, 31]]  # two lengths, interleaved
    cfg = DecodeConfig(max_new_tokens=12, temperature=0.8, top_k=40)
    a = generate_all(model, prompts, (0, 1, 2), cfg, NO_EOT, batch_size=64)
    b = generate_all(model, prompts, (0, 1, 2), cfg, NO_EOT, batch_size=1)
    assert a == b
    assert len(a) == 4 and all(len(row) == 3 for row in a)
    assert a[0][0] != a[0][1]  # different seeds differ
    # row [p][s] really belongs to prompt p and seed s
    u = torch.from_numpy(uniforms_for(2, 1, 12)).unsqueeze(0)
    assert generate_batch(model, torch.tensor([prompts[2]]), cfg, u, NO_EOT)[0] == a[2][1]


def test_greedy_generate_all_returns_one_run_per_prompt() -> None:
    model = make_model()
    out = generate_all(model, [[5, 9], [7, 7, 7]], None, DecodeConfig(8, temperature=0), NO_EOT)
    assert [len(row) for row in out] == [1, 1]
    assert all(len(row[0].tokens) == 8 for row in out)


def test_inverse_cdf_sampling_follows_temperature_and_top_k() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, -5.0]]).repeat(40_000, 1)
    u = torch.from_numpy(np.random.default_rng(0).random(40_000))
    cfg = DecodeConfig(temperature=0.5, top_k=3)
    seen = torch.zeros_like(logits, dtype=torch.bool)
    picks = next_token(logits, cfg, u, seen)
    freq = torch.bincount(picks, minlength=5).double() / len(picks)
    want = torch.softmax(torch.tensor([2.0, 1.0, 0.0]) / 0.5, dim=0).double()
    assert freq[3] == 0 and freq[4] == 0  # outside the top 3: never sampled
    assert torch.allclose(freq[:3], want, atol=0.01)


def test_repetition_penalty_lowers_only_seen_tokens() -> None:
    logits = torch.tensor([[3.0, 2.9, -1.0, -1.05]])
    seen = torch.tensor([[True, False, True, False]])
    plain = DecodeConfig(temperature=0)
    penal = DecodeConfig(temperature=0, repetition_penalty=1.2)
    assert next_token(logits, plain, None, seen).item() == 0
    # 3.0 / 1.2 = 2.5 < 2.9, so the unseen token now wins
    assert next_token(logits, penal, None, seen).item() == 1
    # negative scores are multiplied: -1.0 * 1.2 = -1.2 < -1.05
    low = torch.tensor([[-1.0, -1.05]])
    assert next_token(low, penal, None, torch.tensor([[True, False]])).item() == 1
