from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tinygpt.config import load_config
from tinygpt.model import GPT
from tinygpt.wsp.scoring import story_nll

CFG = load_config(Path(__file__).parent.parent / "configs" / "tiny.toml").model
EOT = 0


def make_model() -> GPT:
    torch.manual_seed(0)
    return GPT(CFG).eval()


def test_story_loss_equals_the_model_loss_on_eot_story_eot() -> None:
    model = make_model()
    story = [5, 9, 12, 40, 41, 3]
    nll, count = story_nll(model, [story], EOT)
    x = torch.tensor([[EOT, *story]])
    y = torch.tensor([[*story, EOT]])
    _, loss = model(x, y)
    assert loss is not None
    assert count[0] == len(story) + 1  # every story token plus the stop token
    assert abs(nll[0] / count[0] - loss.item()) < 1e-5


def test_padding_and_batching_do_not_change_a_story_loss() -> None:
    model = make_model()
    rng = np.random.default_rng(0)
    stories = [rng.integers(1, CFG.vocab_size, size=n).tolist() for n in (3, 17, 40, 9, 25)]
    together, counts = story_nll(model, stories, EOT, batch_size=3)
    for i, story in enumerate(stories):
        alone, count = story_nll(model, [story], EOT)
        assert count[0] == counts[i]
        assert abs(alone[0] - together[i]) < 1e-3


def test_long_stories_are_scored_in_non_overlapping_windows() -> None:
    model = make_model()
    ctx = CFG.context_length
    story = np.random.default_rng(1).integers(1, CFG.vocab_size, size=2 * ctx + 10).tolist()
    nll, count = story_nll(model, [story], EOT)
    assert count[0] == len(story) + 1
    seq = [EOT, *story, EOT]
    expected = 0.0
    for start in range(0, len(seq) - 1, ctx):
        x = torch.tensor([seq[start : start + ctx][: len(seq) - 1 - start]])
        y = torch.tensor([seq[start + 1 : start + ctx + 1]])
        logits, _ = model(x)
        expected += F.cross_entropy(logits[0], y[0], reduction="sum").item()
    assert abs(nll[0] - expected) < 1e-3


def test_penalised_story_loss_matches_the_penalised_decoder() -> None:
    from tinygpt.wsp.generation import DecodeConfig, next_token

    model = make_model()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(8.0)  # spread the logits so the penalty matters
    story = [5, 9, 5, 12, 9, 9, 40, 5]
    penalty = 1.3
    nll, count = story_nll(model, [story], EOT, repetition_penalty=penalty)
    seq = [EOT, *story, EOT]
    expected = 0.0
    for t in range(len(seq) - 1):
        ctx = seq[: t + 1]
        logits = model(torch.tensor([ctx]))[0][0, -1]
        seen = {tok for tok in ctx if tok != EOT}
        for tok in seen:  # the rule the sampler applies: divide if positive, else multiply
            logits[tok] = logits[tok] / penalty if logits[tok] > 0 else logits[tok] * penalty
        expected -= torch.log_softmax(logits, dim=-1)[seq[t + 1]].item()
    assert count[0] == len(story) + 1
    assert abs(nll[0] - expected) < 1e-3
    plain, _ = story_nll(model, [story], EOT)
    assert abs(plain[0] - nll[0]) > 1e-3  # the penalty really changed the distribution
    # and the same rule picks the same greedy token as the sampler's next_token
    ctx = torch.tensor([[5, 9, 5]])
    raw = model(ctx)[0][:, -1]
    seen_mask = torch.zeros_like(raw, dtype=torch.bool)
    seen_mask[0, [5, 9]] = True
    from tinygpt.wsp.scoring import penalise_seen

    ours = penalise_seen(model(ctx)[0], ctx, EOT, penalty)[:, -1].argmax(dim=-1)
    cfg = DecodeConfig(temperature=0, repetition_penalty=penalty)
    assert ours.item() == next_token(raw, cfg, None, seen_mask).item()


def test_continuation_loss_scores_only_the_new_tokens() -> None:
    from tinygpt.wsp.scoring import continuation_nll

    model = make_model()
    prompts = [[5, 9, 12], [7, 7], [3, 4, 5, 6]]
    conts = [[40, 41, 3], [8, 9, 10, 11, 12], []]
    nll, count = continuation_nll(model, prompts, conts, batch_size=2)
    assert count.tolist() == [3, 5, 0] and nll[2] == 0.0
    for i in (0, 1):
        seq = [*prompts[i], *conts[i]]
        logits, _ = model(torch.tensor([seq[:-1]]))
        start = len(prompts[i]) - 1
        want = F.cross_entropy(logits[0, start:], torch.tensor(conts[i]), reduction="sum").item()
        assert abs(nll[i] - want) < 1e-3
