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
