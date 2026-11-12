from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tinygpt.config import load_config
from tinygpt.model import GPT
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp.evaluate import (
    LENGTH_FLOOR,
    LOSS_TOLERANCE_NATS,
    Decoding,
    paired,
    run_model,
    specific,
    success_rule,
)
from tinygpt.wsp.generation import DecodeConfig

CFG = load_config(Path(__file__).parent.parent / "configs" / "tiny.toml").model


def _diff(loop_high: float, rep4_high: float, loss_high: float, length: float) -> dict:
    """A paired-difference dict with the fields the rule reads, shaped like `paired()` output."""
    interval = lambda high: {"estimate": high - 0.01, "low": high - 0.02, "high": high}  # noqa: E731
    return {
        "loop_rate": interval(loop_high),
        "rep4": interval(rep4_high),
        "loss_diff": interval(loss_high),
        "length_ratio": length,
    }


def test_rule_passes_only_when_every_criterion_holds() -> None:
    good = success_rule(_diff(loop_high=-0.01, rep4_high=-0.001, loss_high=0.01, length=0.95))
    assert good == {
        "loop_rate_falls": True,
        "rep4_falls": True,
        "loss_within_tolerance": True,
        "length_kept": True,
        "all": True,
    }


@pytest.mark.parametrize(
    ("kwargs", "failing"),
    [
        (
            {"loop_high": 0.01, "rep4_high": -0.001, "loss_high": 0.0, "length": 1.0},
            "loop_rate_falls",
        ),
        ({"loop_high": -0.01, "rep4_high": 0.001, "loss_high": 0.0, "length": 1.0}, "rep4_falls"),
        (
            {
                "loop_high": -0.01,
                "rep4_high": -0.001,
                "loss_high": LOSS_TOLERANCE_NATS + 1e-6,
                "length": 1.0,
            },
            "loss_within_tolerance",
        ),
        (
            {
                "loop_high": -0.01,
                "rep4_high": -0.001,
                "loss_high": 0.0,
                "length": LENGTH_FLOOR - 0.01,
            },
            "length_kept",
        ),
    ],
)
def test_rule_fails_on_each_criterion_separately(kwargs: dict, failing: str) -> None:
    result = success_rule(_diff(**kwargs))
    assert result[failing] is False
    assert result["all"] is False


def test_specificity_needs_to_beat_every_control() -> None:
    assert specific(0.10, [0.02, 0.03, 0.04, 0.05, 0.06])
    assert not specific(0.06, [0.02, 0.03, 0.04, 0.05, 0.06])  # tie with the best control fails
    assert not specific(0.10, [])  # no controls, no claim of specificity


def test_identical_models_cannot_pass_the_rule(
    tmp_path: Path, tiny_tokenizer: BPETokenizer
) -> None:
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    stories = [[5, 9, 12, 40, 41, 3, 7, 8], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]]
    prompts = [[5, 9], [7, 7, 7]]
    decodings = (
        Decoding("sample", DecodeConfig(6, temperature=0.8, top_k=40), (0, 1)),
        Decoding("greedy", DecodeConfig(6, temperature=0), None),
    )
    a = run_model(model, "a", stories, prompts, eot_id=0, decodings=decodings)
    b = run_model(model, "b", stories, prompts, eot_id=0, decodings=decodings)
    diff = paired(a, b, "sample")
    assert diff["length_ratio"] == 1.0
    assert diff["loss_diff"]["estimate"] == 0.0
    assert success_rule(diff)["all"] is False  # zero change cannot show a reduction
