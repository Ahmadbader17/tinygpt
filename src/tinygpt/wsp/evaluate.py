"""Evaluation of models on given stories and prompts, and paired comparisons between them.

Pure functions on token ids. Nothing here reads sealed data: `final_eval.py` supplies the
sealed stories and prompts, and the harness uses the same functions on dev data.

Success rule for one edit (PLAN.md section 7, Exp 3), fixed before any edit:
  loop rate and rep-4 fall (interval of the paired difference entirely below zero),
  test loss rises by at most 0.02 nats (upper end of the interval),
  mean length is at least 90% of the unedited model's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tinygpt.model import GPT
from tinygpt.wsp.generation import DecodeConfig, Generated, generate_all
from tinygpt.wsp.metrics import (
    PER_GENERATION_METRICS,
    bootstrap_mean,
    bootstrap_ratio,
    bootstrap_ratio_diff,
    per_prompt_stats,
)
from tinygpt.wsp.scoring import story_nll

LOSS_TOLERANCE_NATS = 0.02
LENGTH_FLOOR = 0.90
HEADLINE_ALPHA = 0.05 / 3  # Bonferroni over the three budgets: 98.3% intervals for the headline


@dataclass(frozen=True)
class Decoding:
    name: str
    decode: DecodeConfig
    seeds: tuple[int, ...] | None  # None means one greedy run per prompt


@dataclass
class ModelRun:
    name: str
    story_nll: NDArray[np.float64]
    story_tokens: NDArray[np.int64]
    generations: dict[str, list[list[Generated]]] = field(default_factory=dict)


def run_model(
    model: GPT,
    name: str,
    stories: Sequence[Sequence[int]],
    prompts: Sequence[Sequence[int]],
    eot_id: int,
    decodings: Sequence[Decoding],
) -> ModelRun:
    nll, count = story_nll(model, stories, eot_id)
    run = ModelRun(name, nll, count)
    for d in decodings:
        run.generations[d.name] = generate_all(model, prompts, d.seeds, d.decode, eot_id)
    return run


def _tokens(gens: list[list[Generated]]) -> list[list[list[int]]]:
    return [[g.tokens for g in runs] for runs in gens]


def generation_metrics(gens: list[list[Generated]], alpha: float = 0.05) -> dict[str, Any]:
    """Per-prompt metrics with bootstrap intervals over prompts, plus the stop rate."""
    stats = per_prompt_stats(_tokens(gens))
    out: dict[str, Any] = {
        name: bootstrap_mean(stats[name], alpha=alpha).as_dict() for name in PER_GENERATION_METRICS
    }
    flat = [g for runs in gens for g in runs]
    out["eot_rate"] = float(np.mean([g.hit_eot for g in flat]))
    out["generations"] = len(flat)
    return out


def loss_metrics(run: ModelRun, alpha: float = 0.05) -> dict[str, Any]:
    ci = bootstrap_ratio(run.story_nll, run.story_tokens.astype(np.float64), alpha=alpha)
    return {
        "loss": ci.as_dict(),
        "perplexity": float(np.exp(ci.estimate)),
        "stories": len(run.story_nll),
    }


def paired(a: ModelRun, b: ModelRun, decoding: str, alpha: float = 0.05) -> dict[str, Any]:
    """Model a minus model b on the same prompts, seeds and stories."""
    ga, gb = a.generations[decoding], b.generations[decoding]
    sa, sb = per_prompt_stats(_tokens(ga)), per_prompt_stats(_tokens(gb))
    out: dict[str, Any] = {
        name: bootstrap_mean(sa[name] - sb[name], alpha=alpha).as_dict()
        for name in PER_GENERATION_METRICS
    }
    len_a = float(np.mean([len(g.tokens) for runs in ga for g in runs]))
    len_b = float(np.mean([len(g.tokens) for runs in gb for g in runs]))
    out["length_ratio"] = len_a / len_b
    out["loss_diff"] = bootstrap_ratio_diff(
        a.story_nll, b.story_nll, a.story_tokens.astype(np.float64), alpha=alpha
    ).as_dict()
    return out


def success_rule(diff: dict[str, Any]) -> dict[str, Any]:
    """Each criterion of the preregistered rule, and whether all of them hold."""
    checks = {
        "loop_rate_falls": diff["loop_rate"]["high"] < 0.0,
        "rep4_falls": diff["rep4"]["high"] < 0.0,
        "loss_within_tolerance": diff["loss_diff"]["high"] <= LOSS_TOLERANCE_NATS,
        "length_kept": diff["length_ratio"] >= LENGTH_FLOOR,
    }
    checks["all"] = all(checks.values())
    return checks


def specific(edit_loop_reduction: float, control_reductions: Sequence[float]) -> bool:
    """The edit's loop-rate reduction must beat every matched random control."""
    return bool(control_reductions) and edit_loop_reduction > max(control_reductions)
