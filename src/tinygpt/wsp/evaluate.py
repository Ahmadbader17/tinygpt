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
import torch
from numpy.typing import NDArray
from torch import Tensor

from tinygpt.model import GPT
from tinygpt.wsp.generation import DecodeConfig, Generated, generate_all
from tinygpt.wsp.metrics import (
    PER_GENERATION_METRICS,
    bootstrap_mean,
    bootstrap_ratio,
    bootstrap_ratio_diff,
    bootstrap_recovery,
    distinct_n,
    per_prompt_stats,
)
from tinygpt.wsp.scoring import continuation_nll, story_nll

LOSS_TOLERANCE_NATS = 0.02
LENGTH_FLOOR = 0.90
HEADLINE_ALPHA = 0.05 / 3  # Bonferroni over the three budgets: 98.3% intervals for the headline
REPETITION_PENALTY = 1.2  # Keskar et al. 2019; fixed in the plan, not tuned


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
    judge: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = field(default_factory=dict)


def run_model(
    model: GPT,
    name: str,
    stories: Sequence[Sequence[int]],
    prompts: Sequence[Sequence[int]],
    eot_id: int,
    decodings: Sequence[Decoding],
    loss_penalty: float = 1.0,
) -> ModelRun:
    """Score the stories and generate for every decoding.

    `loss_penalty` other than 1 scores the stories with the penalised decoder's distribution;
    it is used for the repetition-penalty baseline, whose decodings carry the same penalty.
    """
    nll, count = story_nll(model, stories, eot_id, repetition_penalty=loss_penalty)
    run = ModelRun(name, nll, count)
    for d in decodings:
        run.generations[d.name] = generate_all(model, prompts, d.seeds, d.decode, eot_id)
    return run


def add_judge(
    run: ModelRun, judge: GPT, prompts: Sequence[Sequence[int]], eot_id: int, decoding: str
) -> None:
    """Score a run's generations with another model (the unedited one); stored per prompt."""
    gens = run.generations[decoding]
    flat_prompts, flat_conts, owner = [], [], []
    for p, runs in enumerate(gens):
        for g in runs:
            flat_prompts.append(prompts[p])
            flat_conts.append([*g.tokens, eot_id] if g.hit_eot else g.tokens)
            owner.append(p)
    nll, count = continuation_nll(judge, flat_prompts, flat_conts)
    per_prompt_nll = np.zeros(len(gens), dtype=np.float64)
    per_prompt_count = np.zeros(len(gens), dtype=np.float64)
    np.add.at(per_prompt_nll, owner, nll)
    np.add.at(per_prompt_count, owner, count.astype(np.float64))
    run.judge[decoding] = (per_prompt_nll, per_prompt_count)


def _tokens(gens: list[list[Generated]]) -> list[list[list[int]]]:
    return [[g.tokens for g in runs] for runs in gens]


def text_metrics(generations: list[list[list[int]]], alpha: float = 0.05) -> dict[str, Any]:
    """Repetition metrics of token lists `generations[prompt][run]`, with intervals over prompts."""
    stats = per_prompt_stats(generations)
    out: dict[str, Any] = {
        name: bootstrap_mean(stats[name], alpha=alpha).as_dict() for name in PER_GENERATION_METRICS
    }
    flat = [tokens for runs in generations for tokens in runs]
    for n in (1, 2, 3):
        out[f"distinct{n}"] = distinct_n(flat, n)
    out["generations"] = len(flat)
    return out


def generation_metrics(run: ModelRun, decoding: str, alpha: float = 0.05) -> dict[str, Any]:
    gens = run.generations[decoding]
    out = text_metrics(_tokens(gens), alpha)
    out["eot_rate"] = float(np.mean([g.hit_eot for runs in gens for g in runs]))
    if decoding in run.judge:
        nll, count = run.judge[decoding]
        keep = count > 0
        out["judge_loss"] = bootstrap_ratio(nll[keep], count[keep], alpha=alpha).as_dict()
    return out


def loss_metrics(run: ModelRun, alpha: float = 0.05) -> dict[str, Any]:
    ci = bootstrap_ratio(run.story_nll, run.story_tokens.astype(np.float64), alpha=alpha)
    return {
        "loss": ci.as_dict(),
        "perplexity": float(np.exp(ci.estimate)),
        "stories": len(run.story_nll),
    }


def loss_diff(a: ModelRun, b: ModelRun, alpha: float = 0.05) -> dict[str, float]:
    """Paired test-loss difference a minus b on the same stories."""
    return bootstrap_ratio_diff(
        a.story_nll, b.story_nll, a.story_tokens.astype(np.float64), alpha=alpha
    ).as_dict()


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
    out["loss_diff"] = loss_diff(a, b, alpha)
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


def recovery(
    damaged: NDArray[np.float64], repaired: NDArray[np.float64], clean: NDArray[np.float64]
) -> dict[str, float]:
    return bootstrap_recovery(damaged, repaired, clean).as_dict()


def repair_cosine(
    damaged: dict[str, Tensor], repaired: dict[str, Tensor], clean: dict[str, Tensor]
) -> float | None:
    """Cosine between the repair and the exact inverse of the fault, over all weights.

    Near 1 means the repair restored the clean weights. For a fault that destroyed
    information that is a warning sign, not a success.
    """
    names = sorted(damaged)
    made = torch.cat([(repaired[n].double() - damaged[n].double()).flatten() for n in names])
    true = torch.cat([(clean[n].double() - damaged[n].double()).flatten() for n in names])
    if float(made.norm()) == 0 or float(true.norm()) == 0:
        return None
    return float(made @ true / (made.norm() * true.norm()))
