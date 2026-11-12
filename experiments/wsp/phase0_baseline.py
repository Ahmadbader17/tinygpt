"""Phase 0: baseline numbers for PLAN.md, measured on dev data only.

Reproduces the README validation loss with the repo's own recipe, then measures the unedited
model on the dev protocol of this study. Only aggregate numbers are printed and saved; no
generated text leaves this script, so reading its output costs the editor nothing that a
forward-pass budget is meant to ration. Run from the repo root:

    uv run python experiments/wsp/phase0_baseline.py
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from tinygpt.data.dataset import MemmapTokenSampler
from tinygpt.model import GPT
from tinygpt.tokenizer import BPETokenizer
from tinygpt.training.trainer import estimate_loss
from tinygpt.wsp import protocol
from tinygpt.wsp.generation import DecodeConfig, Generated, generate_all
from tinygpt.wsp.metrics import (
    PER_GENERATION_METRICS,
    bootstrap_mean,
    bootstrap_ratio,
    distinct_n,
    per_prompt_stats,
)
from tinygpt.wsp.scoring import story_nll

ORIGINAL = Path("runs/wsp/original")  # the Hub download, see PLAN.md section 2
OUT = Path("experiments/wsp/results/phase0_baseline.json")
THREADS = 12  # docs/DECISIONS.md 0006
REPETITION_PENALTY = 1.2  # the value recommended by Keskar et al. 2019; fixed, not tuned


def tokens_of(gens: list[list[Generated]]) -> list[list[list[int]]]:
    return [[g.tokens for g in runs] for runs in gens]


def summarise(gens: list[list[Generated]]) -> dict[str, object]:
    stats = per_prompt_stats(tokens_of(gens))
    flat = [g for runs in gens for g in runs]
    out: dict[str, object] = {
        name: bootstrap_mean(stats[name]).as_dict() for name in PER_GENERATION_METRICS
    }
    out["eot_rate"] = float(np.mean([g.hit_eot for g in flat]))
    for n in (1, 2, 3):
        out[f"distinct{n}"] = distinct_n([g.tokens for g in flat], n)
    out["generations"] = len(flat)
    return out


def real_text_reference(
    tok: BPETokenizer, stories: list[str], prompts: list[str]
) -> dict[str, object]:
    """The same metrics on the real continuations of the dev prompts. No model is run.

    Real stories repeat phrases too, so this is the level an edit should move towards,
    not zero.
    """
    by_prompt = {}
    for i in protocol.DEV_PROMPT_STORIES:
        prompt = protocol.prompt_from_story(stories[i])
        if prompt is not None and prompt not in by_prompt:
            by_prompt[prompt] = stories[i]
    real = []
    for prompt in prompts:
        ids = tok.encode(by_prompt[prompt])
        real.append(ids[len(tok.encode(prompt)) :][: protocol.MAX_NEW_TOKENS])
    stats = per_prompt_stats([[tokens] for tokens in real])
    out: dict[str, object] = {
        name: bootstrap_mean(stats[name]).as_dict() for name in PER_GENERATION_METRICS
    }
    for n in (1, 2, 3):
        out[f"distinct{n}"] = distinct_n(real, n)
    out["stories"] = len(real)
    return out


def paired(a: list[list[Generated]], b: list[list[Generated]]) -> dict[str, object]:
    """Paired difference a - b of every per-generation metric, clustered by prompt."""
    sa, sb = per_prompt_stats(tokens_of(a)), per_prompt_stats(tokens_of(b))
    return {name: bootstrap_mean(sa[name] - sb[name]).as_dict() for name in PER_GENERATION_METRICS}


def main() -> None:
    torch.set_num_threads(THREADS)
    model = GPT.from_pretrained(ORIGINAL).eval()
    tok = BPETokenizer.load(ORIGINAL / "tokenizer.json")
    result: dict[str, object] = {
        "environment": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "threads": THREADS,
            "commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip(),
        }
    }
    seconds: dict[str, float] = {}

    start = time.perf_counter()
    val = MemmapTokenSampler(Path("data/processed/val.bin"), model.cfg.context_length)
    readme_loss = estimate_loss(model, val, 200, 32, torch.device("cpu"))
    seconds["readme_recipe_200_batches"] = time.perf_counter() - start
    result["readme_recipe"] = {"val_loss": readme_loss, "perplexity": math.exp(readme_loss)}
    print(f"README recipe: val loss {readme_loss:.4f}", flush=True)

    stories = protocol.load_dev_stories()
    loss_ids = tok.encode_batch(protocol.dev_loss_stories(stories))
    start = time.perf_counter()
    nll, count = story_nll(model, loss_ids, tok.eot_id)
    seconds["dev_story_loss"] = time.perf_counter() - start
    loss = bootstrap_ratio(nll, count.astype(np.float64))
    result["dev_story_loss"] = {
        "stories": len(loss_ids),
        "tokens": int(count.sum()),
        "loss": loss.as_dict(),
        "perplexity": math.exp(loss.estimate),
    }
    print(f"dev story loss: {loss.estimate:.4f} [{loss.low:.4f}, {loss.high:.4f}]", flush=True)

    prompts = protocol.dev_prompts(stories)
    prompt_ids = [tok.encode(p) for p in prompts]
    lengths = [len(p) for p in prompt_ids]
    result["dev_prompts"] = {
        "count": len(prompts),
        "tokens_min": min(lengths),
        "tokens_max": max(lengths),
        "candidate_stories": len(protocol.DEV_PROMPT_STORIES),
    }
    result["dev_real_text_reference"] = real_text_reference(tok, stories, prompts)
    sample = DecodeConfig(protocol.MAX_NEW_TOKENS, temperature=0.8, top_k=40)
    greedy = DecodeConfig(protocol.MAX_NEW_TOKENS, temperature=0)
    penalised = DecodeConfig(
        protocol.MAX_NEW_TOKENS, temperature=0.8, top_k=40, repetition_penalty=REPETITION_PENALTY
    )
    seeds = protocol.SAMPLING_SEEDS
    other_seeds = tuple(s + len(seeds) for s in seeds)  # a disjoint seed set for the A/A check

    start = time.perf_counter()
    g_sample = generate_all(model, prompt_ids, seeds, sample, tok.eot_id)
    seconds["sampling_1500_generations"] = time.perf_counter() - start
    start = time.perf_counter()
    g_greedy = generate_all(model, prompt_ids, None, greedy, tok.eot_id)
    seconds["greedy_300_generations"] = time.perf_counter() - start
    g_other = generate_all(model, prompt_ids, other_seeds, sample, tok.eot_id)
    g_penalty = generate_all(model, prompt_ids, seeds, penalised, tok.eot_id)

    result["dev_generation"] = {
        "sampling": summarise(g_sample),
        "greedy": summarise(g_greedy),
        "sampling_repetition_penalty": summarise(g_penalty),
        "sampling_other_seeds": summarise(g_other),
    }
    result["dev_paired"] = {
        # same model, disjoint seeds: how large a difference pure sampling noise produces
        "aa_other_seeds_minus_seeds": paired(g_other, g_sample),
        "repetition_penalty_minus_plain": paired(g_penalty, g_sample),
    }
    result["decode"] = {
        "max_new_tokens": protocol.MAX_NEW_TOKENS,
        "seeds": list(seeds),
        "repetition_penalty": REPETITION_PENALTY,
    }
    result["seconds"] = seconds
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
