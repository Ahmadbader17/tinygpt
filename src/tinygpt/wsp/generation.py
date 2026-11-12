"""Batched generation with a key/value cache, for evaluating many prompts and seeds.

The repo's `generate` re-runs the whole context for every new token and stops a batch only
when every row has ended, which is too slow for thousands of evaluation samples. This module
computes the same distribution (temperature, then top-k, then sample) and adds:

* a key/value cache, so each step runs one token per row through the blocks;
* common random numbers: token t of (prompt p, seed s) is drawn by inverse CDF from a
  uniform number that depends only on (base seed, p, s, t). Two models compared on the same
  prompts and seeds therefore consume identical randomness, whatever the batching;
* an optional decoding-time repetition penalty (Keskar et al. 2019), the conventional fix
  the weight edits are compared against.

tests/test_wsp_generation.py checks the cached logits against `GPT.forward` and greedy
output against `tinygpt.inference.generate`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from tinygpt.model import GPT, Block

GENERATION_SEED = 7_011_2026  # base seed for the uniform numbers; fixed for the whole study


@dataclass(frozen=True)
class DecodeConfig:
    max_new_tokens: int = 200
    temperature: float = 0.8  # 0 means greedy argmax
    top_k: int | None = 40
    repetition_penalty: float = 1.0  # 1.0 disables it


@dataclass(frozen=True)
class Generated:
    tokens: list[int]  # new tokens only, without the prompt and without the stop token
    hit_eot: bool  # True if the model ended the story itself


class _Cache:
    """Keys and values of every layer for the tokens processed so far."""

    def __init__(self, n_layer: int) -> None:
        self.k: list[Tensor | None] = [None] * n_layer
        self.v: list[Tensor | None] = [None] * n_layer
        self.length = 0


def _advance(model: GPT, idx: Tensor, cache: _Cache) -> Tensor:
    """Run new tokens `idx` (B, t) through the model, extending the cache.

    Returns next-token logits for the last new position, shape (B, vocab). The first call
    may pass a whole prompt (t > 1, empty cache); later calls pass one token per row.
    """
    b, t = idx.shape
    if t > 1 and cache.length:
        raise ValueError("only the first call may pass more than one token per row")
    if cache.length + t > model.cfg.context_length:
        raise ValueError("generation would exceed the model's context length")
    pos = torch.arange(cache.length, cache.length + t, device=idx.device)  # (t,)
    x = model.wte(idx) + model.wpe(pos)  # (B, t, C); dropout is off in eval mode
    c = x.shape[-1]
    for i, block in enumerate(model.blocks):
        assert isinstance(block, Block)
        attn = block.attn
        q, k, v = attn.qkv(block.ln1(x)).split(c, dim=2)  # 3 x (B, t, C)
        # (B, t, C) -> (B, n_head, t, head_dim)
        q = q.view(b, t, attn.n_head, attn.head_dim).transpose(1, 2)
        k = k.view(b, t, attn.n_head, attn.head_dim).transpose(1, 2)
        v = v.view(b, t, attn.n_head, attn.head_dim).transpose(1, 2)
        old_k, old_v = cache.k[i], cache.v[i]
        if old_k is not None and old_v is not None:
            k = torch.cat([old_k, k], dim=2)  # (B, n_head, T_so_far, head_dim)
            v = torch.cat([old_v, v], dim=2)
        cache.k[i], cache.v[i] = k, v
        # a single new token may see every cached position; a fresh prompt needs the mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=t > 1)  # (B, n_head, t, head_dim)
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # (B, t, C)
        x = x + attn.proj(y)
        x = x + block.mlp(block.ln2(x))
    cache.length += t
    logits: Tensor = model.lm_head(model.ln_f(x[:, -1]))  # (B, C) -> (B, vocab)
    return logits


def next_token(logits: Tensor, cfg: DecodeConfig, uniform: Tensor | None, seen: Tensor) -> Tensor:
    """Pick the next token of each row. `seen` (B, vocab) marks tokens already in the row."""
    if cfg.repetition_penalty != 1.0:
        # shrink the score of every token already used: divide if positive, multiply if not
        penalised = torch.where(
            logits > 0, logits / cfg.repetition_penalty, logits * cfg.repetition_penalty
        )
        logits = torch.where(seen, penalised, logits)
    if cfg.temperature == 0:
        return logits.argmax(dim=-1)
    if uniform is None:
        raise ValueError("sampling needs uniform random numbers")
    logits = logits / cfg.temperature
    if cfg.top_k is not None:
        kth = torch.topk(logits, min(cfg.top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    cdf = torch.softmax(logits.double(), dim=-1).cumsum(dim=-1)  # (B, vocab), float64
    # first index whose cumulative probability exceeds u: that token has non-zero probability
    target = (uniform.double() * cdf[:, -1]).unsqueeze(1)  # (B, 1)
    choice = torch.searchsorted(cdf, target, right=True).squeeze(1)  # (B,)
    return choice.clamp(max=logits.size(-1) - 1)


@torch.inference_mode()
def generate_batch(
    model: GPT, prompts: Tensor, cfg: DecodeConfig, uniforms: Tensor | None, eot_id: int
) -> list[Generated]:
    """Continue every row of `prompts` (B, T0), all of the same length.

    `uniforms` is (B, max_new_tokens) in [0, 1), or None for greedy decoding.
    """
    if model.training:
        raise ValueError("put the model in eval mode before generating")
    b, t0 = prompts.shape
    if t0 + cfg.max_new_tokens > model.cfg.context_length:
        raise ValueError("prompt plus max_new_tokens must fit in the context")
    cache = _Cache(len(model.blocks))
    seen = torch.zeros(b, model.cfg.vocab_size, dtype=torch.bool)
    seen.scatter_(1, prompts, True)
    done = torch.zeros(b, dtype=torch.bool)
    out = torch.full((b, cfg.max_new_tokens), eot_id, dtype=torch.long)
    length = torch.full((b,), cfg.max_new_tokens, dtype=torch.long)
    logits = _advance(model, prompts, cache)
    for step in range(cfg.max_new_tokens):
        u = None if uniforms is None else uniforms[:, step]
        nxt = next_token(logits, cfg, u, seen)  # (B,)
        newly_done = (nxt == eot_id) & ~done
        length[newly_done] = step
        done |= newly_done
        out[:, step] = nxt
        seen.scatter_(1, nxt.unsqueeze(1), True)
        if bool(done.all()) or step == cfg.max_new_tokens - 1:
            break
        logits = _advance(model, nxt.unsqueeze(1), cache)  # finished rows are simply ignored
    return [Generated(out[r, : int(length[r])].tolist(), bool(done[r])) for r in range(b)]


def uniforms_for(prompt_index: int, seed: int, max_new_tokens: int) -> np.ndarray:
    """The random numbers of one (prompt, seed) pair; independent of batching and of the model."""
    rng = np.random.default_rng([GENERATION_SEED, prompt_index, seed])
    return rng.random(max_new_tokens)


def generate_all(
    model: GPT,
    prompts: Sequence[Sequence[int]],
    seeds: Sequence[int] | None,
    cfg: DecodeConfig,
    eot_id: int,
    batch_size: int = 128,
) -> list[list[Generated]]:
    """Generate for every prompt and seed; result[p][s]. `seeds=None` means one greedy run.

    Rows are grouped by prompt length because the model has no padding token.
    """
    greedy = cfg.temperature == 0
    if greedy != (seeds is None):
        raise ValueError("pass seeds for sampling, and seeds=None for greedy decoding")
    seed_list: list[int] = [0] if seeds is None else list(seeds)
    rows = [(p, s) for p in range(len(prompts)) for s in range(len(seed_list))]
    rows.sort(key=lambda r: (len(prompts[r[0]]), r))
    result: list[list[Generated | None]] = [[None] * len(seed_list) for _ in prompts]
    start = 0
    while start < len(rows):
        t0 = len(prompts[rows[start][0]])
        end = start
        while end < len(rows) and end - start < batch_size and len(prompts[rows[end][0]]) == t0:
            end += 1
        chunk = rows[start:end]
        idx = torch.tensor([list(prompts[p]) for p, _ in chunk], dtype=torch.long)
        uniforms = None
        if not greedy:
            drawn = [uniforms_for(p, seed_list[s], cfg.max_new_tokens) for p, s in chunk]
            uniforms = torch.from_numpy(np.stack(drawn))  # (B, max_new_tokens)
        for (p, s), gen in zip(
            chunk, generate_batch(model, idx, cfg, uniforms, eot_id), strict=True
        ):
            result[p][s] = gen
        start = end
    return [[g for g in row if g is not None] for row in result]
