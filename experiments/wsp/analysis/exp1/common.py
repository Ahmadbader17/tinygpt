"""Shared helpers for the Exp 1 weight reading. Weights only: nothing here runs the model.

LayerNorm is folded as PLAN.md interpretation I2 allows: LN(x) = g * (x - mean(x)) / s + b is
linear in x apart from the scalar s, so each component v of the residual stream contributes
g * centre(v) / s. For s we use one fixed typical value per analysis, never a value computed
from a token sequence.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp.edits import load_base_params

REPO = Path(__file__).resolve().parents[4]
ORIGINAL = REPO / "runs" / "wsp" / "original"
N_LAYER, N_HEAD, D_MODEL, HEAD_DIM = 6, 8, 256, 32
HEADS = [(layer, head) for layer in range(N_LAYER) for head in range(N_HEAD)]


def load() -> dict[str, Tensor]:
    return {k: v.double() for k, v in load_base_params(ORIGINAL).items()}


def tokenizer() -> BPETokenizer:
    return BPETokenizer.load(ORIGINAL / "tokenizer.json")


def centre(x: Tensor) -> Tensor:
    """Remove the mean over the feature dimension, as LayerNorm does."""
    return x - x.mean(dim=-1, keepdim=True)


def head_rows(head: int) -> slice:
    return slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)


def qkv(p: dict[str, Tensor], layer: int, head: int) -> tuple[Tensor, ...]:
    """W_Q, W_K, W_V (each 32 x 256) and b_Q, b_K, b_V (each 32) of one head."""
    w, b = p[f"blocks.{layer}.attn.qkv.weight"], p[f"blocks.{layer}.attn.qkv.bias"]
    r = head_rows(head)
    wq, wk, wv = w[:D_MODEL][r], w[D_MODEL : 2 * D_MODEL][r], w[2 * D_MODEL :][r]
    bq, bk, bv = b[:D_MODEL][r], b[D_MODEL : 2 * D_MODEL][r], b[2 * D_MODEL :][r]
    return wq, wk, wv, bq, bk, bv


def w_out(p: dict[str, Tensor], layer: int, head: int) -> Tensor:
    """W_O (256 x 32): how one head's mixed values are written into the residual stream."""
    return p[f"blocks.{layer}.attn.proj.weight"][:, head_rows(head)]


def ln1(p: dict[str, Tensor], layer: int) -> tuple[Tensor, Tensor]:
    return p[f"blocks.{layer}.ln1.weight"], p[f"blocks.{layer}.ln1.bias"]


def embedding_scale(p: dict[str, Tensor]) -> float:
    """Typical LayerNorm scale s of a layer-0 input wte[token] + wpe[position].

    Token and position embeddings are nearly orthogonal at random, so the variance of the sum
    is taken as the sum of the two mean variances. One number, from the embeddings alone.
    """
    tok_var = centre(p["wte.weight"]).pow(2).mean()
    pos_var = centre(p["wpe.weight"]).pow(2).mean()
    return float((tok_var + pos_var).sqrt())


def fmt_row(values: Tensor, width: int = 7, digits: int = 2) -> str:
    return " ".join(f"{float(v):{width}.{digits}f}" for v in values)
