"""Unigram and bigram baselines: how well do plain counts predict the validation tokens?

A trained model must beat these clearly, otherwise it has only learned token frequencies.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

CHUNK = 5_000_000  # keeps temporary int64 arrays small on a 16 GB machine


def _load(path: Path) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode="r")


def unigram_perplexity(train: np.ndarray, val: np.ndarray, vocab: int) -> float:
    """p(token) = (count + 1) / (N + V), add-one smoothing so unseen tokens are not impossible."""
    counts = np.zeros(vocab, dtype=np.int64)
    for i in range(0, len(train), CHUNK):
        counts += np.bincount(train[i : i + CHUNK], minlength=vocab)
    logp = np.log((counts + 1) / (counts.sum() + vocab))
    return math.exp(-float(logp[val].mean()))


def bigram_perplexity(train: np.ndarray, val: np.ndarray, vocab: int) -> float:
    """p(b | a) = (count(a, b) + 1) / (count(a) + V), a (V, V) table of next-token counts."""
    counts = np.zeros(vocab * vocab, dtype=np.int64)
    for i in range(0, len(train) - 1, CHUNK):
        a = train[i : i + CHUNK + 1][:-1].astype(np.int64)
        b = train[i + 1 : i + CHUNK + 1].astype(np.int64)
        counts += np.bincount(a * vocab + b, minlength=vocab * vocab)
    table = counts.reshape(vocab, vocab)
    row_totals = table.sum(axis=1)
    a, b = val[:-1].astype(np.int64), val[1:].astype(np.int64)
    logp = np.log((table[a, b] + 1) / (row_totals[a] + vocab))
    return math.exp(-float(logp.mean()))


def compute_baselines(train_bin: Path, val_bin: Path, vocab: int) -> dict[str, float]:
    train, val = _load(train_bin), _load(val_bin)
    return {
        "unigram_perplexity": unigram_perplexity(train, val, vocab),
        "bigram_perplexity": bigram_perplexity(train, val, vocab),
        "uniform_perplexity": float(vocab),
    }
