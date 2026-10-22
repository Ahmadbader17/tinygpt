import math
from pathlib import Path

import numpy as np

from tinygpt.baselines import bigram_perplexity, compute_baselines, unigram_perplexity


def test_unigram_on_uniform_data_is_near_vocab_size() -> None:
    rng = np.random.default_rng(0)
    train = rng.integers(0, 8, 200_000).astype(np.uint16)
    val = rng.integers(0, 8, 50_000).astype(np.uint16)
    assert abs(unigram_perplexity(train, val, 8) - 8) < 0.05


def test_bigram_learns_a_deterministic_sequence() -> None:
    train = (np.arange(10_000) % 5).astype(np.uint16)  # 0 1 2 3 4 0 1 ...
    val = (np.arange(1_000) % 5).astype(np.uint16)
    assert bigram_perplexity(train, val, 5) < 1.01  # next token is always known
    assert unigram_perplexity(train, val, 5) > 4.9  # unigram cannot see order


def test_bigram_hand_computed_value() -> None:
    train = np.array([0, 1, 0, 1, 0, 0], dtype=np.uint16)  # pairs: 01 10 01 10 00
    val = np.array([0, 1], dtype=np.uint16)
    # count(0,1)=2, count(0,.)=3 -> p = (2 + 1) / (3 + 2) = 0.6
    assert math.isclose(bigram_perplexity(train, val, 2), 1 / 0.6)


def test_compute_baselines_reads_files(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    for name in ("train.bin", "val.bin"):
        rng.integers(0, 16, 5000).astype(np.uint16).tofile(tmp_path / name)
    result = compute_baselines(tmp_path / "train.bin", tmp_path / "val.bin", 16)
    assert result["uniform_perplexity"] == 16
    assert result["bigram_perplexity"] >= result["unigram_perplexity"] * 0.9
