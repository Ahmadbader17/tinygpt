"""Oracle for rewrite exercise 2 (docs/LEARNING_PLAN.md): your own pure-Python byte-level BPE.

Skipped until `src/tinygpt/tokenizer/my_bpe.py` exists. Expected API:
    train_merges(text: str, num_merges: int) -> list[tuple[int, int]]  # merge i -> token 256 + i
    encode(text: str, merges: list[tuple[int, int]]) -> list[int]
    decode(ids: list[int], merges: list[tuple[int, int]]) -> str
"""

from pathlib import Path

import pytest

my_bpe = pytest.importorskip("tinygpt.tokenizer.my_bpe")
FIXTURE = (Path(__file__).parent / "fixtures" / "sample_stories.txt").read_text(encoding="utf-8")


def test_first_merge_is_the_most_frequent_pair() -> None:
    merges = my_bpe.train_merges("aaabdaaabac", 1)
    assert merges == [(97, 97)]  # "aa" occurs 4 times, more than any other pair


@pytest.mark.parametrize(
    "text", ["hello world", "Once upon a time.", "naïve café", "emoji 🦊 fox", "", "\n\n  tabs\t"]
)
def test_round_trip_is_lossless(text: str) -> None:
    merges = my_bpe.train_merges(FIXTURE, 100)
    assert my_bpe.decode(my_bpe.encode(text, merges), merges) == text


def test_merges_shorten_the_sequence() -> None:
    merges = my_bpe.train_merges(FIXTURE, 100)
    assert len(my_bpe.encode(FIXTURE, merges)) < len(FIXTURE.encode("utf-8")) * 0.8


def test_training_is_deterministic() -> None:
    assert my_bpe.train_merges(FIXTURE, 50) == my_bpe.train_merges(FIXTURE, 50)


def test_ids_stay_inside_the_vocabulary() -> None:
    merges = my_bpe.train_merges(FIXTURE, 50)
    assert max(my_bpe.encode(FIXTURE, merges)) < 256 + 50
