from pathlib import Path

import pytest

from tinygpt.tokenizer import EOT_TOKEN, BPETokenizer, Tokenizer, train_bpe


@pytest.mark.parametrize(
    "text",
    [
        "Once upon a time, a little fox",
        "Hello, world!  Two  spaces\nand a newline\tand a tab.",
        'Punctuation: (a) [b] {c} -- "quoted" it\'s 100% #1?',
        "Unicode: café, naïve, 日本語, Привет",
        "Emoji: 🦊🌳 and a ZWJ family 👨‍👩‍👧",
        "Akwirw ier",  # the notebook's out-of-vocabulary example
        "",
    ],
)
def test_round_trip_is_lossless(tiny_tokenizer: BPETokenizer, text: str) -> None:
    assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text


def test_vocab_size_matches_request(tiny_tokenizer: BPETokenizer) -> None:
    assert tiny_tokenizer.vocab_size == 512


def test_eot_maps_to_eot_id_and_round_trips(tiny_tokenizer: BPETokenizer) -> None:
    ids = tiny_tokenizer.encode(f"The end.{EOT_TOKEN}Next story")
    assert ids.count(tiny_tokenizer.eot_id) == 1
    assert tiny_tokenizer.decode(ids) == f"The end.{EOT_TOKEN}Next story"


def test_ids_stay_inside_vocab(tiny_tokenizer: BPETokenizer) -> None:
    ids = tiny_tokenizer.encode("anything 🦊 at all, even Привет")
    assert all(0 <= i < tiny_tokenizer.vocab_size for i in ids)


def test_merges_compress_trained_text(tiny_tokenizer: BPETokenizer, sample_text: str) -> None:
    n_bytes = len(sample_text.encode("utf-8"))
    assert len(tiny_tokenizer.encode(sample_text)) < 0.6 * n_bytes


def test_training_is_deterministic(sample_text: str) -> None:
    a = train_bpe(sample_text.split("\n\n"), vocab_size=400)
    b = train_bpe(sample_text.split("\n\n"), vocab_size=400)
    assert a.encode(sample_text) == b.encode(sample_text)


def test_save_load_gives_same_ids(
    tiny_tokenizer: BPETokenizer, sample_text: str, tmp_path: Path
) -> None:
    path = tmp_path / "tok" / "tokenizer.json"
    tiny_tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.encode(sample_text) == tiny_tokenizer.encode(sample_text)
    assert loaded.eot_id == tiny_tokenizer.eot_id
    assert len(BPETokenizer.file_sha256(path)) == 64


def test_satisfies_protocol(tiny_tokenizer: BPETokenizer) -> None:
    tok: Tokenizer = tiny_tokenizer
    assert tok.vocab_size > 0


def test_gpt2_comparison(tiny_tokenizer: BPETokenizer, sample_text: str) -> None:
    """GPT-2's 50k vocab compresses better but costs ~100x more embedding parameters."""
    pytest.importorskip("tiktoken")
    from tinygpt.tokenizer.tiktoken_adapter import TiktokenAdapter

    try:
        gpt2 = TiktokenAdapter("gpt2")
    except Exception as exc:  # no network to fetch the vocab file
        pytest.skip(f"gpt2 vocab unavailable: {exc}")
    assert gpt2.decode(gpt2.encode(sample_text)) == sample_text
    assert len(gpt2.encode(sample_text)) <= len(tiny_tokenizer.encode(sample_text))
    assert gpt2.vocab_size > 50_000
