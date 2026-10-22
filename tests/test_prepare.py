import json
from pathlib import Path

import numpy as np
import pytest

from tinygpt.config import load_config
from tinygpt.data.download import STORY_SEPARATOR, open_stories, split_stories
from tinygpt.data.prepare import (
    clean_story,
    encode_split,
    prepare_data,
    read_clean_split,
    write_clean_split,
)
from tinygpt.tokenizer import BPETokenizer

CONFIGS = Path(__file__).parent.parent / "configs"


def _chunked(data: bytes, size: int) -> list[bytes]:
    return [data[i : i + size] for i in range(0, len(data), size)]


@pytest.mark.parametrize("size", [1, 2, 3, 7, 1000])
def test_split_stories_is_chunk_size_independent(size: int) -> None:
    raw = f"Café 🦊 one.\n{STORY_SEPARATOR}\nTwo ünï.\n{STORY_SEPARATOR}\nThree.".encode()
    stories = [s.strip() for s in split_stories(_chunked(raw, size))]
    assert stories == ["Café 🦊 one.", "Two ünï.", "Three."]


def test_split_stories_limit_stops_early() -> None:
    raw = (STORY_SEPARATOR.join(f"story {i}" for i in range(10)) + STORY_SEPARATOR).encode()
    assert len(list(split_stories(_chunked(raw, 5), limit=3))) == 3


def test_clean_story() -> None:
    assert (
        clean_story("  Hello \t  world \r\n\r\n\n  Next   line\x00\x07 ")
        == "Hello world\nNext line"
    )
    assert clean_story(" \n\t\n") == ""
    assert "\n\n" not in clean_story("a\n\n\n\nb")


def test_clean_split_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "s.txt"
    stats = write_clean_split(["One.\nTwo.", "   ", "Three."], path)
    assert (stats.read, stats.kept, stats.dropped) == (3, 2, 1)
    assert list(read_clean_split(path)) == ["One.\nTwo.", "Three."]


def test_encode_split_appends_eot_after_each_story(
    tiny_tokenizer: BPETokenizer, tmp_path: Path
) -> None:
    stories = ["Once upon a time.", "The end.", "Tom had a ball."]
    out = tmp_path / "x.bin"
    n = encode_split(tiny_tokenizer, stories, out, batch_size=2)  # forces two batches
    data = np.fromfile(out, dtype=np.uint16)
    assert n == data.size
    assert int((data == tiny_tokenizer.eot_id).sum()) == 3
    assert data[-1] == tiny_tokenizer.eot_id
    decoded = tiny_tokenizer.decode(data.tolist()).split(STORY_SEPARATOR)
    assert decoded[:3] == stories


@pytest.fixture
def stories_file(sample_text: str, tmp_path: Path) -> Path:
    path = tmp_path / "stories.txt"
    stories = sample_text.strip().split("\n\n")
    path.write_text(STORY_SEPARATOR.join(f"\n{s}\n" for s in stories), encoding="utf-8")
    return path


def test_prepare_data_end_to_end(stories_file: Path, tmp_path: Path) -> None:
    cfg = load_config(CONFIGS / "tiny.toml")
    data_dir = tmp_path / "data"
    meta = prepare_data(cfg, data_dir, str(stories_file), str(stories_file))

    processed = data_dir / "processed"
    train = np.fromfile(processed / "train.bin", dtype=np.uint16)
    on_disk = json.loads((processed / "meta.json").read_text())
    assert on_disk["train"]["tokens"] == train.size == meta["train"]["tokens"]  # type: ignore[index]
    assert on_disk["train"]["stories"] == 5
    assert int((train == on_disk["eot_id"]).sum()) == 5
    assert int(train.max()) < on_disk["vocab_size"]
    assert len(on_disk["tokenizer_sha256"]) == 64
    assert (data_dir / "tokenizer" / "tokenizer.json").exists()


def test_open_stories_local_limit(stories_file: Path) -> None:
    assert len(list(open_stories(str(stories_file), 2))) == 2
