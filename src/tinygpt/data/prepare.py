"""Raw text -> cleaned stories -> uint16 token files + meta.json."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from tinygpt.config import Config
from tinygpt.data.download import open_stories
from tinygpt.tokenizer import BPETokenizer, train_bpe

log = logging.getLogger(__name__)

# control characters except \n (tabs are turned into spaces first)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_SPACES = re.compile(r"[ \t]+")
STORY_FILE_SEP = "\n\n"  # stories never contain a blank line after cleaning


def clean_story(text: str) -> str:
    """Light cleaning: normalise whitespace, strip control characters.

    Every run of newlines becomes a single "\\n", so a blank line can safely separate
    stories in the raw files. Returns "" for stories with no content.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", _SPACES.sub(" ", text))
    lines = (line.strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


@dataclass(frozen=True)
class CleanStats:
    read: int
    kept: int

    @property
    def dropped(self) -> int:
        return self.read - self.kept


def write_clean_split(stories: Iterable[str], out_path: Path) -> CleanStats:
    """Clean `stories` and write them blank-line separated. Logs counts before and after."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    read = kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for story in stories:
            read += 1
            cleaned = clean_story(story)
            if cleaned:
                f.write(cleaned + STORY_FILE_SEP)
                kept += 1
    stats = CleanStats(read, kept)
    log.info("%s: read %d stories, kept %d, dropped %d", out_path.name, read, kept, stats.dropped)
    return stats


def read_clean_split(path: Path) -> Iterator[str]:
    """Stream stories back from a file written by `write_clean_split`."""
    story: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                story.append(line)
            elif story:
                yield "\n".join(story)
                story = []
    if story:
        yield "\n".join(story)


def encode_split(
    tokenizer: BPETokenizer, stories: Iterable[str], out_path: Path, batch_size: int = 2000
) -> int:
    """Encode stories, append <|endoftext|> after each, write raw uint16. Returns token count."""
    if tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("vocab does not fit in uint16")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    batch: list[str] = []

    def flush(f: BinaryIO) -> int:
        arrays = []
        for ids in tokenizer.encode_batch(batch):
            ids.append(tokenizer.eot_id)
            arrays.append(np.asarray(ids, dtype=np.uint16))
        data = np.concatenate(arrays)
        f.write(data.tobytes())
        return int(data.size)

    with open(out_path, "wb") as f:
        for story in stories:
            batch.append(story)
            if len(batch) >= batch_size:
                total += flush(f)
                batch = []
        if batch:
            total += flush(f)
    return total


def prepare_data(
    cfg: Config,
    data_dir: Path,
    train_source: str | None = None,
    val_source: str | None = None,
) -> dict[str, object]:
    """Run the whole pipeline: stream + clean -> tokenizer -> encode -> meta.json.

    Sources default to the URLs in `cfg.data`; pass local paths for tests or offline runs.
    """
    train_source = train_source or cfg.data.train_url
    val_source = val_source or cfg.data.val_url
    raw, processed = data_dir / "raw", data_dir / "processed"

    train_stats = write_clean_split(
        open_stories(train_source, cfg.data.train_stories), raw / "train.txt"
    )
    val_stats = write_clean_split(open_stories(val_source, cfg.data.val_stories), raw / "val.txt")

    tok_path = data_dir / "tokenizer" / "tokenizer.json"
    # the tokenizer sees training text only, so validation stays a fair held-out set
    tokenizer = train_bpe(read_clean_split(raw / "train.txt"), cfg.data.tokenizer_vocab_size)
    tokenizer.save(tok_path)

    train_tokens = encode_split(
        tokenizer, read_clean_split(raw / "train.txt"), processed / "train.bin"
    )
    val_tokens = encode_split(tokenizer, read_clean_split(raw / "val.txt"), processed / "val.bin")

    meta: dict[str, object] = {
        "dtype": "uint16",
        "vocab_size": tokenizer.vocab_size,
        "eot_id": tokenizer.eot_id,
        "tokenizer_sha256": BPETokenizer.file_sha256(tok_path),
        "train": {
            "stories": train_stats.kept,
            "tokens": train_tokens,
            "tokens_per_story": round(train_tokens / max(train_stats.kept, 1), 1),
            "dropped": train_stats.dropped,
        },
        "val": {
            "stories": val_stats.kept,
            "tokens": val_tokens,
            "tokens_per_story": round(val_tokens / max(val_stats.kept, 1), 1),
            "dropped": val_stats.dropped,
        },
        "source": {"train": train_source, "val": val_source},
        "licence": "CDLA-Sharing-1.0 (roneneldan/TinyStories)",
        "config": asdict(cfg.data),
    }
    (processed / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote %s", processed / "meta.json")
    return meta
