"""Stream the first N stories of a TinyStories text file, without downloading the rest.

Dataset: roneneldan/TinyStories (licence CDLA-Sharing-1.0, see docs/DATA.md). Stories in the
text files are separated by `<|endoftext|>` lines.
"""

from __future__ import annotations

import codecs
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path

STORY_SEPARATOR = "<|endoftext|>"
_CHUNK_BYTES = 1 << 20


def split_stories(chunks: Iterable[bytes], limit: int | None = None) -> Iterator[str]:
    """Yield stories from a byte stream, stopping after `limit`.

    The incremental decoder matters: a chunk boundary can fall inside a multi-byte UTF-8
    character, and decoding each chunk separately would corrupt it.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""
    count = 0
    for chunk in chunks:
        buffer += decoder.decode(chunk)
        *done, buffer = buffer.split(STORY_SEPARATOR)
        for story in done:
            yield story
            count += 1
            if limit is not None and count >= limit:
                return
    buffer += decoder.decode(b"", final=True)
    # a trailing story with no separator is complete only if the file really ended there
    if buffer.strip() and (limit is None or count < limit):
        yield buffer


def stream_url(url: str, limit: int | None) -> Iterator[str]:
    """Stream stories over HTTP; leaving the generator early closes the connection."""
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (our own configured https URL)
        yield from split_stories(iter(lambda: resp.read(_CHUNK_BYTES), b""), limit)


def stream_file(path: Path, limit: int | None) -> Iterator[str]:
    """Same as `stream_url` for a local file (tests, offline use)."""
    with open(path, "rb") as f:
        yield from split_stories(iter(lambda: f.read(_CHUNK_BYTES), b""), limit)


def open_stories(source: str, limit: int | None) -> Iterator[str]:
    """`source` is an http(s) URL or a local path."""
    if source.startswith(("http://", "https://")):
        return stream_url(source, limit)
    return stream_file(Path(source), limit)
