"""Byte-level BPE built on Hugging Face `tokenizers`.

Byte-level means the base alphabet is the 256 byte values, so any string can be encoded and
there is no `<|unk|>` token (compare the notebook's "Akwirw ier" example).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

EOT_TOKEN = "<|endoftext|>"


class BPETokenizer:
    """Wraps a trained HF byte-level BPE; satisfies the `Tokenizer` protocol."""

    def __init__(self, hf: HFTokenizer) -> None:
        self._hf = hf
        eot = hf.token_to_id(EOT_TOKEN)
        if eot is None:
            raise ValueError(f"tokenizer has no {EOT_TOKEN} token")
        self._eot_id: int = eot

    @property
    def vocab_size(self) -> int:
        return self._hf.get_vocab_size()

    @property
    def eot_id(self) -> int:
        return self._eot_id

    def encode(self, text: str) -> list[int]:
        # Special tokens in the text are parsed to their id, so "<|endoftext|>" round-trips.
        return list(self._hf.encode(text).ids)

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Encode many texts at once; the Rust backend spreads them over all cores."""
        return [list(e.ids) for e in self._hf.encode_batch(list(texts))]

    def decode(self, ids: Sequence[int]) -> str:
        return self._hf.decode(list(ids), skip_special_tokens=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._hf.save(str(path))

    @classmethod
    def load(cls, path: Path) -> BPETokenizer:
        return cls(HFTokenizer.from_file(str(path)))

    @staticmethod
    def file_sha256(path: Path) -> str:
        """Hash of the saved file; stored in meta.json to tie data to its tokenizer."""
        return hashlib.sha256(path.read_bytes()).hexdigest()


def train_bpe(texts: Iterable[str], vocab_size: int) -> BPETokenizer:
    """Learn BPE merges from `texts`.

    Merges are learned by repeatedly fusing the most frequent adjacent pair, starting
    from single bytes. `vocab_size` counts the 256 bytes, the special token and the merges.
    """
    hf = HFTokenizer(models.BPE())
    # add_prefix_space=False: " the" and "the" stay distinct, as in GPT-2.
    hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    hf.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    hf.train_from_iterator(texts, trainer=trainer)
    return BPETokenizer(hf)
