"""GPT-2 tokenizer behind the same protocol, for comparison only (`--group compare`)."""

from collections.abc import Sequence

import tiktoken


class TiktokenAdapter:
    def __init__(self, encoding: str = "gpt2") -> None:
        self._enc = tiktoken.get_encoding(encoding)

    @property
    def vocab_size(self) -> int:
        return int(self._enc.n_vocab)

    @property
    def eot_id(self) -> int:
        return int(self._enc.eot_token)

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids: Sequence[int]) -> str:
        return self._enc.decode(list(ids))
