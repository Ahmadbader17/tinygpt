"""The tokenizer interface the rest of the code depends on."""

from collections.abc import Sequence
from typing import Protocol


class Tokenizer(Protocol):
    """Anything that turns text into ids and back. Implementations are swappable."""

    @property
    def vocab_size(self) -> int: ...

    @property
    def eot_id(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...
