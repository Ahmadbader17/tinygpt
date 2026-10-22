from pathlib import Path

import pytest

from tinygpt.tokenizer import BPETokenizer, train_bpe

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_text() -> str:
    return (FIXTURES / "sample_stories.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def tiny_tokenizer(sample_text: str) -> BPETokenizer:
    """Vocab-512 BPE trained on the fixture stories (matches configs/tiny.toml)."""
    return train_bpe(sample_text.split("\n\n"), vocab_size=512)
