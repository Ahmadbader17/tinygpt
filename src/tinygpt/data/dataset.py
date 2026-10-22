"""Training data access: random windows from a uint16 memmap, plus a deterministic dataset."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# LEARN: rewrite this yourself, see docs/LEARNING_PLAN.md item 1
class MemmapTokenSampler:
    """Samples (x, y) windows from a flat uint16 token file.

    The file is memory-mapped, so RAM use is a few pages instead of the whole corpus.
    Overlap between windows comes from random start positions; there is no stride.
    """

    def __init__(self, path: Path, context_length: int) -> None:
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.context_length = context_length
        if len(self.data) <= context_length:
            raise ValueError(f"{path} has {len(self.data)} tokens; need > {context_length}")

    def __len__(self) -> int:
        return len(self.data)

    def batch(self, batch_size: int, rng: np.random.Generator) -> tuple[Tensor, Tensor]:
        """Return x, y of shape (B, T), int64; y is x shifted left by one token."""
        # the last valid start leaves room for context_length + 1 tokens (x plus the extra y token)
        starts = rng.integers(0, len(self.data) - self.context_length, size=batch_size)
        t = self.context_length
        # uint16 -> int64: embedding lookups need int64 and the memmap is read-only
        x = np.stack([self.data[s : s + t] for s in starts]).astype(np.int64)  # (B, T)
        y = np.stack([self.data[s + 1 : s + t + 1] for s in starts]).astype(np.int64)  # (B, T)
        return torch.from_numpy(x), torch.from_numpy(y)


class SlidingWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Deterministic windows over a token sequence, equal to the notebook's `GPTDatasetV1`.

    Used for evaluation and tests where the exact windows must be reproducible.
    """

    def __init__(self, ids: Sequence[int], max_length: int, stride: int) -> None:
        self.input_ids: list[Tensor] = []
        self.target_ids: list[Tensor] = []
        for i in range(0, len(ids) - max_length, stride):
            self.input_ids.append(torch.tensor(ids[i : i + max_length]))
            self.target_ids.append(torch.tensor(ids[i + 1 : i + max_length + 1]))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.input_ids[idx], self.target_ids[idx]
