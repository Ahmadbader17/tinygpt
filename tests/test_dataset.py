from pathlib import Path

import numpy as np
import pytest
import torch

from tinygpt.data.dataset import MemmapTokenSampler, SlidingWindowDataset


@pytest.fixture
def counting_bin(tmp_path: Path) -> Path:
    """Token i is the number i, so any window is easy to verify by eye."""
    path = tmp_path / "counting.bin"
    np.arange(1000, dtype=np.uint16).tofile(path)
    return path


def test_target_is_input_shifted_by_one(counting_bin: Path) -> None:
    x, y = MemmapTokenSampler(counting_bin, 32).batch(8, np.random.default_rng(0))
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(y, x + 1)  # counting data: the next token is always +1


def test_shapes_and_dtype(counting_bin: Path) -> None:
    x, y = MemmapTokenSampler(counting_bin, 16).batch(5, np.random.default_rng(0))
    assert x.shape == y.shape == (5, 16)
    assert x.dtype == y.dtype == torch.int64


def test_windows_never_run_past_the_end(tmp_path: Path) -> None:
    path = tmp_path / "short.bin"
    np.arange(20, dtype=np.uint16).tofile(path)
    x, y = MemmapTokenSampler(path, 8).batch(500, np.random.default_rng(0))
    assert int(y.max()) == 19  # the last token is reachable ...
    assert int(x.min()) == 0  # ... and so is the first, with no out-of-range slice


def test_same_seed_same_batch(counting_bin: Path) -> None:
    sampler = MemmapTokenSampler(counting_bin, 8)
    a = sampler.batch(4, np.random.default_rng(7))
    b = sampler.batch(4, np.random.default_rng(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_too_short_file_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tiny.bin"
    np.arange(4, dtype=np.uint16).tofile(path)
    with pytest.raises(ValueError):
        MemmapTokenSampler(path, 8)


def test_sliding_window_matches_notebook_behaviour() -> None:
    """Same windows as GPTDatasetV1 in notebook 01: range(0, n - max_length, stride)."""
    ids = list(range(10))
    ds = SlidingWindowDataset(ids, max_length=4, stride=2)
    assert len(ds) == 3
    x, y = ds[1]
    assert x.tolist() == [2, 3, 4, 5] and y.tolist() == [3, 4, 5, 6]
    assert ds[-1][1].tolist()[-1] <= 9


@pytest.mark.parametrize(("max_length", "stride"), [(4, 1), (8, 4), (5, 5)])
def test_sliding_window_count(max_length: int, stride: int) -> None:
    ds = SlidingWindowDataset(list(range(50)), max_length, stride)
    assert len(ds) == len(range(0, 50 - max_length, stride))
    assert all(len(x) == len(y) == max_length for x, y in ds)
