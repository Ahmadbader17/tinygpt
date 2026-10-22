import math

import pytest

from tinygpt.config import TrainConfig
from tinygpt.training import lr_at

CFG = TrainConfig(max_steps=1000, warmup_steps=100, lr_peak=1e-3, lr_min=1e-4)


def test_warmup_rises_linearly_to_peak() -> None:
    assert lr_at(0, CFG) == pytest.approx(1e-5)  # (0 + 1) / 100 of the peak
    assert lr_at(49, CFG) == pytest.approx(5e-4)
    assert lr_at(99, CFG) == pytest.approx(1e-3)  # last warmup step hits the peak


def test_cosine_starts_at_peak_and_ends_at_floor() -> None:
    assert lr_at(100, CFG) == pytest.approx(1e-3)
    assert lr_at(1000, CFG) == pytest.approx(1e-4)
    assert lr_at(5000, CFG) == pytest.approx(1e-4)  # stays at the floor afterwards


def test_cosine_midpoint_is_halfway() -> None:
    mid = (100 + 1000) // 2
    assert lr_at(mid, CFG) == pytest.approx((1e-3 + 1e-4) / 2)


def test_decay_is_monotone_after_warmup() -> None:
    lrs = [lr_at(s, CFG) for s in range(100, 1001)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:], strict=False))
    assert all(math.isfinite(v) and v > 0 for v in lrs)
