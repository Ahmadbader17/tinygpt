"""The injector's calibration logic, with the model replaced by a stub. No checkpoint is read."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def _module() -> ModuleType:
    path = REPO / "experiments" / "wsp" / "inject_faults.py"
    spec = importlib.util.spec_from_file_location("inject_faults", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _calibrator(inj: ModuleType, cost_of: Any) -> Any:
    cal = object.__new__(inj.Calibrator)  # skip __init__: no model, no data
    cal.damage = lambda spec: (cost_of(spec), True)
    return cal


def test_magnitude_is_scaled_up_into_the_window() -> None:
    inj = _module()
    cal = _calibrator(inj, lambda spec: 0.004 * spec["factor"])  # damage grows with the factor
    ok, spec = cal.calibrate({"family": "head_scale", "factor": 3.0})
    assert ok and inj.DAMAGE_MIN <= spec["dev_damage"] <= inj.DAMAGE_MAX
    assert spec["factor"] > 3.0 and abs(spec["factor"] / 3.0 - 1.5**4) < 1e-9


def test_magnitude_is_scaled_down_into_the_window() -> None:
    inj = _module()
    cal = _calibrator(inj, lambda spec: 2.0 * spec["std_rel"])
    ok, spec = cal.calibrate({"family": "noise", "std_rel": 1.2})
    assert ok and spec["dev_damage"] <= inj.DAMAGE_MAX and spec["std_rel"] < 1.2


def test_a_fault_that_cannot_reach_the_window_is_not_confirmed() -> None:
    inj = _module()
    cal = _calibrator(inj, lambda spec: 0.0001)
    ok, _ = cal.calibrate({"family": "ln_gain", "factor": 5.0})
    assert not ok


def test_zeroed_head_has_no_magnitude_and_a_lower_floor() -> None:
    inj = _module()
    ok, spec = _calibrator(inj, lambda spec: 0.03).calibrate({"family": "head_zero"})
    assert ok and spec["dev_damage"] == 0.03
    ok, _ = _calibrator(inj, lambda spec: 0.01).calibrate({"family": "head_zero"})
    assert not ok
