from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tinygpt.config import load_config
from tinygpt.model import GPT
from tinygpt.wsp.controls import edit_ranks, matched_gaussian, matched_low_rank
from tinygpt.wsp.faults import FAMILIES, apply_fault, draw_fault, head_columns, score_diagnosis
from tinygpt.wsp.metrics import bootstrap_recovery

CFG = load_config(Path(__file__).parent.parent / "configs" / "tiny.toml").model


def params() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {n: p.detach().clone() for n, p in GPT(CFG).named_parameters()}


def draw(family: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return draw_fault(family, rng, CFG.n_layer, CFG.n_head, CFG.d_model)


@pytest.mark.parametrize("family", FAMILIES)
def test_a_fault_changes_only_its_tensor_and_is_deterministic(family: str) -> None:
    base = params()
    spec = draw(family)
    a = apply_fault(base, spec, CFG.n_head)
    b = apply_fault(base, spec, CFG.n_head)
    changed = [n for n in base if not torch.equal(base[n], a[n])]
    assert changed == [spec["tensor"]]
    assert all(torch.equal(a[n], b[n]) for n in base)


def test_head_faults_touch_exactly_one_head_slice() -> None:
    base = params()
    spec = draw("head_scale")
    cols = head_columns(spec["head"], CFG.n_head, CFG.d_model)
    w0, w1 = base[spec["tensor"]], apply_fault(base, spec, CFG.n_head)[spec["tensor"]]
    assert torch.allclose(w1[:, cols], w0[:, cols] * spec["factor"])
    untouched = torch.ones(CFG.d_model, dtype=torch.bool)
    untouched[cols] = False
    assert torch.equal(w1[:, untouched], w0[:, untouched])
    zero = draw("head_zero")
    wz = apply_fault(base, zero, CFG.n_head)[zero["tensor"]]
    assert float(wz[:, head_columns(zero["head"], CFG.n_head, CFG.d_model)].abs().sum()) == 0.0


def test_rank_one_fault_adds_a_rank_one_term_of_the_stated_size() -> None:
    base = params()
    spec = draw("rank_one")
    delta = apply_fault(base, spec, CFG.n_head)[spec["tensor"]] - base[spec["tensor"]]
    sv = torch.linalg.svdvals(delta)
    assert float(sv[1] / sv[0]) < 1e-4
    top = torch.linalg.matrix_norm(base[spec["tensor"]], ord=2).item()
    assert abs(float(sv[0]) - spec["alpha_rel"] * top) < 1e-3 * top


def test_ln_gain_fault_scales_a_quarter_of_the_gains() -> None:
    base = params()
    spec = draw("ln_gain")
    g0, g1 = base[spec["tensor"]], apply_fault(base, spec, CFG.n_head)[spec["tensor"]]
    assert len(spec["indices"]) == CFG.d_model // 4
    assert int((g0 != g1).sum()) == CFG.d_model // 4
    assert torch.allclose(g1[spec["indices"]], g0[spec["indices"]] * spec["factor"])


def test_diagnosis_scoring() -> None:
    spec = {"tensor": "blocks.1.attn.proj.weight", "head": 1, "family": "head_zero"}
    right = {"tensor": "blocks.1.attn.proj.weight", "head": 1}
    wrong_head = {"tensor": "blocks.1.attn.proj.weight", "head": 0}
    wrong = {"tensor": "blocks.0.mlp.proj.weight", "head": None}
    assert score_diagnosis(right, spec) == {"tensor_correct": True, "sublocation_correct": True}
    assert score_diagnosis(wrong_head, spec) == {
        "tensor_correct": True,
        "sublocation_correct": False,
    }
    assert score_diagnosis(wrong, spec) == {"tensor_correct": False, "sublocation_correct": False}
    no_head = {"tensor": "blocks.0.ln1.weight", "head": None, "family": "ln_gain"}
    assert score_diagnosis({"tensor": "blocks.0.ln1.weight"}, no_head) == {
        "tensor_correct": True,
        "sublocation_correct": None,
    }


def test_gaussian_control_matches_support_and_norm() -> None:
    base = params()
    edited = {n: v.clone() for n, v in base.items()}
    edited["blocks.0.attn.proj.weight"][:, :32] *= 0.5
    edited["ln_f.weight"][:5] += 1.0
    c0 = matched_gaussian(base, edited, seed=0)
    c1 = matched_gaussian(base, edited, seed=1)
    for name in base:
        mask = base[name] != edited[name]
        assert torch.equal(c0[name][~mask], base[name][~mask])  # nothing outside the support
        want = (edited[name].double() - base[name].double()).norm()
        got = (c0[name].double() - base[name].double()).norm()
        assert abs(float(got - want)) <= 1e-4 * float(want) + 1e-9
    assert not torch.equal(c0["ln_f.weight"], c1["ln_f.weight"])
    assert torch.equal(c0["ln_f.weight"], matched_gaussian(base, edited, 0)["ln_f.weight"])


def test_low_rank_control_only_for_low_rank_edits() -> None:
    base = params()
    name = "blocks.0.mlp.proj.weight"
    low = {n: v.clone() for n, v in base.items()}
    u, v = torch.randn(CFG.d_model), torch.randn(4 * CFG.d_model)
    low[name][:, :40] += 0.1 * torch.outer(u, v)[:, :40]
    assert edit_ranks(base, low) == {name: 1}
    control = matched_low_rank(base, low, seed=0)
    assert control is not None
    delta = control[name] - base[name]
    assert edit_ranks(base, control) == {name: 1}
    assert float(delta[:, 40:].abs().sum()) == 0.0  # stays on the edit's columns
    want = (low[name].double() - base[name].double()).norm()
    assert abs(float(delta.double().norm() - want)) < 1e-4 * float(want)
    full = {n: v.clone() for n, v in base.items()}
    full[name] += 0.01 * torch.randn_like(full[name])
    assert matched_low_rank(base, full, seed=0) is None


def test_recovery_fraction() -> None:
    clean = np.array([10.0, 20.0, 30.0])
    damaged = clean + np.array([4.0, 2.0, 6.0])
    assert bootstrap_recovery(damaged, clean, clean, n_boot=200).estimate == 1.0
    assert bootstrap_recovery(damaged, damaged, clean, n_boot=200).estimate == 0.0
    half = bootstrap_recovery(damaged, (damaged + clean) / 2, clean, n_boot=200)
    assert abs(half.estimate - 0.5) < 1e-12 and abs(half.low - 0.5) < 1e-12
    worse = bootstrap_recovery(damaged, damaged + 1.0, clean, n_boot=200)
    assert worse.estimate < 0
