from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

from tinygpt.config import load_config
from tinygpt.model import GPT
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp.edits import (
    EditError,
    check_new_params,
    diff_summary,
    load_base_params,
    write_edit,
)

REPO = Path(__file__).resolve().parents[1]
CFG = load_config(REPO / "configs" / "tiny.toml").model
IDENTITY = REPO / "experiments" / "wsp" / "edits" / "identity.py"


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def base(tmp_path: Path, tiny_tokenizer: BPETokenizer) -> Path:
    torch.manual_seed(0)
    ckpt = tmp_path / "base"
    GPT(CFG).save_pretrained(ckpt)
    tiny_tokenizer.save(ckpt / "tokenizer.json")
    return ckpt


def test_identity_edit_changes_nothing(tmp_path: Path, base: Path) -> None:
    identity = _load_module(IDENTITY)
    summary = write_edit(base, identity.build(base), tmp_path / "out", "identity", IDENTITY)
    assert summary["weights_changed"] == 0
    assert summary["tensors_touched"] == 0
    assert summary["relative_l2"] == 0.0
    edited = GPT.from_pretrained(tmp_path / "out").eval()
    original = GPT.from_pretrained(base).eval()
    idx = torch.randint(0, CFG.vocab_size, (2, 16), generator=torch.Generator().manual_seed(3))
    with torch.inference_mode():
        assert torch.equal(edited(idx)[0], original(idx)[0])  # bit for bit, not just close


def test_identity_edit_rebuilds_to_the_same_bytes(tmp_path: Path, base: Path) -> None:
    identity = _load_module(IDENTITY)
    a = write_edit(base, identity.build(base), tmp_path / "a", "identity", IDENTITY)
    b = write_edit(base, identity.build(base), tmp_path / "b", "identity", IDENTITY)
    assert a["edited_sha256"] == b["edited_sha256"]


def test_scaled_head_touches_exactly_its_slice(tmp_path: Path, base: Path) -> None:
    new = load_base_params(base)
    head_dim = CFG.d_model // CFG.n_head
    # the output projection reads concatenated heads: head 0 owns input columns 0..head_dim-1
    w = new["blocks.0.attn.proj.weight"]
    w[:, :head_dim] *= 0.5
    summary = write_edit(base, new, tmp_path / "out", "head0_half", Path(__file__))
    assert summary["tensors_touched"] == 1
    assert summary["per_tensor"]["blocks.0.attn.proj.weight"]["changed"] == CFG.d_model * head_dim
    assert summary["weights_changed"] == CFG.d_model * head_dim


def test_invalid_edits_are_refused(base: Path) -> None:
    good = load_base_params(base)
    missing = {k: v for k, v in good.items() if k != "ln_f.weight"}
    with pytest.raises(EditError, match="names differ"):
        check_new_params(good, missing)
    wrong_shape = dict(good)
    wrong_shape["ln_f.weight"] = torch.zeros(3)
    with pytest.raises(EditError, match="shape"):
        check_new_params(good, wrong_shape)
    nan = dict(good)
    nan["ln_f.bias"] = torch.full_like(good["ln_f.bias"], float("nan"))
    with pytest.raises(EditError, match="NaN"):
        check_new_params(good, nan)


def test_diff_summary_fractions(base: Path) -> None:
    old = load_base_params(base)
    new = {k: v.clone() for k, v in old.items()}
    new["ln_f.weight"] = new["ln_f.weight"] * 2.0
    summary = diff_summary(old, new)
    entry = summary["per_tensor"]["ln_f.weight"]
    assert entry["changed"] == CFG.d_model
    assert abs(entry["relative_l2"] - 1.0) < 1e-6  # every entry doubled: change equals the norm
    assert summary["weights_changed"] == CFG.d_model
    assert summary["fraction_changed"] == CFG.d_model / summary["weights_total"]
