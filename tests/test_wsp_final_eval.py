"""The final evaluation's computation, run on synthetic data. The sealed file is never touched."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch

from tinygpt.config import load_config
from tinygpt.data.prepare import clean_story
from tinygpt.model import GPT
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp import protocol
from tinygpt.wsp.controls import matched_gaussian
from tinygpt.wsp.edits import load_base_params, write_edit
from tinygpt.wsp.evaluate import Decoding
from tinygpt.wsp.faults import apply_fault
from tinygpt.wsp.generation import DecodeConfig
from tinygpt.wsp.sandbox import SEALED_DIRS, guard_path

REPO = Path(__file__).resolve().parents[1]
CFG = load_config(REPO / "configs" / "tiny.toml").model
FINAL = REPO / "experiments" / "wsp" / "final_eval.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("final_eval", FINAL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stories(n: int, tag: str) -> str:
    """Stories separated by <|endoftext|>, each with a unique 8-word first line."""
    parts = [
        f"{tag}{i} alpha beta gamma delta epsilon zeta eta theta\nThe body of story {i}."
        for i in range(n)
    ]
    return "\n<|endoftext|>\n".join(parts) + "\n<|endoftext|>\n"


def _pin(path: Path) -> dict[str, str]:
    sha = hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()
    return {"checkpoint": str(path), "sha256": sha}  # absolute: repo / absolute is absolute


@pytest.fixture
def world(
    tmp_path: Path, tiny_tokenizer: BPETokenizer, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    fe = _module()
    # shrink the protocol so the tiny model can run it; the logic is unchanged
    monkeypatch.setattr(protocol, "SEALED_LOSS_STORIES", range(0, 4))
    monkeypatch.setattr(protocol, "SEALED_PROMPT_STORIES_FROM", 4)
    monkeypatch.setattr(protocol, "N_SEALED_PROMPTS", 3)
    monkeypatch.setattr(protocol, "MAX_NEW_TOKENS", 6)
    monkeypatch.setattr(
        fe,
        "decodings",
        lambda penalty=1.0: (
            Decoding(
                "sample",
                DecodeConfig(6, temperature=0.8, top_k=40, repetition_penalty=penalty),
                (0, 1),
            ),
            Decoding("greedy", DecodeConfig(6, temperature=0, repetition_penalty=penalty), None),
        ),
    )
    torch.manual_seed(0)
    base = tmp_path / "base"
    GPT(CFG).save_pretrained(base)
    tiny_tokenizer.save(base / "tokenizer.json")
    script = Path(__file__)
    clean = load_base_params(base)

    edited = {n: v.clone() for n, v in clean.items()}
    edited["blocks.0.attn.proj.weight"][:, :32] *= 0.5
    write_edit(base, edited, tmp_path / "edit", "edit", script)
    write_edit(base, matched_gaussian(clean, edited, 0), tmp_path / "ctrl", "ctrl", script)

    fault = {
        "family": "head_scale",
        "layer": 1,
        "head": 1,
        "tensor": "blocks.1.attn.proj.weight",
        "factor": 6.0,
    }
    damaged = apply_fault(clean, fault, CFG.n_head)
    write_edit(base, damaged, tmp_path / "damaged", "damaged", script)
    write_edit(tmp_path / "damaged", clean, tmp_path / "repaired", "repaired", script)
    noise = matched_gaussian(damaged, clean, 0)
    write_edit(tmp_path / "damaged", noise, tmp_path / "repair_ctrl", "c", script)
    specs = tmp_path / "specs.json"
    specs.write_text(json.dumps({"checkpoints": {"ckpt_1": fault}}))
    diagnosis = tmp_path / "diagnosis.json"
    diagnosis.write_text(json.dumps({"tensor": "blocks.1.attn.proj.weight", "head": 1}))

    manifest = {
        "exp3": {
            "unedited": _pin(base),
            "attempts": [
                {
                    "name": "b10",
                    "budget": 10,
                    "passes_used": 9,
                    "edited": True,
                    **_pin(tmp_path / "edit"),
                    "controls": [
                        {
                            "name": "b10_gauss_0",
                            "kind": "gauss",
                            "seed": 0,
                            **_pin(tmp_path / "ctrl"),
                        }
                    ],
                },
                {"name": "b25", "budget": 25, "passes_used": 3, "edited": False, "controls": []},
            ],
        },
        "exp2": {
            "clean": _pin(base),
            "specs": str(specs),
            "specs_sha256": hashlib.sha256(specs.read_bytes()).hexdigest(),
            "cases": [
                {
                    "name": "ckpt_1",
                    "passes_used": 12,
                    "damaged": _pin(tmp_path / "damaged"),
                    "repaired": _pin(tmp_path / "repaired"),
                    "diagnosis": str(diagnosis),
                    "controls": [
                        {"name": "ckpt_1_gauss_0", "kind": "gauss", "seed": 0}
                        | _pin(tmp_path / "repair_ctrl")
                    ],
                }
            ],
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    for name, text in (("sealed", _stories(12, "s")), ("dev", _stories(2, "d"))):
        (tmp_path / f"{name}.txt").write_text(text)
    (tmp_path / "train.txt").write_text(_stories(2, "t"))
    return {"fe": fe, "tmp": tmp_path, "manifest": manifest_path, "manifest_data": manifest}


def _run(world: dict[str, Any]) -> dict[str, Any]:
    tmp = world["tmp"]
    report: dict[str, Any] = world["fe"].run(
        tmp / "sealed.txt",
        world["manifest"],
        tmp / "results",
        dev_file=tmp / "dev.txt",
        train_file=tmp / "train.txt",
        tokenizer_path=tmp / "base" / "tokenizer.json",
    )
    return report


def test_exp3_report_has_criteria_controls_penalty_and_real_text(world: dict[str, Any]) -> None:
    report = _run(world)
    exp3 = report["exp3"]
    assert report["sealed"]["stories_scored"] == 4 and report["sealed"]["prompts"] == 3
    assert set(exp3["models"]) == {"unedited", "repetition_penalty", "b10"}
    b10 = exp3["attempts"]["b10"]
    assert set(b10["criteria"]) == {"95", "98.3"}
    assert set(b10["criteria"]["95"]) == {
        "loop_rate_falls",
        "rep4_falls",
        "loss_within_tolerance",
        "length_kept",
        "all",
    }
    assert list(b10["controls"]) == ["b10_gauss_0"] and "gauss" in b10["specific"]
    assert exp3["attempts"]["b25"]["criteria"]["98.3"]["all"] is False  # declared unedited
    assert "judge_loss" in exp3["models"]["b10"]["sample"]
    assert exp3["real_text"]["generations"] == 3
    # the penalty changes the decoder's own loss on real text, and never lowers it to below zero
    assert exp3["versus_unedited"]["repetition_penalty"]["sample"]["loss_diff"]["estimate"] != 0
    assert isinstance(exp3["headline_h3_met"], bool)
    tmp = world["tmp"]
    gens = json.loads((tmp / "results" / "final_generations.json").read_text())
    assert len(gens["unedited"]["sample"]) == 3 and len(gens["b10"]["greedy"]) == 3
    with np.load(tmp / "results" / "final_arrays.npz") as data:
        assert data["exp3/b10/sample/loop_rate"].shape == (3,)
        assert data["exp2/ckpt_1/repaired/story_nll"].shape == (4,)


def test_exp2_exact_repair_recovers_everything_and_is_located(world: dict[str, Any]) -> None:
    exp2 = _run(world)["exp2"]
    case = exp2["cases"]["ckpt_1"]
    assert case["tensor_correct"] is True and case["sublocation_correct"] is True
    assert abs(case["recovery"]["estimate"] - 1.0) < 1e-9
    assert abs(case["cosine_with_true_inverse"] - 1.0) < 1e-6
    assert case["controls"]["ckpt_1_gauss_0"]["recovery"] < 1.0
    assert case["beats_all_controls"] is True and case["repair_succeeded"] is True
    assert exp2["tensors_located"] == 1 and exp2["h2a_met"] is False  # 1 of 1 is not 3 of 5
    assert case["fault"]["family"] == "head_scale"


def test_too_good_results_are_flagged(world: dict[str, Any]) -> None:
    fe = world["fe"]
    report = _run(world)
    assert report["audit_triggers"] == []  # an exact undo of a scaling fault is legitimate
    report["exp2"]["cases"]["ckpt_1"]["fault"]["family"] = "noise"
    flagged = fe.audit_triggers(report)
    assert any("destroyed information" in f for f in flagged)
    assert any("almost exactly" in f for f in flagged)
    report["exp3"]["versus_unedited"]["b10"]["sample"]["loss_diff"]["high"] = -0.01
    assert any("test loss below" in f for f in fe.audit_triggers(report))


def test_a_checkpoint_that_differs_from_the_manifest_is_refused(world: dict[str, Any]) -> None:
    manifest = world["manifest_data"]
    manifest["exp3"]["attempts"][0]["sha256"] = "0" * 64
    world["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="does not match the manifest"):
        _run(world)


def test_changed_fault_specs_are_refused(world: dict[str, Any]) -> None:
    (world["tmp"] / "specs.json").write_text(json.dumps({"checkpoints": {}}))
    with pytest.raises(RuntimeError, match="fault specs"):
        _run(world)


def test_sealed_stories_duplicating_train_are_dropped(world: dict[str, Any]) -> None:
    fe = world["fe"]
    raw = [s for s in (world["tmp"] / "sealed.txt").read_text().split("\n<|endoftext|>\n") if s]
    known = {fe._digest(clean_story(raw[0]))}  # pretend story 0 is in train
    loss, prompts, prompt_stories, dropped = fe.select_sealed(raw, known)
    assert dropped["duplicate_of_dev_or_train"] == 1
    assert len(loss) == 3 and len(prompts) == len(prompt_stories) == 3
    assert all(
        story.startswith(prompt) for prompt, story in zip(prompts, prompt_stories, strict=True)
    )


def test_fetch_refuses_an_existing_file_with_the_wrong_hash(tmp_path: Path) -> None:
    fe = _module()
    f = tmp_path / "x.txt"
    f.write_text("not the sealed file")
    with pytest.raises(RuntimeError, match="pinned sha256"):
        fe.fetch_sealed(f, "unused", "0" * 64)


def test_sealed_file_location_is_guarded() -> None:
    assert any(str(d).endswith(str(Path("data") / "sealed")) for d in SEALED_DIRS)
    with pytest.raises(PermissionError):
        guard_path(REPO / "data" / "sealed" / "TinyStoriesV2-GPT4-valid.txt")


def test_figures_and_tables_are_built_from_result_files_only(world: dict[str, Any]) -> None:
    _run(world)
    results = world["tmp"] / "results"
    figures = world["tmp"] / "figures"
    exp1 = REPO / "experiments" / "wsp" / "results" / "exp1_verification.json"
    (results / "exp1_verification.json").write_text(exp1.read_text())
    spec = importlib.util.spec_from_file_location("make_figures", FINAL.parent / "make_figures.py")
    assert spec is not None and spec.loader is not None
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    assert mf.main(["--results", str(results), "--figures", str(figures)]) == 0
    names = {p.name for p in figures.glob("wsp_*.png")}
    assert names == {
        "wsp_exp1_heads.png",
        "wsp_exp2_recovery.png",
        "wsp_exp3_headline.png",
        "wsp_exp3_tradeoff.png",
    }
    tables = (results / "tables.md").read_text()
    for heading in ("## Exp 1", "## Exp 2", "## Exp 3", "Headline H3 met"):
        assert heading in tables
