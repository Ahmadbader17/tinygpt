"""The final evaluation's computation, run on synthetic data. The sealed file is never touched."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import torch

from tinygpt.config import load_config
from tinygpt.model import GPT
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp import protocol
from tinygpt.wsp.evaluate import Decoding
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


def _sealed_text(n: int) -> str:
    """Stories separated by <|endoftext|>, each with a unique 8-word first line."""
    parts = []
    for i in range(n):
        parts.append(f"w{i} alpha beta gamma delta epsilon zeta eta theta\nThe body of story {i}.")
    return "\n<|endoftext|>\n".join(parts) + "\n<|endoftext|>\n"


@pytest.fixture
def world(tmp_path: Path, tiny_tokenizer: BPETokenizer, monkeypatch: pytest.MonkeyPatch) -> dict:
    fe = _module()
    # shrink the protocol so the tiny model can run it; the logic is unchanged
    monkeypatch.setattr(protocol, "SEALED_LOSS_STORIES", range(0, 4))
    monkeypatch.setattr(protocol, "SEALED_PROMPT_STORIES_FROM", 4)
    monkeypatch.setattr(protocol, "N_SEALED_PROMPTS", 3)
    monkeypatch.setattr(
        fe,
        "DECODINGS",
        (
            Decoding("sample", DecodeConfig(6, temperature=0.8, top_k=40), (0, 1)),
            Decoding("greedy", DecodeConfig(6, temperature=0), None),
        ),
    )
    torch.manual_seed(0)
    base = GPT(CFG)
    base.save_pretrained(tmp_path / "base")
    tiny_tokenizer.save(tmp_path / "base" / "tokenizer.json")
    edited = GPT.from_pretrained(tmp_path / "base")
    with torch.no_grad():
        edited.blocks[0].attn.proj.weight[:, :32] *= 0.5
    edited.save_pretrained(tmp_path / "edit")
    control = GPT.from_pretrained(tmp_path / "base")
    with torch.no_grad():
        control.blocks[0].attn.proj.weight.add_(
            0.01 * torch.randn_like(control.blocks[0].attn.proj.weight)
        )
    control.save_pretrained(tmp_path / "control")

    def pin(path: Path) -> str:
        import hashlib

        return hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()

    rel = lambda p: str(p)  # noqa: E731  (absolute paths: REPO / absolute is absolute)
    manifest = {
        "models": [
            {
                "name": "unedited",
                "role": "baseline",
                "checkpoint": rel(tmp_path / "base"),
                "sha256": pin(tmp_path / "base"),
            },
            {
                "name": "edit_b10",
                "role": "edit",
                "group": "exp3_b10",
                "checkpoint": rel(tmp_path / "edit"),
                "sha256": pin(tmp_path / "edit"),
            },
            {
                "name": "control_0",
                "role": "control",
                "group": "exp3_b10",
                "checkpoint": rel(tmp_path / "control"),
                "sha256": pin(tmp_path / "control"),
            },
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    sealed = tmp_path / "sealed.txt"
    sealed.write_text(_sealed_text(12))
    dev = tmp_path / "dev.txt"
    dev.write_text(_sealed_text(2))
    train = tmp_path / "train.txt"
    train.write_text(_sealed_text(2))
    return {
        "fe": fe,
        "tmp": tmp_path,
        "sealed": sealed,
        "dev": dev,
        "train": train,
        "manifest": manifest_path,
        "tokenizer": tmp_path / "base" / "tokenizer.json",
    }


def test_run_produces_criteria_controls_and_generations(world: dict) -> None:
    fe, tmp = world["fe"], world["tmp"]
    report = fe.run(
        world["sealed"],
        world["manifest"],
        tmp / "final_eval.json",
        tmp / "final_generations.json",
        dev_file=world["dev"],
        train_file=world["train"],
        tokenizer_path=world["tokenizer"],
    )
    assert report["summary"]["sealed_stories_scored"] == 4
    assert report["summary"]["sealed_prompts"] == 3
    assert set(report["models"]) == {"unedited", "edit_b10", "control_0"}
    assert "edit_b10" in report["criteria"]
    assert set(report["criteria"]["edit_b10"]) == {"95", "98.3", "specific"}
    assert report["criteria"]["edit_b10"]["98.3"]["all"] in (True, False)
    gens = json.loads((tmp / "final_generations.json").read_text())
    assert len(gens["unedited"]["sample"]) == 3  # one first generation per prompt
    assert json.loads((tmp / "final_eval.json").read_text())["summary"]["sealed_prompts"] == 3


def test_a_checkpoint_that_differs_from_the_manifest_is_refused(world: dict) -> None:
    fe, tmp = world["fe"], world["tmp"]
    manifest = json.loads(world["manifest"].read_text())
    manifest["models"][1]["sha256"] = "0" * 64
    world["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="does not match the manifest"):
        fe.run(
            world["sealed"],
            world["manifest"],
            tmp / "r.json",
            tmp / "g.json",
            dev_file=world["dev"],
            train_file=world["train"],
            tokenizer_path=world["tokenizer"],
        )


def test_sealed_stories_duplicating_train_are_dropped(world: dict) -> None:
    fe = world["fe"]
    raw = [s for s in world["sealed"].read_text().split("\n<|endoftext|>\n") if s]
    from tinygpt.data.prepare import clean_story

    known = {fe._digest(clean_story(raw[0]))}  # pretend story 0 is in train
    loss, prompts, dropped = fe.select_sealed(raw, known)
    assert dropped["duplicate_of_dev_or_train"] == 1
    assert len(loss) == 3


def test_fetch_refuses_an_existing_file_with_the_wrong_hash(tmp_path: Path) -> None:
    fe = _module()
    f = tmp_path / "x.txt"
    f.write_text("not the sealed file")
    with pytest.raises(RuntimeError, match="pinned sha256"):
        fe.fetch_sealed(f, "unused", "0" * 64)


def test_sealed_file_location_is_guarded() -> None:
    # the sealed file's location must be inside a directory the sandbox refuses
    assert any(str(d).endswith(str(Path("data") / "sealed")) for d in SEALED_DIRS)
    with pytest.raises(PermissionError):
        guard_path(REPO / "data" / "sealed" / "TinyStoriesV2-GPT4-valid.txt")
