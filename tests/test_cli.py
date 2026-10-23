from pathlib import Path

import pytest
import torch

from tinygpt.cli import build_parser, main
from tinygpt.config import load_config
from tinygpt.inference import export_bundle
from tinygpt.model import GPT
from tinygpt.tokenizer import BPETokenizer

CFG = load_config(Path(__file__).parent.parent / "configs" / "tiny.toml")


def test_generate_needs_exactly_one_model_source() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["generate"])
    with pytest.raises(SystemExit):
        parser.parse_args(["generate", "--run", "a", "--hub", "b"])


def test_generate_from_hub_uses_the_downloaded_bundle(
    tmp_path: Path,
    tiny_tokenizer: BPETokenizer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import huggingface_hub

    torch.manual_seed(0)
    GPT(CFG.model).save_pretrained(tmp_path / "run" / "checkpoints" / "best")
    tiny_tokenizer.save(tmp_path / "tok.json")
    export_bundle(tmp_path / "run", tmp_path / "tok.json", tmp_path / "bundle", {})
    asked: list[str] = []

    def fake_download(repo_id: str) -> str:
        asked.append(repo_id)
        return str(tmp_path / "bundle")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    args = ["generate", "--hub", "me/tiny-gpt", "--prompt", "The fox", "--max-new-tokens", "5"]
    assert main([*args, "--seed", "1"]) == 0
    first = capsys.readouterr().out
    assert asked == ["me/tiny-gpt"]
    assert first.startswith("The fox")
    assert main([*args, "--seed", "1"]) == 0
    assert capsys.readouterr().out == first  # same seed, same text
