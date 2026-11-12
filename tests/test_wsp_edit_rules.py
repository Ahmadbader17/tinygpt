"""Editor rules 1 and 2, enforced statically on every edit and analysis script.

Rule 1: no gradients, backward calls, optimizers or training loops.
Rule 2: no fitting weights to data (no least squares, no closed form regression).
Rule 5: edit code never names or reads sealed data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tinygpt.wsp.rules import scan_source

REPO = Path(__file__).resolve().parents[1]
RULED_DIRS = (
    REPO / "experiments" / "wsp" / "edits",
    REPO / "experiments" / "wsp" / "analysis",
    REPO / "experiments" / "wsp" / "logs",  # the editors' own scripts are archived here
)


def _ruled_files() -> list[Path]:
    files: list[Path] = []
    for folder in RULED_DIRS:
        if folder.exists():
            files.extend(sorted(folder.rglob("*.py")))
    return files


@pytest.mark.parametrize("path", _ruled_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_edit_and_analysis_scripts_follow_the_rules(path: Path) -> None:
    assert scan_source(path.read_text(encoding="utf-8")) == []


def test_the_scan_catches_each_forbidden_pattern() -> None:
    bad = {
        "import torch.optim as o": "torch.optim",
        "from torch import optim": "imports optim",
        "x = loss.backward()": ".backward",
        "w.requires_grad_(True)": "requires_grad_",
        "with torch.enable_grad(): pass": "enable_grad",
        "w = torch.linalg.lstsq(a, b)": "lstsq",
        "w = torch.linalg.pinv(a) @ b": "pinv",
        "from scipy.optimize import minimize": "scipy.optimize",
        "path = 'data/sealed/x.txt'": "sealed",
        "from tinygpt.model import GPT": "tinygpt.model",
        "m = GPT.from_pretrained(p)": "from_pretrained",
        "from tinygpt.wsp.generation import generate_all": "generate_all",
        "name = 'TinyStoriesV2-GPT4-valid.txt'": "TinyStoriesV2",
    }
    for source, marker in bad.items():
        assert scan_source(source), f"not caught: {source}"
        assert any(marker in p for p in scan_source(source)), source


def test_allowed_weight_editing_passes_the_scan() -> None:
    ok = (
        "import torch\n"
        "from tinygpt.wsp.edits import load_base_params\n"
        "w = load_base_params('runs/wsp/original')['wte.weight']\n"
        "with torch.no_grad():\n"
        "    w[:, :8] *= 0.5\n"
        "u, s, vh = torch.linalg.svd(w, full_matrices=False)\n"
    )
    assert scan_source(ok) == []
