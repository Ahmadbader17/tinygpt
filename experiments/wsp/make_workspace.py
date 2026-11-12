"""Build an editor's workspace outside the repository, with its budget already recorded.

The workspace holds only what the editor may use: one checkpoint, the tokenizer, dev data,
reference source, the task text, and a ledger whose first record fixes the budget.

    uv run python experiments/wsp/make_workspace.py exp2 ckpt_1 --budget 20 \\
        --checkpoint runs/wsp/sealed/exp2/ckpt_1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from tinygpt.data.prepare import read_clean_split
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp import protocol
from tinygpt.wsp.sandbox import init_budget

REPO = Path(__file__).resolve().parents[2]
ROOT = Path.home() / "wsp_workspaces"
PROMPTS = REPO / "experiments" / "wsp" / "prompts"
REGISTRY = REPO / "runs" / "wsp" / "workspaces.json"
CHECKPOINT_FILES = ("model.safetensors", "model_config.json", "tokenizer.json")
HARNESS_SOURCE = ("sandbox.py", "activations.py", "edits.py", "generation.py", "metrics.py")


def task_text(kind: str, budget: int) -> str:
    task = (PROMPTS / f"{kind}_editor.md").read_text(encoding="utf-8")
    rules = (PROMPTS / "common_rules.md").read_text(encoding="utf-8")
    text = task.rstrip() + "\n\n" + rules
    return (
        text.replace("{BUDGET}", str(budget))
        .replace("{PYTHON}", str(REPO / ".venv" / "bin" / "python"))
        .replace("{REPO}", str(REPO))
    )


def build(kind: str, name: str, checkpoint: Path, budget: int) -> Path:
    ws = ROOT / f"{kind}_{name}"
    if ws.exists():
        raise SystemExit(f"{ws} already exists")
    (ws / "checkpoint").mkdir(parents=True)
    for file in CHECKPOINT_FILES:  # never the diff summary: it would describe the fault
        shutil.copyfile(checkpoint / file, ws / "checkpoint" / file)
    shutil.copyfile(checkpoint / "tokenizer.json", ws / "tokenizer.json")

    tok = BPETokenizer.load(ws / "tokenizer.json")
    stories = list(read_clean_split(REPO / "data" / "raw" / "val.txt"))
    (ws / "dev").mkdir()
    shutil.copyfile(REPO / "data" / "raw" / "val.txt", ws / "dev" / "stories.txt")
    prompts = [*protocol.dev_prompts(stories), *protocol.README_PROMPTS]
    (ws / "dev" / "prompts.json").write_text(
        json.dumps([{"text": p, "tokens": tok.encode(p)} for p in prompts], indent=1),
        encoding="utf-8",
    )
    shutil.copytree(
        REPO / "src" / "tinygpt" / "model",
        ws / "model_source",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (ws / "harness_source").mkdir()
    for file in HARNESS_SOURCE:
        shutil.copyfile(REPO / "src" / "tinygpt" / "wsp" / file, ws / "harness_source" / file)
    (ws / "TASK.md").write_text(task_text(kind, budget), encoding="utf-8")
    genesis = init_budget(ws / "work", budget)

    registry = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    registry[f"{kind}_{name}"] = {"workspace": str(ws), "budget": budget, "genesis": genesis}
    REGISTRY.write_text(json.dumps(registry, indent=1))
    return ws


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an editor workspace.")
    parser.add_argument("kind", choices=("exp2", "exp3"))
    parser.add_argument("name")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    args = parser.parse_args()
    print(build(args.kind, args.name, args.checkpoint, args.budget))
    return 0


if __name__ == "__main__":
    sys.exit(main())
