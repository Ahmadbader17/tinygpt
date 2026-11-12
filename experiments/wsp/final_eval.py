"""The final evaluation: the only script that may read the sealed test data.

It runs once, after the editors are frozen (tag `wsp-freeze`). Before it reads anything it:
  1. refuses to run if a result file already exists (one run, no second look),
  2. refuses a dirty working tree or a HEAD that is not the `wsp-freeze` commit,
  3. verifies every checkpoint in experiments/wsp/manifest.json against its pinned sha256,
  4. downloads the sealed file if needed and verifies its sha256,
  5. appends an access record to logs/sealed_access.jsonl.

`run()` holds the computation and takes its paths as arguments, so tests can exercise it on
synthetic files. Only `main()` passes the sealed path.

    uv run python experiments/wsp/final_eval.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from tinygpt.data.download import open_stories
from tinygpt.data.prepare import clean_story, read_clean_split
from tinygpt.model import GPT
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp import protocol
from tinygpt.wsp.evaluate import (
    HEADLINE_ALPHA,
    Decoding,
    generation_metrics,
    loss_metrics,
    paired,
    run_model,
    specific,
    success_rule,
)
from tinygpt.wsp.generation import DecodeConfig

REPO = Path(__file__).resolve().parents[2]
SEALED_FILE = REPO / "data" / "sealed" / "TinyStoriesV2-GPT4-valid.txt"
ACCESS_LOG = REPO / "experiments" / "wsp" / "logs" / "sealed_access.jsonl"
MANIFEST = REPO / "experiments" / "wsp" / "manifest.json"
RESULTS = REPO / "experiments" / "wsp" / "results" / "final_eval.json"
GENERATIONS = REPO / "experiments" / "wsp" / "results" / "final_generations.json"
TOKENIZER = REPO / "runs" / "wsp" / "original" / "tokenizer.json"
DEV_FILE = REPO / "data" / "raw" / "val.txt"
TRAIN_FILE = REPO / "data" / "raw" / "train.txt"

DECODINGS = (
    Decoding(
        "sample",
        DecodeConfig(protocol.MAX_NEW_TOKENS, temperature=0.8, top_k=40),
        protocol.SAMPLING_SEEDS,
    ),
    Decoding("greedy", DecodeConfig(protocol.MAX_NEW_TOKENS, temperature=0), None),
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_sealed(raw: list[str], known: set[str]) -> tuple[list[str], list[str], dict[str, int]]:
    """Sealed loss stories and prompts; exact duplicates of dev or train stories are removed."""
    dropped = {"empty": 0, "duplicate_of_dev_or_train": 0}
    loss_stories: list[str] = []
    for i in protocol.SEALED_LOSS_STORIES:
        cleaned = clean_story(raw[i]) if i < len(raw) else ""
        if not cleaned:
            dropped["empty"] += 1
        elif _digest(cleaned) in known:
            dropped["duplicate_of_dev_or_train"] += 1
        else:
            loss_stories.append(cleaned)
    exclude = set(protocol.README_PROMPTS)
    prompts: list[str] = []
    for i in range(protocol.SEALED_PROMPT_STORIES_FROM, len(raw)):
        if len(prompts) == protocol.N_SEALED_PROMPTS:
            break
        cleaned = clean_story(raw[i])
        prompt = protocol.prompt_from_story(cleaned) if cleaned else None
        if prompt is None or prompt in exclude or _digest(cleaned) in known:
            continue
        exclude.add(prompt)
        prompts.append(prompt)
    if len(prompts) < protocol.N_SEALED_PROMPTS:
        raise RuntimeError(f"only {len(prompts)} sealed prompts available")
    return loss_stories, prompts, dropped


def run(
    sealed_file: Path,
    manifest_path: Path,
    results_path: Path,
    generations_path: Path,
    dev_file: Path = DEV_FILE,
    train_file: Path = TRAIN_FILE,
    tokenizer_path: Path = TOKENIZER,
) -> dict[str, Any]:
    tok = BPETokenizer.load(tokenizer_path)
    known = {_digest(s) for s in read_clean_split(dev_file)}
    known |= {_digest(s) for s in read_clean_split(train_file)}
    raw = list(open_stories(str(sealed_file), None))
    loss_stories, prompts, dropped = select_sealed(raw, known)
    summary: dict[str, Any] = {
        "sealed_stories_scored": len(loss_stories),
        "sealed_prompts": len(prompts),
        "dropped": dropped,
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loss_ids = [tok.encode(s) for s in loss_stories]
    prompt_ids = [tok.encode(p) for p in prompts]

    runs: dict[str, Any] = {}
    gens_out: dict[str, Any] = {}
    for entry in manifest["models"]:
        ckpt = REPO / entry["checkpoint"]
        sha = hashlib.sha256((ckpt / "model.safetensors").read_bytes()).hexdigest()
        if sha != entry["sha256"]:
            raise RuntimeError(f"{entry['name']}: checkpoint does not match the manifest")
        model = GPT.from_pretrained(ckpt).eval()
        decodings = DECODINGS if entry.get("decodings", "full") == "full" else ()
        with torch.inference_mode():
            runs[entry["name"]] = (
                entry,
                run_model(model, entry["name"], loss_ids, prompt_ids, tok.eot_id, decodings),
            )
        gens_out[entry["name"]] = {
            d: [
                {"prompt": prompts[p], "text": tok.decode(runs_p[0].tokens)}
                for p, runs_p in enumerate(g)
            ]
            for d, g in runs[entry["name"]][1].generations.items()
        }

    report: dict[str, Any] = {
        "summary": summary,
        "models": {},
        "versus_unedited": {},
        "criteria": {},
    }
    for name, (entry, r) in runs.items():
        report["models"][name] = {
            "role": entry["role"],
            "group": entry.get("group"),
            "loss": loss_metrics(r),
            "generation": {d: generation_metrics(g) for d, g in r.generations.items()},
        }
    _, base = runs["unedited"]
    for name, (entry, r) in runs.items():
        if name == "unedited" or "sample" not in r.generations:
            continue
        pair = {d: paired(r, base, d) for d in r.generations}
        report["versus_unedited"][name] = pair
        if entry["role"] == "edit":
            report["criteria"][name] = {
                "95": success_rule(pair["sample"]),
                "98.3": success_rule(paired(r, base, "sample", alpha=HEADLINE_ALPHA)),
            }

    controls: dict[str, list[float]] = {}
    for name, (entry, r) in runs.items():
        if entry["role"] == "control" and "sample" in r.generations:
            controls.setdefault(entry["group"], []).append(
                -report["versus_unedited"][name]["sample"]["loop_rate"]["estimate"]
            )
    for name, (entry, _) in runs.items():
        if entry["role"] == "edit" and entry["group"] in controls:
            edit_drop = -report["versus_unedited"][name]["sample"]["loop_rate"]["estimate"]
            report["criteria"][name]["specific"] = specific(edit_drop, controls[entry["group"]])

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    generations_path.write_text(json.dumps(gens_out, indent=1), encoding="utf-8")
    return report


def fetch_sealed(dest: Path, url: str, sha256: str) -> None:
    """Download once and verify. A partial or mismatched file is deleted, never kept."""
    if dest.exists():
        if hashlib.sha256(dest.read_bytes()).hexdigest() != sha256:
            raise RuntimeError(f"{dest} exists but does not match the pinned sha256")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:  # noqa: S310 (pinned https URL)
        h = hashlib.sha256()
        for chunk in iter(lambda: resp.read(1 << 20), b""):
            h.update(chunk)
            f.write(chunk)
    if h.hexdigest() != sha256:
        tmp.unlink()
        raise RuntimeError("downloaded file does not match the pinned sha256; deleted")
    tmp.rename(dest)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.parse_args()
    if RESULTS.exists():
        print("final_eval.json already exists: the final evaluation runs once.", file=sys.stderr)
        return 1
    if _git("status", "--porcelain"):
        print("working tree is not clean; commit or stash first.", file=sys.stderr)
        return 1
    if _git("rev-parse", "HEAD") != _git("rev-parse", "wsp-freeze^{commit}"):
        print("HEAD is not the wsp-freeze tag.", file=sys.stderr)
        return 1
    fetch_sealed(SEALED_FILE, protocol.SEALED_URL, protocol.SEALED_SHA256)
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ACCESS_LOG.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "time": datetime.now(UTC).isoformat(),
                    "commit": _git("rev-parse", "HEAD"),
                    "file": SEALED_FILE.name,
                    "sha256": protocol.SEALED_SHA256,
                    "purpose": "final evaluation",
                }
            )
            + "\n"
        )
    run(SEALED_FILE, MANIFEST, RESULTS, GENERATIONS)
    print(f"wrote {RESULTS.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
