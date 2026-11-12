"""The final evaluation: the only script that may read the sealed test data.

It runs once, after the editors are frozen (tag `wsp-freeze`). Before it reads anything it:
  1. refuses to run if a result file already exists (one run, no second look),
  2. refuses a dirty working tree or a HEAD that is not the `wsp-freeze` commit,
  3. downloads the sealed file if needed and verifies its sha256,
  4. appends an access record to logs/sealed_access.jsonl.
Every checkpoint is verified against the sha256 pinned in experiments/wsp/manifest.json as it
is loaded. `run()` holds the computation and takes its paths as arguments, so tests exercise
it on synthetic files. Only `main()` passes the sealed path.

    uv run python experiments/wsp/final_eval.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tinygpt.data.download import open_stories
from tinygpt.data.prepare import clean_story, read_clean_split
from tinygpt.model import GPT
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp import protocol
from tinygpt.wsp.edits import load_base_params
from tinygpt.wsp.evaluate import (
    HEADLINE_ALPHA,
    REPETITION_PENALTY,
    Decoding,
    ModelRun,
    add_judge,
    generation_metrics,
    loss_metrics,
    paired,
    recovery,
    repair_cosine,
    run_model,
    specific,
    success_rule,
    text_metrics,
)
from tinygpt.wsp.faults import score_diagnosis
from tinygpt.wsp.generation import DecodeConfig
from tinygpt.wsp.metrics import PER_GENERATION_METRICS, per_prompt_stats
from tinygpt.wsp.scoring import story_nll

REPO = Path(__file__).resolve().parents[2]
SEALED_FILE = REPO / "data" / "sealed" / "TinyStoriesV2-GPT4-valid.txt"
ACCESS_LOG = REPO / "experiments" / "wsp" / "logs" / "sealed_access.jsonl"
MANIFEST = REPO / "experiments" / "wsp" / "manifest.json"
RESULTS_DIR = REPO / "experiments" / "wsp" / "results"
TOKENIZER = REPO / "runs" / "wsp" / "original" / "tokenizer.json"
DEV_FILE = REPO / "data" / "raw" / "val.txt"
TRAIN_FILE = REPO / "data" / "raw" / "train.txt"
INFORMATION_DESTROYING = ("head_zero", "noise")  # faults no repair can truly undo


def decodings(penalty: float = 1.0) -> tuple[Decoding, Decoding]:
    n = protocol.MAX_NEW_TOKENS
    return (
        Decoding(
            "sample",
            DecodeConfig(n, temperature=0.8, top_k=40, repetition_penalty=penalty),
            protocol.SAMPLING_SEEDS,
        ),
        Decoding("greedy", DecodeConfig(n, temperature=0, repetition_penalty=penalty), None),
    )


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_sealed(
    raw: list[str], known: set[str]
) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    """Sealed loss stories, prompts, and the stories the prompts open.

    Exact duplicates of dev or train stories are removed and counted.
    """
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
    prompt_stories: list[str] = []
    for i in range(protocol.SEALED_PROMPT_STORIES_FROM, len(raw)):
        if len(prompts) == protocol.N_SEALED_PROMPTS:
            break
        cleaned = clean_story(raw[i])
        prompt = protocol.prompt_from_story(cleaned) if cleaned else None
        if prompt is None or prompt in exclude or _digest(cleaned) in known:
            continue
        exclude.add(prompt)
        prompts.append(prompt)
        prompt_stories.append(cleaned)
    if len(prompts) < protocol.N_SEALED_PROMPTS:
        raise RuntimeError(f"only {len(prompts)} sealed prompts available")
    return loss_stories, prompts, prompt_stories, dropped


class Loader:
    """Loads checkpoints named in the manifest and refuses any whose hash has changed."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo

    def path(self, entry: dict[str, Any]) -> Path:
        path: Path = self.repo / entry["checkpoint"]
        sha = hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()
        if sha != entry["sha256"]:
            raise RuntimeError(f"{entry['checkpoint']}: checkpoint does not match the manifest")
        return path

    def model(self, entry: dict[str, Any]) -> GPT:
        model = GPT.from_pretrained(self.path(entry)).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model


def _describe(run: ModelRun) -> dict[str, Any]:
    out: dict[str, Any] = {"loss": loss_metrics(run)}
    for name in run.generations:
        out[name] = generation_metrics(run, name)
    return out


def _store(arrays: dict[str, Any], prefix: str, run: ModelRun) -> None:
    arrays[f"{prefix}/story_nll"] = run.story_nll
    arrays[f"{prefix}/story_tokens"] = run.story_tokens
    for name, gens in run.generations.items():
        stats = per_prompt_stats([[g.tokens for g in runs] for runs in gens])
        for metric in PER_GENERATION_METRICS:
            arrays[f"{prefix}/{name}/{metric}"] = stats[metric]


def _first_generations(run: ModelRun, prompts: list[str], tok: BPETokenizer) -> dict[str, Any]:
    """The first generation of every prompt (seed 0, or the greedy run): no picking."""
    return {
        name: [
            {"prompt": prompts[p], "text": tok.decode(runs[0].tokens), "ended": runs[0].hit_eot}
            for p, runs in enumerate(gens)
        ]
        for name, gens in run.generations.items()
    }


def evaluate_exp3(
    spec: dict[str, Any],
    loader: Loader,
    tok: BPETokenizer,
    loss_ids: list[list[int]],
    prompts: list[str],
    prompt_stories: list[str],
    arrays: dict[str, Any],
    generations: dict[str, Any],
) -> dict[str, Any]:
    prompt_ids = [tok.encode(p) for p in prompts]
    eot = tok.eot_id
    plain = decodings()
    unedited_model = loader.model(spec["unedited"])

    def full_run(model: GPT, name: str, penalty: float = 1.0) -> ModelRun:
        started = time.perf_counter()
        decs = decodings(penalty)
        run = run_model(model, name, loss_ids, prompt_ids, eot, decs, loss_penalty=penalty)
        add_judge(run, unedited_model, prompt_ids, eot, "sample")
        _store(arrays, f"exp3/{name}", run)
        generations[name] = _first_generations(run, prompts, tok)
        print(f"exp3 {name}: {time.perf_counter() - started:.0f}s", flush=True)
        return run

    base = full_run(unedited_model, "unedited")
    penalty = full_run(unedited_model, "repetition_penalty", REPETITION_PENALTY)

    real = []
    for prompt, story in zip(prompts, prompt_stories, strict=True):
        ids = tok.encode(story)
        real.append([ids[len(tok.encode(prompt)) :][: protocol.MAX_NEW_TOKENS]])
    report: dict[str, Any] = {
        "real_text": text_metrics(real),
        "models": {"unedited": _describe(base), "repetition_penalty": _describe(penalty)},
        "versus_unedited": {
            "repetition_penalty": {d.name: paired(penalty, base, d.name) for d in plain}
        },
        "attempts": {},
    }
    report["penalty_under_the_same_rule"] = success_rule(
        report["versus_unedited"]["repetition_penalty"]["sample"]
    )

    for attempt in spec["attempts"]:
        name = attempt["name"]
        info: dict[str, Any] = {
            "budget": attempt["budget"],
            "passes_used": attempt["passes_used"],
            "edited": attempt["edited"],
        }
        report["attempts"][name] = info
        if not attempt["edited"]:
            # the editor declared the unedited model final: no change, so no criterion can hold
            info["criteria"] = {"95": {"all": False}, "98.3": {"all": False}}
            continue
        run = full_run(loader.model(attempt), name)
        report["models"][name] = _describe(run)
        versus = {d.name: paired(run, base, d.name) for d in plain}
        report["versus_unedited"][name] = versus
        strict = paired(run, base, "sample", alpha=HEADLINE_ALPHA)
        info["criteria"] = {"95": success_rule(versus["sample"]), "98.3": success_rule(strict)}
        info["loop_rate_drop"] = -versus["sample"]["loop_rate"]["estimate"]
        info["controls"] = {}
        drops: dict[str, list[float]] = {}
        for control in attempt["controls"]:
            started = time.perf_counter()
            c_run = run_model(
                loader.model(control), control["name"], loss_ids, prompt_ids, eot, plain
            )
            _store(arrays, f"exp3/{control['name']}", c_run)
            c_versus = {d.name: paired(c_run, base, d.name) for d in plain}
            info["controls"][control["name"]] = {
                "kind": control["kind"],
                "seed": control["seed"],
                "loss": loss_metrics(c_run),
                "versus_unedited": c_versus,
            }
            drops.setdefault(control["kind"], []).append(
                -c_versus["sample"]["loop_rate"]["estimate"]
            )
            print(f"exp3 {control['name']}: {time.perf_counter() - started:.0f}s", flush=True)
        info["control_loop_rate_drops"] = drops
        info["specific"] = {
            kind: specific(info["loop_rate_drop"], values) for kind, values in drops.items()
        }
    report["headline_h3_met"] = any(
        a["criteria"]["98.3"]["all"] for a in report["attempts"].values()
    )
    return report


def evaluate_exp2(
    spec: dict[str, Any],
    loader: Loader,
    tok: BPETokenizer,
    loss_ids: list[list[int]],
    arrays: dict[str, Any],
) -> dict[str, Any]:
    specs_path = loader.repo / spec["specs"]
    if hashlib.sha256(specs_path.read_bytes()).hexdigest() != spec["specs_sha256"]:
        raise RuntimeError("the fault specs do not match the manifest")
    specs = json.loads(specs_path.read_text(encoding="utf-8"))["checkpoints"]
    eot = tok.eot_id

    def nll_of(entry: dict[str, Any], key: str) -> np.ndarray:
        nll, _ = story_nll(loader.model(entry), loss_ids, eot)
        arrays[f"exp2/{key}/story_nll"] = nll
        return nll

    clean = nll_of(spec["clean"], "clean")
    tokens = float(sum(len(s) + 1 for s in loss_ids))
    clean_params = load_base_params(loader.path(spec["clean"]))
    report: dict[str, Any] = {"clean_loss": float(clean.sum() / tokens), "cases": {}}
    for case in spec["cases"]:
        name = case["name"]
        started = time.perf_counter()
        fault = specs[name]
        damaged = nll_of(case["damaged"], f"{name}/damaged")
        info: dict[str, Any] = {
            "fault": {k: v for k, v in fault.items() if k not in ("indices", "sha256")},
            "passes_used": case["passes_used"],
            "damaged_loss": float(damaged.sum() / tokens),
            "damage": float((damaged.sum() - clean.sum()) / tokens),
        }
        diagnosis = (
            json.loads((loader.repo / case["diagnosis"]).read_text(encoding="utf-8"))
            if case["diagnosis"]
            else {}
        )
        info["diagnosis"] = diagnosis
        info.update(score_diagnosis(diagnosis, fault))
        if case["repaired"] is None:
            info.update(repaired=False, recovery=None, repair_succeeded=False)
        else:
            repaired = nll_of(case["repaired"], f"{name}/repaired")
            info["repaired"] = True
            info["repaired_loss"] = float(repaired.sum() / tokens)
            info["recovery"] = recovery(damaged, repaired, clean)
            info["controls"] = {}
            for control in case["controls"]:
                c_nll = nll_of(control, f"{name}/{control['name']}")
                info["controls"][control["name"]] = {
                    "kind": control["kind"],
                    "recovery": recovery(damaged, c_nll, clean)["estimate"],
                }
            gauss = [c["recovery"] for c in info["controls"].values() if c["kind"] == "gauss"]
            info["beats_all_controls"] = bool(gauss) and info["recovery"]["estimate"] > max(gauss)
            info["repair_succeeded"] = bool(
                info["recovery"]["low"] > 0 and info["beats_all_controls"]
            )
            info["cosine_with_true_inverse"] = repair_cosine(
                load_base_params(loader.path(case["damaged"])),
                load_base_params(loader.path(case["repaired"])),
                clean_params,
            )
        report["cases"][name] = info
        print(f"exp2 {name}: {time.perf_counter() - started:.0f}s", flush=True)
    cases = report["cases"].values()
    report["tensors_located"] = sum(c["tensor_correct"] for c in cases)
    report["repairs_succeeded"] = sum(c["repair_succeeded"] for c in cases)
    report["h2a_met"] = report["tensors_located"] >= 3
    report["h2b_met"] = report["repairs_succeeded"] >= 3
    done = [c["recovery"]["estimate"] for c in cases if c["recovery"] is not None]
    report["mean_recovery_of_attempted_repairs"] = float(np.mean(done)) if done else None
    report["mean_recovery_counting_no_repair_as_zero"] = float(
        np.mean([c["recovery"]["estimate"] if c["recovery"] else 0.0 for c in cases])
    )
    return report


def audit_triggers(report: dict[str, Any]) -> list[str]:
    """Results that PLAN.md section 9 says must be audited before they are reported."""
    found: list[str] = []
    for name, versus in report["exp3"]["versus_unedited"].items():
        if name == "repetition_penalty":
            continue
        sample = versus["sample"]
        if sample["loss_diff"]["high"] < 0:
            found.append(f"exp3 {name}: test loss below the unedited model's")
        base_rate = report["exp3"]["models"]["unedited"]["sample"]["loop_rate"]["estimate"]
        halved = -sample["loop_rate"]["estimate"] > 0.5 * base_rate
        if halved and sample["loss_diff"]["high"] <= 0.002:
            found.append(f"exp3 {name}: loop rate more than halved at no measurable loss cost")
    for name, case in report["exp2"]["cases"].items():
        if case["recovery"] is None:
            continue
        value = case["recovery"]["estimate"]
        destroying = case["fault"]["family"] in INFORMATION_DESTROYING
        if value > 1.0:
            found.append(f"exp2 {name}: recovery above 100%")
        if destroying and value > 0.9:
            found.append(f"exp2 {name}: recovery above 90% for a fault that destroyed information")
        cosine = case["cosine_with_true_inverse"]
        if destroying and cosine is not None and cosine > 0.99:
            found.append(f"exp2 {name}: repair restores the clean weights almost exactly")
    return found


def run(
    sealed_file: Path,
    manifest_path: Path,
    results_dir: Path,
    dev_file: Path = DEV_FILE,
    train_file: Path = TRAIN_FILE,
    tokenizer_path: Path = TOKENIZER,
    repo: Path = REPO,
) -> dict[str, Any]:
    tok = BPETokenizer.load(tokenizer_path)
    known = {_digest(s) for s in read_clean_split(dev_file)}
    known |= {_digest(s) for s in read_clean_split(train_file)}
    raw = list(open_stories(str(sealed_file), None))
    loss_stories, prompts, prompt_stories, dropped = select_sealed(raw, known)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loss_ids = tok.encode_batch(loss_stories)
    loader = Loader(repo)
    arrays: dict[str, Any] = {}
    generations: dict[str, Any] = {}
    report: dict[str, Any] = {
        "sealed": {
            "stories_in_file": len(raw),
            "stories_scored": len(loss_stories),
            "tokens_scored": int(sum(len(s) + 1 for s in loss_ids)),
            "prompts": len(prompts),
            "dropped": dropped,
        }
    }
    with torch.inference_mode():
        report["exp2"] = evaluate_exp2(manifest["exp2"], loader, tok, loss_ids, arrays)
        report["exp3"] = evaluate_exp3(
            manifest["exp3"], loader, tok, loss_ids, prompts, prompt_stories, arrays, generations
        )
    report["audit_triggers"] = audit_triggers(report)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "final_eval.json").write_text(
        json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8"
    )
    (results_dir / "final_generations.json").write_text(
        json.dumps(generations, indent=1), encoding="utf-8"
    )
    np.savez_compressed(str(results_dir / "final_arrays.npz"), **arrays)
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
    if (RESULTS_DIR / "final_eval.json").exists():
        print("final_eval.json already exists: the final evaluation runs once.", file=sys.stderr)
        return 1
    if _git("status", "--porcelain"):
        print("working tree is not clean; commit or stash first.", file=sys.stderr)
        return 1
    if _git("rev-parse", "HEAD") != _git("rev-parse", "wsp-freeze^{commit}"):
        print("HEAD is not the wsp-freeze tag.", file=sys.stderr)
        return 1
    torch.set_num_threads(12)
    fetch_sealed(SEALED_FILE, protocol.SEALED_URL, protocol.SEALED_SHA256)
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ACCESS_LOG.open("a", encoding="utf-8") as f:
        record = {
            "time": datetime.now(UTC).isoformat(),
            "commit": _git("rev-parse", "HEAD"),
            "file": SEALED_FILE.name,
            "sha256": protocol.SEALED_SHA256,
            "purpose": "final evaluation",
        }
        f.write(json.dumps(record) + "\n")
    started = time.perf_counter()
    report = run(SEALED_FILE, MANIFEST, RESULTS_DIR)
    print(f"done in {(time.perf_counter() - started) / 60:.0f} min")
    print(json.dumps({"audit_triggers": report["audit_triggers"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
