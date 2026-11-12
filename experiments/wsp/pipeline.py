"""After an editor finishes: collect its work, rebuild its edit from the script alone, build
matched random controls, and write the manifest that the final evaluation is pinned to.

    uv run python experiments/wsp/pipeline.py collect exp2 ckpt_1
    uv run python experiments/wsp/pipeline.py rebuild exp2 ckpt_1
    uv run python experiments/wsp/pipeline.py controls exp2 ckpt_1
    uv run python experiments/wsp/pipeline.py manifest

`rebuild` runs the editor's script in a fresh folder that holds only the base checkpoint and
the editor's Python files. There is no dev data and there are no saved activations in that
folder, so an edit whose numbers depend on data cannot be rebuilt there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from tinygpt.wsp.controls import edit_ranks, matched_gaussian, matched_low_rank
from tinygpt.wsp.edits import load_base_params, write_edit
from tinygpt.wsp.rules import scan_source
from tinygpt.wsp.sandbox import Ledger

REPO = Path(__file__).resolve().parents[2]
WSP = REPO / "experiments" / "wsp"
RUNS = REPO / "runs" / "wsp"
ORIGINAL = RUNS / "original"
SEALED_EXP2 = RUNS / "sealed" / "exp2"
REGISTRY = RUNS / "workspaces.json"
MANIFEST = WSP / "manifest.json"
CHECKPOINT_FILES = ("model.safetensors", "model_config.json", "tokenizer.json")
CONTROL_SEEDS = (0, 1, 2, 3, 4)
EXP2_NAMES = tuple(f"ckpt_{k}" for k in range(1, 6))
EXP3_NAMES = ("b10", "b25", "b50")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def base_dir(kind: str, name: str) -> Path:
    return SEALED_EXP2 / name if kind == "exp2" else ORIGINAL


def logs_dir(kind: str, name: str) -> Path:
    return WSP / "logs" / kind / name


def edits_dir(kind: str, name: str) -> Path:
    return WSP / "edits" / f"{kind}_{name}"


def rebuild_dir(kind: str, name: str) -> Path:
    return RUNS / "rebuild" / f"{kind}_{name}"


def final_name(kind: str, name: str) -> str | None:
    """Exp 3: the version the editor declared final (None if it declared `unedited`)."""
    declared = (logs_dir(kind, name) / "final.txt").read_text(encoding="utf-8").split()
    if not declared or declared[0].lower() == "unedited":
        return None
    return declared[0].removesuffix(".py")


def output_dir(kind: str, name: str, root: Path) -> Path | None:
    if kind == "exp2":
        return root / "repaired"
    final = final_name(kind, name)
    return None if final is None else root / "edited" / final


def collect(kind: str, name: str) -> dict[str, Any]:
    entry = json.loads(REGISTRY.read_text())[f"{kind}_{name}"]
    ws = Path(entry["workspace"])
    logs, edits = logs_dir(kind, name), edits_dir(kind, name)
    for folder in (logs, edits):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    listing, scans = [], {}
    for path in sorted(p for p in ws.rglob("*") if p.is_file()):
        relative = path.relative_to(ws)
        listing.append({"path": str(relative), "bytes": path.stat().st_size})
        top = relative.parts[0]
        if "__pycache__" in relative.parts or top in ("model_source", "harness_source"):
            continue
        if relative.name == "tokenizer.json":
            continue
        if path.suffix == ".py":
            (edits / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, edits / relative)
            scans[str(relative)] = scan_source(path.read_text(encoding="utf-8"))
        elif top == "work":
            keep = relative.name == "ledger.jsonl" or path.suffix == ".json"
            if keep:
                (logs / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, logs / relative)
        elif top not in ("checkpoint", "dev", "repaired", "edited") and path.suffix in (
            ".md",
            ".json",
            ".txt",
        ):
            (logs / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, logs / relative)

    ledger = Ledger(ws / "work" / "ledger.jsonl")  # raises if the chain does not verify
    records = ledger.records()
    calls = [r for r in records if r["status"] == "ok"]
    out = output_dir(kind, name, ws)
    weights = None if out is None else out / "model.safetensors"
    audit = {
        "workspace": str(ws),
        "budget": entry["budget"],
        "passes_used": len(calls),
        "calls_refused": sum(r["status"] == "refused" for r in records),
        "within_budget": len(calls) <= entry["budget"],
        "ledger_verifies": True,
        "genesis_matches": records[0]["hash"] == entry["genesis"],
        "ledger_final_hash": records[-1]["hash"],
        "calls": [
            {
                "n": r["n"],
                "kind": r["kind"],
                "weights": r["weights_sha256"][:12],
                "purpose": r["purpose"],
            }
            for r in calls
        ],
        "weights_run": sorted({r["weights_sha256"] for r in calls}),
        "static_scan": scans,
        "static_scan_clean": not any(scans.values()),
        "workspace_checkpoint_sha256": sha256(weights) if weights and weights.exists() else None,
        "files": listing,
    }
    (logs / "audit.json").write_text(json.dumps(audit, indent=1) + "\n", encoding="utf-8")
    brief = {k: v for k, v in audit.items() if k not in ("files", "calls", "static_scan")}
    print(json.dumps(brief, indent=1))
    return audit


def rebuild(kind: str, name: str) -> dict[str, Any]:
    """Run the editor's script in a clean folder; the result is the checkpoint that counts."""
    root = rebuild_dir(kind, name)
    if root.exists():
        shutil.rmtree(root)
    (root / "checkpoint").mkdir(parents=True)
    base = base_dir(kind, name)
    for file in CHECKPOINT_FILES:
        shutil.copyfile(base / file, root / "checkpoint" / file)
    shutil.copyfile(base / "tokenizer.json", root / "tokenizer.json")
    shutil.copytree(edits_dir(kind, name), root, dirs_exist_ok=True)
    result: dict[str, Any] = {"rebuilt": False, "sha256": None, "matches_workspace": None}
    if kind == "exp2":
        script = root / "repair.py"
    else:
        final = final_name(kind, name)
        if final is None:
            result["declared_unedited"] = True
            _save_rebuild(kind, name, result)
            return result
        script = root / "edits" / f"{final}.py"
    if not script.exists():
        result["error"] = f"no script at {script.relative_to(root)}"
    else:
        run = subprocess.run(
            [str(REPO / ".venv" / "bin" / "python"), str(script.relative_to(root))],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        out = output_dir(kind, name, root)
        assert out is not None
        if run.returncode != 0 or not (out / "model.safetensors").exists():
            result["error"] = (run.stderr or run.stdout)[-2000:]
        else:
            audit = json.loads((logs_dir(kind, name) / "audit.json").read_text())
            result.update(
                rebuilt=True,
                checkpoint=rel(out),
                sha256=sha256(out / "model.safetensors"),
                diff_summary=json.loads((out / "diff_summary.json").read_text()),
            )
            result["matches_workspace"] = result["sha256"] == audit["workspace_checkpoint_sha256"]
    _save_rebuild(kind, name, result)
    return result


def _save_rebuild(kind: str, name: str, result: dict[str, Any]) -> None:
    path = logs_dir(kind, name) / "rebuild.json"
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    brief = {k: v for k, v in result.items() if k != "diff_summary"}
    if "diff_summary" in result:
        d = result["diff_summary"]
        brief["tensors_touched"] = d["tensors_touched"]
        brief["weights_changed"] = d["weights_changed"]
        brief["relative_l2"] = d["relative_l2"]
    print(json.dumps(brief, indent=1))


def controls(kind: str, name: str) -> list[dict[str, Any]]:
    """Random edits with the same support and norm as the rebuilt edit, 5 seeds."""
    info = json.loads((logs_dir(kind, name) / "rebuild.json").read_text())
    made: list[dict[str, Any]] = []
    if info.get("rebuilt"):
        base_path = base_dir(kind, name)
        base = load_base_params(base_path)
        edited = load_base_params(REPO / info["checkpoint"])
        if any(not bool((base[n] == edited[n]).all()) for n in base):
            root = RUNS / "controls" / f"{kind}_{name}"
            if root.exists():
                shutil.rmtree(root)
            for seed in CONTROL_SEEDS:
                variants = {"gauss": matched_gaussian(base, edited, seed)}
                low = matched_low_rank(base, edited, seed)
                if low is not None:
                    variants["lowrank"] = low
                for label, params in variants.items():
                    tag = f"{label}_{seed}"
                    summary = write_edit(base_path, params, root / tag, tag, Path(__file__))
                    made.append(
                        {
                            "name": f"{name}_{tag}",
                            "kind": label,
                            "seed": seed,
                            "checkpoint": rel(root / tag),
                            "sha256": summary["edited_sha256"],
                            "relative_l2": summary["relative_l2"],
                        }
                    )
            ranks = edit_ranks(base, edited)
            print(f"{kind} {name}: {len(made)} controls; edit ranks {ranks}")
    (logs_dir(kind, name) / "controls.json").write_text(
        json.dumps(made, indent=1) + "\n", encoding="utf-8"
    )
    return made


def _pinned(path: Path) -> dict[str, str]:
    return {"checkpoint": rel(path), "sha256": sha256(path / "model.safetensors")}


def manifest() -> dict[str, Any]:
    """Everything the final evaluation will load, pinned by sha256."""

    def load(kind: str, name: str, file: str) -> Any:
        return json.loads((logs_dir(kind, name) / file).read_text())

    exp3: dict[str, Any] = {"unedited": _pinned(ORIGINAL), "attempts": []}
    for name in EXP3_NAMES:
        info = load("exp3", name, "rebuild.json")
        audit = load("exp3", name, "audit.json")
        attempt: dict[str, Any] = {
            "name": name,
            "budget": audit["budget"],
            "passes_used": audit["passes_used"],
            "edited": info.get("rebuilt", False),
            "controls": load("exp3", name, "controls.json"),
        }
        if attempt["edited"]:
            attempt.update(_pinned(REPO / info["checkpoint"]))
        exp3["attempts"].append(attempt)
    exp2: dict[str, Any] = {
        "clean": _pinned(ORIGINAL),
        "specs": rel(SEALED_EXP2 / "specs.json"),
        "specs_sha256": sha256(SEALED_EXP2 / "specs.json"),
        "cases": [],
    }
    for name in EXP2_NAMES:
        info = load("exp2", name, "rebuild.json")
        audit = load("exp2", name, "audit.json")
        diagnosis = logs_dir("exp2", name) / "diagnosis.json"
        case: dict[str, Any] = {
            "name": name,
            "passes_used": audit["passes_used"],
            "damaged": _pinned(SEALED_EXP2 / name),
            "repaired": _pinned(REPO / info["checkpoint"]) if info.get("rebuilt") else None,
            "diagnosis": rel(diagnosis) if diagnosis.exists() else None,
            "controls": load("exp2", name, "controls.json"),
        }
        exp2["cases"].append(case)
    data = {"exp3": exp3, "exp2": exp2}
    MANIFEST.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"manifest: {len(exp3['attempts'])} attempts, {len(exp2['cases'])} repair cases")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("action", choices=("collect", "rebuild", "controls", "manifest"))
    parser.add_argument("kind", nargs="?", choices=("exp2", "exp3"))
    parser.add_argument("name", nargs="?")
    args = parser.parse_args()
    if args.action == "manifest":
        manifest()
    else:
        {"collect": collect, "rebuild": rebuild, "controls": controls}[args.action](
            args.kind, args.name
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
