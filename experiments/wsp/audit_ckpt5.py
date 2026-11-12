"""Audit required by PLAN.md section 9: Exp 2 `ckpt_5` showed recovery above 100%.

Repeats the loss measurement on the sealed audit block (valid-file stories 20,000 to 21,999),
which no other script has read: clean, damaged and repaired `ckpt_5`, paired on the same
stories. Appends its own record to logs/sealed_access.jsonl. Written and run after the final
evaluation; it changes no edit and no result of that evaluation.

    uv run python experiments/wsp/audit_ckpt5.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from tinygpt.data.download import open_stories
from tinygpt.data.prepare import clean_story
from tinygpt.model import GPT
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp import protocol
from tinygpt.wsp.metrics import bootstrap_ratio_diff, bootstrap_recovery
from tinygpt.wsp.scoring import story_nll

REPO = Path(__file__).resolve().parents[2]
WSP = REPO / "experiments" / "wsp"
SEALED_FILE = REPO / "data" / "sealed" / "TinyStoriesV2-GPT4-valid.txt"
AUDIT_STORIES = range(20_000, 22_000)  # PLAN.md section 3, "sealed audit block"
CASE = "ckpt_5"


def main() -> None:
    if hashlib.sha256(SEALED_FILE.read_bytes()).hexdigest() != protocol.SEALED_SHA256:
        raise SystemExit("sealed file does not match the pinned sha256")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    with (WSP / "logs" / "sealed_access.jsonl").open("a", encoding="utf-8") as f:
        record = {
            "time": datetime.now(UTC).isoformat(),
            "commit": commit,
            "file": SEALED_FILE.name,
            "sha256": protocol.SEALED_SHA256,
            "purpose": f"section 9 audit of exp2 {CASE}: stories 20000-21999 only",
        }
        f.write(json.dumps(record) + "\n")

    manifest = json.loads((WSP / "manifest.json").read_text(encoding="utf-8"))["exp2"]
    case = next(c for c in manifest["cases"] if c["name"] == CASE)
    entries = {"clean": manifest["clean"], "damaged": case["damaged"], "repaired": case["repaired"]}
    tok = BPETokenizer.load(REPO / manifest["clean"]["checkpoint"] / "tokenizer.json")
    raw = list(open_stories(str(SEALED_FILE), None))
    stories = [s for s in (clean_story(raw[i]) for i in AUDIT_STORIES) if s]
    ids = tok.encode_batch(stories)
    torch.set_num_threads(12)
    nll: dict[str, np.ndarray] = {}
    tokens = np.zeros(0)
    with torch.inference_mode():
        for name, entry in entries.items():
            path = REPO / entry["checkpoint"]
            sha = hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()
            if sha != entry["sha256"]:
                raise SystemExit(f"{name}: checkpoint does not match the manifest")
            values, count = story_nll(GPT.from_pretrained(path).eval(), ids, tok.eot_id)
            nll[name], tokens = values, count.astype(np.float64)
    total = float(tokens.sum())
    out = {
        "case": CASE,
        "audit_block": [AUDIT_STORIES.start, AUDIT_STORIES.stop],
        "stories": len(ids),
        "tokens": int(total),
        "loss": {name: float(v.sum() / total) for name, v in nll.items()},
        "repaired_minus_clean": bootstrap_ratio_diff(
            nll["repaired"], nll["clean"], tokens
        ).as_dict(),
        "recovery": bootstrap_recovery(nll["damaged"], nll["repaired"], nll["clean"]).as_dict(),
        "share_of_stories_where_repaired_is_better": float((nll["repaired"] < nll["clean"]).mean()),
    }
    (WSP / "results" / "audit_ckpt5.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
