"""Phase 1 sanity checks on the real checkpoint, using dev data only.

Checks: the identity edit rebuilds the original bytes; the dev loss of the identity edit
equals the original's exactly; evaluation is deterministic; the budget refuses call N+1;
the ledger verifies. Writes experiments/wsp/logs/phase1/sanity.json.

    uv run python experiments/wsp/phase1_sanity.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from tinygpt.data.prepare import read_clean_split
from tinygpt.model import GPT
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp import protocol
from tinygpt.wsp.generation import DecodeConfig, generate_all
from tinygpt.wsp.sandbox import BudgetExceeded, Sandbox
from tinygpt.wsp.scoring import sequence_nll, story_nll

REPO = Path(__file__).resolve().parents[2]
ORIGINAL = REPO / "runs" / "wsp" / "original"
IDENTITY_SCRIPT = REPO / "experiments" / "wsp" / "edits" / "identity.py"
WORK = REPO / "runs" / "wsp" / "phase1_check"
OUT = REPO / "experiments" / "wsp" / "logs" / "phase1" / "sanity.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    torch.set_num_threads(12)
    checks: dict[str, object] = {}

    spec = importlib.util.spec_from_file_location("identity", IDENTITY_SCRIPT)
    assert spec is not None and spec.loader is not None
    identity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identity)
    identity.main()
    edited_dir = identity.OUT
    checks["identity_bytes_equal_original"] = sha(edited_dir / "model.safetensors") == sha(
        ORIGINAL / "model.safetensors"
    )

    tok = BPETokenizer.load(ORIGINAL / "tokenizer.json")
    stories = list(read_clean_split(REPO / "data" / "raw" / "val.txt"))
    loss_ids = tok.encode_batch(protocol.dev_loss_stories(stories)[:200])
    original = GPT.from_pretrained(ORIGINAL).eval()
    identical = GPT.from_pretrained(edited_dir).eval()
    a_nll, a_cnt = story_nll(original, loss_ids, tok.eot_id)
    b_nll, b_cnt = story_nll(identical, loss_ids, tok.eot_id)
    checks["identity_dev_loss_difference"] = float(np.abs(a_nll - b_nll).max())
    checks["dev_loss_repeat_identical"] = bool(
        np.array_equal(a_nll, story_nll(original, loss_ids, tok.eot_id)[0])
    )

    prompts = [tok.encode(p) for p in protocol.dev_prompts(stories)[:10]]
    cfg = DecodeConfig(protocol.MAX_NEW_TOKENS, temperature=0.8, top_k=40)
    g1 = generate_all(original, prompts, protocol.SAMPLING_SEEDS, cfg, tok.eot_id)
    g2 = generate_all(original, prompts, protocol.SAMPLING_SEEDS, cfg, tok.eot_id)
    checks["generation_repeat_identical"] = [[g.tokens for g in r] for r in g1] == [
        [g.tokens for g in r] for r in g2
    ]

    if WORK.exists():
        shutil.rmtree(WORK)
    sb = Sandbox(WORK, role="editor", limit=2)
    first = sb.read(ORIGINAL, prompts[:2], purpose="sanity: first of two")
    sb.read(ORIGINAL, prompts[:2], purpose="sanity: second of two")
    refused = False
    try:
        sb.read(ORIGINAL, prompts[:2], purpose="sanity: third, must be refused")
    except BudgetExceeded:
        refused = True
    checks["budget_refuses_call_n_plus_1"] = refused
    checks["budget_used"] = sb.used
    checks["ledger_last_status"] = sb.ledger.records()[-1]["status"]
    reopened = Sandbox(WORK, role="editor", limit=2)
    sb.ledger.verify()
    checks["ledger_verifies_after_reopen"] = reopened.used == 2
    direct_nll, _ = sequence_nll(original, prompts[:2])
    checks["sandbox_read_equals_direct_computation"] = bool(
        np.array_equal(np.asarray(first["nll"]), direct_nll)
    )

    rules = subprocess.run(
        ["uv", "run", "pytest", "-q", "tests/test_wsp_edit_rules.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    checks["static_rules_test_passed"] = rules.returncode == 0

    expected = {
        "identity_bytes_equal_original": True,
        "identity_dev_loss_difference": 0.0,
        "dev_loss_repeat_identical": True,
        "generation_repeat_identical": True,
        "budget_refuses_call_n_plus_1": True,
        "ledger_last_status": "refused",
        "ledger_verifies_after_reopen": True,
        "sandbox_read_equals_direct_computation": True,
        "static_rules_test_passed": True,
    }
    failures = {k: v for k, v in expected.items() if checks.get(k) != v}
    result = {"checks": checks, "failures": failures, "passed": not failures}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
