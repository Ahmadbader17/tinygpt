"""Exp 1 verification: measure what each head does on dev data and score the predictions.

Refuses to run unless experiments/wsp/results/exp1_predictions.json is committed and
unchanged, and records that commit. The predictions were made from weights alone; this script
is the first time the model is run for Exp 1. Dev data only.

    uv run python experiments/wsp/exp1_verify.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tinygpt.data.prepare import read_clean_split
from tinygpt.model import GPT, Block
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp import protocol
from tinygpt.wsp.activations import forward_capture
from tinygpt.wsp.faults import head_columns
from tinygpt.wsp.headscore import (
    CLASSES,
    evaluate_predictions,
    ground_truth,
    offset_score,
    pattern_cosine,
    previous_token_score,
    start_anchor_score,
)
from tinygpt.wsp.scoring import story_nll

REPO = Path(__file__).resolve().parents[2]
ORIGINAL = REPO / "runs" / "wsp" / "original"
PREDICTIONS = REPO / "experiments" / "wsp" / "results" / "exp1_predictions.json"
OUT = REPO / "experiments" / "wsp" / "results" / "exp1_verification.json"

N_WINDOWS = 256  # natural-text windows of 256 tokens
STORY_STRIDE = 8  # window j starts at the <|endoftext|> before dev story 8j, j = 1..256
REP_ROWS, REP_PERIOD, REP_SEED = 64, 100, 0
ABLATION_STORIES = 500
BATCH = 32


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def committed_predictions() -> tuple[dict[str, object], str]:
    rel = str(PREDICTIONS.relative_to(REPO))
    if not PREDICTIONS.exists() or _git("status", "--porcelain", "--", rel):
        sys.exit("exp1_predictions.json must be committed and unchanged before verification.")
    commit = _git("log", "-1", "--format=%H", "--", rel)
    return json.loads(PREDICTIONS.read_text(encoding="utf-8")), commit


def natural_windows(ctx: int) -> torch.Tensor:
    """(N_WINDOWS, ctx) token ids from the dev token stream, each starting at a story boundary."""
    data = np.memmap(REPO / "data" / "processed" / "val.bin", dtype=np.uint16, mode="r")
    eot = np.flatnonzero(data == 0)  # story i ends at eot[i]; story i+1 starts right after
    rows = []
    for j in range(1, N_WINDOWS + 1):
        start = int(eot[STORY_STRIDE * j - 1])
        rows.append(np.asarray(data[start : start + ctx], dtype=np.int64))
    return torch.from_numpy(np.stack(rows))


def repeated_random(vocab: int) -> torch.Tensor:
    """<|endoftext|> then REP_PERIOD random tokens, twice. (REP_ROWS, 1 + 2 * REP_PERIOD)."""
    rng = np.random.default_rng(REP_SEED)
    block = rng.integers(1, vocab, size=(REP_ROWS, REP_PERIOD))
    rows = np.concatenate([np.zeros((REP_ROWS, 1), dtype=np.int64), block, block], axis=1)
    return torch.from_numpy(rows)


@torch.inference_mode()
def attention_scores(model: GPT) -> dict[str, np.ndarray]:
    n_layer, n_head = model.cfg.n_layer, model.cfg.n_head
    layers = tuple(range(n_layer))
    nat = natural_windows(model.cfg.context_length)
    prev = torch.zeros(n_layer, n_head, dtype=torch.float64)
    start = torch.zeros_like(prev)
    cos = torch.zeros_like(prev)
    pairs = 0
    last: dict[int, torch.Tensor] = {}
    for s in range(0, len(nat), BATCH):
        _, cap = forward_capture(model, nat[s : s + BATCH], ("attn",), layers)
        rows = cap["attn0"].shape[0]
        for layer in layers:
            probs = cap[f"attn{layer}"]  # (B, H, T, T)
            prev[layer] += previous_token_score(probs).double() * rows
            start[layer] += start_anchor_score(probs).double() * rows
            chain = probs if layer not in last else torch.cat([last[layer], probs])
            for i in range(len(chain) - 1):  # consecutive windows hold different text
                cos[layer] += pattern_cosine(chain[i], chain[i + 1])
            last[layer] = probs[-1:].clone()
        pairs += rows if s else rows - 1
    rep = repeated_random(model.cfg.vocab_size)
    induction = torch.zeros_like(prev)
    duplicate = torch.zeros_like(prev)
    for s in range(0, len(rep), BATCH):
        _, cap = forward_capture(model, rep[s : s + BATCH], ("attn",), layers)
        rows = cap["attn0"].shape[0]
        for layer in layers:
            probs = cap[f"attn{layer}"]
            first = 1 + REP_PERIOD  # first position of the second copy
            induction[layer] += offset_score(probs, first, REP_PERIOD - 1).double() * rows
            duplicate[layer] += offset_score(probs, first, REP_PERIOD).double() * rows
    return {
        "previous_token": (prev / len(nat)).flatten().numpy(),
        "start_anchor": (start / len(nat)).flatten().numpy(),
        "positional": (cos / pairs).flatten().numpy(),
        "induction": (induction / len(rep)).flatten().numpy(),
        "duplicate_token": (duplicate / len(rep)).flatten().numpy(),
    }


@torch.inference_mode()
def ablation_effects(model: GPT, tok: BPETokenizer) -> tuple[np.ndarray, float]:
    """Change in dev loss (nats per token) when each head's output is zeroed, one at a time."""
    stories = list(read_clean_split(REPO / "data" / "raw" / "val.txt"))
    ids = tok.encode_batch(protocol.dev_loss_stories(stories)[:ABLATION_STORIES])
    base_nll, count = story_nll(model, ids, tok.eot_id)
    tokens = float(count.sum())
    effects = []
    for layer in range(model.cfg.n_layer):
        block = model.blocks[layer]
        assert isinstance(block, Block)
        weight = block.attn.proj.weight
        for head in range(model.cfg.n_head):
            cols = head_columns(head, model.cfg.n_head, model.cfg.d_model)
            saved = weight[:, cols].clone()
            weight[:, cols] = 0.0
            nll, _ = story_nll(model, ids, tok.eot_id)
            weight[:, cols] = saved
            effects.append(float((nll.sum() - base_nll.sum()) / tokens))
            print(f"ablated {layer}.{head}", flush=True)
    return np.asarray(effects), float(base_nll.sum() / tokens)


def main() -> None:
    predictions, commit = committed_predictions()
    torch.set_num_threads(12)
    model = GPT.from_pretrained(ORIGINAL).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = BPETokenizer.load(ORIGINAL / "tokenizer.json")
    n_layer, n_head = model.cfg.n_layer, model.cfg.n_head
    started = time.perf_counter()
    scores = attention_scores(model)
    scores["inactive"], base_loss = ablation_effects(model, tok)

    heads = [f"{layer}.{head}" for layer in range(n_layer) for head in range(n_head)]
    layer_of = np.repeat(np.arange(n_layer), n_head)
    table = predictions["heads"]
    assert isinstance(table, dict)
    confidence = {c: np.asarray([float(table[h][c]["confidence"]) for h in heads]) for c in CLASSES}
    report = evaluate_predictions(confidence, scores, layer_of)
    truth = ground_truth(scores)
    result = {
        "predictions_commit": commit,
        "heads": heads,
        "scores": {c: scores[c].tolist() for c in CLASSES},
        "truth": {c: truth[c].tolist() for c in CLASSES},
        "confidence": {c: confidence[c].tolist() for c in CLASSES},
        "report": report,
        "probe": {
            "natural_windows": N_WINDOWS,
            "repeated_rows": REP_ROWS,
            "repeated_period": REP_PERIOD,
            "ablation_stories": ABLATION_STORIES,
            "ablation_base_loss": base_loss,
        },
        "harness_passes": {
            "attention_batches": N_WINDOWS // BATCH + REP_ROWS // BATCH,
            "ablation_evaluations": n_layer * n_head + 1,
        },
        "seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
