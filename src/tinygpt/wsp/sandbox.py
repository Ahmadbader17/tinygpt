"""The budgeted sandbox: the only route by which an editor runs the model.

Three things are enforced here, not by convention:

1. **Budget.** Every call to `read` or `generate` appends a line to a hash-chained ledger
   before it runs. The count comes from the ledger, so a new process cannot reset it. A call
   beyond the limit is logged as refused and raises `BudgetExceeded`. For an editor the
   limit is the ledger's first record, written by the harness (`init_budget`), so the editor's
   own code cannot choose it.
2. **Sealed data.** Paths under the sealed directories, and any file whose name contains the
   official validation file's name, raise `SealedAccessError`.
3. **No gradients.** Models are loaded with `requires_grad=False` and every call runs under
   `torch.inference_mode()`.

Limits of this design are written down in experiments/wsp/logs/phase1/journal.md: a process
with shell access can still rewrite a ledger, so the harness re-verifies the chain and records
its final hash after each attempt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from tinygpt.model import GPT
from tinygpt.model.gpt import WEIGHTS_FILE
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp.activations import CAPTURE_KINDS, forward_capture
from tinygpt.wsp.generation import DecodeConfig, generate_all
from tinygpt.wsp.metrics import has_loop, repeated_ngram_fraction
from tinygpt.wsp.scoring import sequence_nll

REPO_ROOT = Path(__file__).resolve().parents[3]
SEALED_DIRS = (REPO_ROOT / "data" / "sealed", REPO_ROOT / "runs" / "wsp" / "sealed")
SEALED_NAME_MARKER = "GPT4-valid"  # the full validation file; dev data is data/raw/val.txt

MAX_ROWS_PER_CALL = 64  # sequences for read, or prompt-and-seed continuations for generate
MAX_READ_LENGTH = 256  # the context length of the model
MAX_NEW_TOKENS = 200

Kind = Literal["read", "generate"]


class BudgetExceeded(RuntimeError):
    """The role has spent every forward pass it was given."""


class SealedAccessError(PermissionError):
    """A path that holds or names sealed evaluation data was requested."""


class LedgerError(RuntimeError):
    """The ledger's hash chain does not verify."""


def guard_path(path: Path | str) -> Path:
    """Return the resolved path, or raise if it is sealed."""
    resolved = Path(path).resolve()
    for sealed in SEALED_DIRS:
        if resolved == sealed or sealed in resolved.parents:
            raise SealedAccessError(f"{path} is inside a sealed directory")
    if SEALED_NAME_MARKER in resolved.name:
        raise SealedAccessError(f"{path} names the sealed validation file")
    return resolved


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _digest(record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class Ledger:
    """Append-only JSON lines; each record stores the hash of the one before it."""

    path: Path
    _records: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.path.exists():
            self._records = [json.loads(line) for line in self.path.read_text().splitlines()]
            self.verify()

    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def append(self, entry: dict[str, Any]) -> dict[str, Any]:
        prev = self._records[-1]["hash"] if self._records else "0" * 64
        record = {**entry, "n": len(self._records), "time": _now(), "prev": prev}
        record["hash"] = _digest(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        self._records.append(record)
        return record

    def verify(self) -> None:
        prev = "0" * 64
        for i, record in enumerate(self._records):
            if record.get("n") != i or record.get("prev") != prev:
                raise LedgerError(f"record {i} is out of sequence")
            if record.get("hash") != _digest(record):
                raise LedgerError(f"record {i} does not match its hash")
            prev = record["hash"]

    def count(self, role: str) -> int:
        """Forward passes this role has spent (refused calls are not counted)."""
        return sum(1 for r in self._records if r["role"] == role and r["status"] == "ok")

    def config(self) -> dict[str, Any] | None:
        """The budget record the harness wrote first, if this ledger has one."""
        if self._records and self._records[0]["status"] == "config":
            return self._records[0]
        return None


EDITOR_ROLE = "editor"


def init_budget(workdir: Path, limit: int) -> str:
    """Harness only: start a workspace ledger whose first record fixes the editor's budget.

    Returns the record's hash; the harness keeps it to check later that the ledger an editor
    hands back still starts with the same record.
    """
    ledger = Ledger(guard_path(workdir) / "ledger.jsonl")
    if ledger.records():
        raise LedgerError("the ledger already has records; a budget is set once, on an empty one")
    record = ledger.append(
        {
            "role": "harness",
            "kind": "config",
            "status": "config",
            "limit": limit,
            "purpose": "forward-pass budget for the editor of this workspace",
        }
    )
    return str(record["hash"])


class Sandbox:
    """Budgeted, logged access to checkpoints.

    `limit=None` is the harness role: logged, but unlimited.
    """

    def __init__(
        self,
        workdir: Path,
        role: str,
        limit: int | None,
        tokenizer_path: Path = REPO_ROOT / "data" / "tokenizer" / "tokenizer.json",
    ) -> None:
        self.workdir = guard_path(workdir)
        self.outputs = self.workdir / "outputs"
        self.role = role
        self.limit = limit
        self.ledger = Ledger(self.workdir / "ledger.jsonl")
        config = self.ledger.config()
        if config is not None:  # a harness-issued budget overrides whatever the caller passed
            self.role = EDITOR_ROLE
            self.limit = int(config["limit"])
        self.tokenizer = BPETokenizer.load(guard_path(tokenizer_path))
        self._models: dict[str, GPT] = {}

    @classmethod
    def open(cls, workdir: Path, tokenizer_path: Path | None = None) -> Sandbox:
        """Open an editor workspace's sandbox; the budget comes from the ledger's first record."""
        workdir = Path(workdir)
        if Ledger(workdir / "ledger.jsonl").config() is None:
            raise LedgerError(f"{workdir} has no budget record; the harness creates workspaces")
        tok = tokenizer_path if tokenizer_path is not None else workdir.parent / "tokenizer.json"
        return cls(workdir, EDITOR_ROLE, None, tok)

    @property
    def used(self) -> int:
        return self.ledger.count(self.role)

    @property
    def remaining(self) -> int | None:
        return None if self.limit is None else self.limit - self.used

    def _load(self, checkpoint: Path) -> tuple[GPT, str]:
        weights = guard_path(checkpoint) / WEIGHTS_FILE
        sha = hashlib.sha256(weights.read_bytes()).hexdigest()
        if sha not in self._models:
            model = GPT.from_pretrained(checkpoint).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self._models = {sha: model}  # keep one model in memory at a time
        return self._models[sha], sha

    def _spend(self, kind: Kind, purpose: str, sha: str, detail: dict[str, Any]) -> int:
        """Record the call before it runs. Returns its ledger number."""
        if not purpose.strip():
            raise ValueError("every call needs a purpose: the hypothesis it tests")
        entry = {
            "role": self.role,
            "kind": kind,
            "purpose": purpose,
            "weights_sha256": sha,
            "detail": detail,
        }
        if self.limit is not None and self.used >= self.limit:
            self.ledger.append({**entry, "status": "refused"})
            raise BudgetExceeded(f"budget of {self.limit} passes is spent; call refused")
        return int(self.ledger.append({**entry, "status": "ok"})["n"])

    @torch.inference_mode()
    def read(
        self,
        checkpoint: Path,
        sequences: list[list[int]],
        *,
        purpose: str,
        capture: tuple[str, ...] = (),
        layers: tuple[int, ...] = (),
        attention_layers: tuple[int, ...] = (),
    ) -> dict[str, Any]:
        """Loss of each sequence, and optionally activations saved to a file.

        `capture` names kinds from `activations.CAPTURE_KINDS`, taken at `layers`; it may also
        hold "logits" and "token_loss" (per-token loss, shape (rows, T-1)). `attention_layers`
        is shorthand for capture=("attn",) at those layers. Capturing needs equal-length rows.
        """
        if not 1 <= len(sequences) <= MAX_ROWS_PER_CALL:
            raise ValueError(f"read takes 1 to {MAX_ROWS_PER_CALL} sequences")
        vocab = self.tokenizer.vocab_size
        for seq in sequences:
            if not 2 <= len(seq) <= MAX_READ_LENGTH:
                raise ValueError(f"each sequence needs 2 to {MAX_READ_LENGTH} tokens")
            if min(seq) < 0 or max(seq) >= vocab:
                raise ValueError("token id outside the vocabulary")
        extra = {"logits", "token_loss"}
        kinds = set(capture) - extra
        unknown = kinds - set(CAPTURE_KINDS)
        if unknown:
            raise ValueError(f"unknown capture kinds {sorted(unknown)}")
        if kinds and not layers:
            raise ValueError("name the layers to capture at")
        capturing = bool(capture) or bool(attention_layers)
        if capturing and len({len(s) for s in sequences}) != 1:
            raise ValueError("capturing needs sequences of equal length")
        model, sha = self._load(checkpoint)
        n = self._spend(
            "read",
            purpose,
            sha,
            {
                "rows": len(sequences),
                "capture": sorted(capture),
                "layers": list(layers),
                "attention_layers": list(attention_layers),
            },
        )
        nll, count = sequence_nll(model, sequences)
        result: dict[str, Any] = {
            "call": n,
            "weights_sha256": sha,
            "nll": nll.tolist(),
            "tokens": count.tolist(),
            "mean_nll": float(nll.sum() / count.sum()),
        }
        if capturing:
            idx = torch.tensor(sequences, dtype=torch.long)
            logits, captured = forward_capture(model, idx, kinds, layers)
            arrays: dict[str, Any] = {key: value.numpy() for key, value in captured.items()}
            if attention_layers:
                _, attn = forward_capture(model, idx, ("attn",), attention_layers)
                arrays.update({key: value.numpy() for key, value in attn.items()})
            if "logits" in capture:
                arrays["logits"] = logits.numpy()
            if "token_loss" in capture:
                arrays["token_loss"] = (
                    F.cross_entropy(logits[:, :-1].transpose(1, 2), idx[:, 1:], reduction="none")
                ).numpy()  # (rows, T-1): loss of token t+1 given tokens up to t
            path = self.outputs / f"read_call{n:03d}.npz"
            self.outputs.mkdir(parents=True, exist_ok=True)
            np.savez(str(path), **arrays)
            result["capture_file"] = str(path)
            result["captured"] = sorted(arrays)
        return result

    @torch.inference_mode()
    def generate(
        self,
        checkpoint: Path,
        prompts: list[list[int]],
        *,
        seeds: tuple[int, ...] | None,
        decode: DecodeConfig,
        purpose: str,
    ) -> dict[str, Any]:
        """Sampled (seeds given) or greedy (seeds=None) continuations, with repetition metrics.

        Full token ids and text go to a file in outputs/; the response holds the summary.
        """
        rows = len(prompts) * (1 if seeds is None else len(seeds))
        if not 1 <= rows <= MAX_ROWS_PER_CALL:
            raise ValueError(f"generate takes 1 to {MAX_ROWS_PER_CALL} continuations per call")
        if not 1 <= decode.max_new_tokens <= MAX_NEW_TOKENS:
            raise ValueError(f"max_new_tokens must be 1 to {MAX_NEW_TOKENS}")
        for prompt in prompts:
            if not prompt or len(prompt) + decode.max_new_tokens > MAX_READ_LENGTH:
                raise ValueError("prompt plus max_new_tokens must fit in the context")
        model, sha = self._load(checkpoint)
        n = self._spend(
            "generate",
            purpose,
            sha,
            {
                "rows": rows,
                "decode": decode.__dict__,
                "seeds": None if seeds is None else list(seeds),
            },
        )
        gens = generate_all(model, prompts, seeds, decode, self.tokenizer.eot_id)
        flat = [(p, s, g) for p, runs in enumerate(gens) for s, g in enumerate(runs)]
        rows_out: list[dict[str, Any]] = [
            {
                "prompt": p,
                "seed": s,
                "tokens": g.tokens,
                "text": self.tokenizer.decode(g.tokens),
                "hit_eot": g.hit_eot,
                "loop": has_loop(g.tokens),
                "rep4": repeated_ngram_fraction(g.tokens, 4),
            }
            for p, s, g in flat
        ]
        self.outputs.mkdir(parents=True, exist_ok=True)
        path = self.outputs / f"generations_call{n:03d}.json"
        path.write_text(json.dumps(rows_out, indent=1), encoding="utf-8")
        return {
            "call": n,
            "weights_sha256": sha,
            "generations_file": str(path),
            "loop_rate": float(np.mean([r["loop"] for r in rows_out])),
            "rep4": float(np.mean([r["rep4"] for r in rows_out])),
            "mean_length": float(np.mean([len(r["tokens"]) for r in rows_out])),
            "eot_rate": float(np.mean([r["hit_eot"] for r in rows_out])),
        }
