"""Turning a set of new weight tensors into a saved checkpoint plus a diff summary.

An edit is a script that returns new values for the named parameters (see
experiments/wsp/edits/identity.py). `write_edit` checks them, saves a checkpoint, reloads it to
confirm the round trip, and writes `diff_summary.json` next to it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from tinygpt.model import GPT
from tinygpt.model.gpt import WEIGHTS_FILE

TOKENIZER_FILE = "tokenizer.json"
SUMMARY_FILE = "diff_summary.json"


class EditError(ValueError):
    """The edit returned tensors that cannot be a valid checkpoint."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_base_params(base_dir: Path) -> dict[str, Tensor]:
    """Parameters of a checkpoint, detached and cloned, keyed by name (tied weights once)."""
    model = GPT.from_pretrained(base_dir)
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def diff_summary(base: dict[str, Tensor], new: dict[str, Tensor]) -> dict[str, Any]:
    """What an edit touched: per tensor counts, fractions and relative L2 change, plus totals."""
    if base.keys() != new.keys():
        raise EditError("edit must return exactly the checkpoint's parameter names")
    per_tensor: dict[str, Any] = {}
    touched = changed_total = weights_total = 0
    sq_change = sq_base = 0.0
    for name in sorted(base):
        old, cur = base[name], new[name]
        if old.shape != cur.shape:
            raise EditError(f"{name}: shape {tuple(cur.shape)} != {tuple(old.shape)}")
        changed = int((old != cur).sum())
        delta = (cur.double() - old.double()).norm().item()
        norm = old.double().norm().item()
        per_tensor[name] = {
            "shape": list(old.shape),
            "numel": old.numel(),
            "changed": changed,
            "fraction_changed": changed / old.numel(),
            "relative_l2": delta / norm if norm > 0 else (0.0 if delta == 0 else float("inf")),
        }
        touched += changed > 0
        changed_total += changed
        weights_total += old.numel()
        sq_change += delta**2
        sq_base += norm**2
    return {
        "tensors_total": len(base),
        "tensors_touched": touched,
        "weights_total": weights_total,
        "weights_changed": changed_total,
        "fraction_changed": changed_total / weights_total,
        "relative_l2": (sq_change**0.5) / (sq_base**0.5) if sq_base > 0 else 0.0,
        "per_tensor": per_tensor,
    }


def check_new_params(base: dict[str, Tensor], new: dict[str, Tensor]) -> None:
    """Reject anything that would not load as the same architecture with finite numbers."""
    if base.keys() != new.keys():
        missing = sorted(base.keys() - new.keys())
        extra = sorted(new.keys() - base.keys())
        raise EditError(f"parameter names differ: missing {missing}, unexpected {extra}")
    for name, value in new.items():
        if value.shape != base[name].shape:
            raise EditError(f"{name}: shape {tuple(value.shape)} != {tuple(base[name].shape)}")
        if value.dtype != torch.float32:
            raise EditError(f"{name}: dtype {value.dtype}, expected float32")
        if not torch.isfinite(value).all():
            raise EditError(f"{name}: contains NaN or infinity")


def write_edit(
    base_dir: Path, new: dict[str, Tensor], out_dir: Path, name: str, script: Path
) -> dict[str, Any]:
    """Save the edited checkpoint to `out_dir` and return its diff summary (also written there)."""
    base = load_base_params(base_dir)
    check_new_params(base, new)
    model = GPT.from_pretrained(base_dir)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for key, value in new.items():
            params[key].copy_(value)
    model.save_pretrained(out_dir)
    shutil.copyfile(base_dir / TOKENIZER_FILE, out_dir / TOKENIZER_FILE)

    reloaded = {n: p.detach().clone() for n, p in GPT.from_pretrained(out_dir).named_parameters()}
    for key, value in new.items():
        if not torch.equal(reloaded[key], value):
            raise EditError(f"{key}: saved values do not reload exactly")

    summary = diff_summary(base, reloaded)
    summary.update(
        {
            "edit": name,
            "script": str(script),
            "script_sha256": file_sha256(script),
            "base_sha256": file_sha256(base_dir / WEIGHTS_FILE),
            "edited_sha256": file_sha256(out_dir / WEIGHTS_FILE),
        }
    )
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
