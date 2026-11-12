"""Exp 2: build five damaged checkpoints from seeded fault specs, and seal the specs.

One fault per family (see tinygpt.wsp.faults), in a seeded order so a checkpoint's number says
nothing about its family. Each fault's magnitude is calibrated on dev data until dev loss
rises by 0.05 to 1.0 nats (a zeroed head: at least 0.02). This script prints only
"checkpoint k: damage confirmed": the specs and the measured damage go to the sealed folder,
which the editors and the person running this do not read before the final evaluation.

    uv run python experiments/wsp/inject_faults.py                # draw, calibrate, seal
    uv run python experiments/wsp/inject_faults.py --from-specs F  # rebuild from published specs
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from tinygpt.data.prepare import read_clean_split
from tinygpt.model import GPT
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp import protocol
from tinygpt.wsp.edits import load_base_params, write_edit
from tinygpt.wsp.faults import FAMILIES, MAGNITUDE_KEY, apply_fault, draw_fault
from tinygpt.wsp.metrics import bootstrap_ratio_diff
from tinygpt.wsp.scoring import story_nll

REPO = Path(__file__).resolve().parents[2]
ORIGINAL = REPO / "runs" / "wsp" / "original"
SEALED = REPO / "runs" / "wsp" / "sealed" / "exp2"
SEED = 20261112
CALIBRATION_STORIES = 500
DAMAGE_MIN, DAMAGE_MAX = 0.05, 1.0  # nats per token on dev
ZERO_HEAD_MIN = 0.02  # a zeroed head has no magnitude to tune
MAX_STEPS, MAX_REDRAWS, STEP = 12, 20, 1.5


def set_params(model: GPT, params: dict[str, Tensor]) -> None:
    own = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in params.items():
            own[name].copy_(value)


class Calibrator:
    """Measures how much a fault costs on dev stories. Harness passes, not editor passes."""

    def __init__(self) -> None:
        self.model = GPT.from_pretrained(ORIGINAL).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.base = load_base_params(ORIGINAL)
        tok = BPETokenizer.load(ORIGINAL / "tokenizer.json")
        stories = list(read_clean_split(REPO / "data" / "raw" / "val.txt"))
        self.ids = tok.encode_batch(protocol.dev_loss_stories(stories)[:CALIBRATION_STORIES])
        self.eot = tok.eot_id
        self.clean_nll, count = story_nll(self.model, self.ids, self.eot)
        self.tokens = count.astype(np.float64)
        self.evaluations = 1

    def damage(self, spec: dict[str, Any]) -> tuple[float, bool]:
        """Dev loss increase in nats per token, and whether its paired interval excludes zero."""
        set_params(self.model, apply_fault(self.base, spec, self.model.cfg.n_head))
        nll, _ = story_nll(self.model, self.ids, self.eot)
        set_params(self.model, self.base)
        self.evaluations += 1
        if not np.isfinite(nll).all():
            return math.inf, True
        diff = bootstrap_ratio_diff(nll, self.clean_nll, self.tokens, n_boot=2000)
        return diff.estimate, diff.low > 0

    def calibrate(self, spec: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        """Scale the fault's magnitude into the damage window. Returns (confirmed, spec)."""
        key = MAGNITUDE_KEY.get(spec["family"])
        for _ in range(MAX_STEPS):
            cost, real = self.damage(spec)
            floor = ZERO_HEAD_MIN if key is None else DAMAGE_MIN
            if real and floor <= cost and (key is None or cost <= DAMAGE_MAX):
                return True, {**spec, "dev_damage": cost}
            if key is None:
                return False, spec  # nothing to scale: the caller draws another head
            spec = {**spec, key: spec[key] * STEP if cost < DAMAGE_MIN else spec[key] / STEP}
        return False, spec


def scaled_fault(
    cal: Calibrator, rng: np.random.Generator, family: str
) -> tuple[dict[str, Any], int]:
    """Draw locations until one can be calibrated into the damage window."""
    cfg = cal.model.cfg
    for redraws in range(MAX_REDRAWS):
        candidate = draw_fault(family, rng, cfg.n_layer, cfg.n_head, cfg.d_model)
        confirmed, spec = cal.calibrate(candidate)
        if confirmed:
            return spec, redraws
    raise SystemExit("a fault could not be calibrated into the damage window")


def zeroed_head(cal: Calibrator, rng: np.random.Generator) -> tuple[dict[str, Any], int]:
    """Try heads in a seeded order; take the first whose removal costs enough.

    If no head reaches the floor, the costliest head is used and the spec says so.
    """
    cfg = cal.model.cfg
    best: tuple[float, dict[str, Any]] | None = None
    for tried, index in enumerate(rng.permutation(cfg.n_layer * cfg.n_head)):
        layer, head = divmod(int(index), cfg.n_head)
        spec: dict[str, Any] = {
            "family": "head_zero",
            "layer": layer,
            "head": head,
            "tensor": f"blocks.{layer}.attn.proj.weight",
        }
        cost, real = cal.damage(spec)
        if real and cost >= ZERO_HEAD_MIN:
            return {**spec, "dev_damage": cost}, tried
        if real and (best is None or cost > best[0]):
            best = (cost, spec)
    if best is None:
        raise SystemExit("no head's removal measurably hurts dev loss")
    return {**best[1], "dev_damage": best[0], "below_floor": True}, cfg.n_layer * cfg.n_head


def save(name: str, spec: dict[str, Any], out_root: Path, n_head: int) -> str:
    base = load_base_params(ORIGINAL)
    summary = write_edit(
        ORIGINAL, apply_fault(base, spec, n_head), out_root / name, name, Path(__file__)
    )
    return str(summary["edited_sha256"])


def draw_and_seal() -> None:
    if SEALED.exists():
        raise SystemExit(f"{SEALED} already exists: faults are drawn once.")
    rng = np.random.default_rng(SEED)
    cal = Calibrator()
    n_head = cal.model.cfg.n_head
    order = [str(f) for f in rng.permutation(FAMILIES)]
    sealed: dict[str, Any] = {"seed": SEED, "checkpoints": {}}
    for k, family in enumerate(order, start=1):
        spec, redraws = (
            zeroed_head(cal, rng) if family == "head_zero" else scaled_fault(cal, rng, family)
        )
        name = f"ckpt_{k}"
        spec["redraws"] = redraws
        spec["sha256"] = save(name, spec, SEALED, n_head)
        sealed["checkpoints"][name] = spec
        print(f"checkpoint {k}: damage confirmed", flush=True)
    sealed["calibration_evaluations"] = cal.evaluations
    sealed["clean_dev_loss"] = float(cal.clean_nll.sum() / cal.tokens.sum())
    (SEALED / "specs.json").write_text(json.dumps(sealed, indent=2) + "\n", encoding="utf-8")
    print(f"sealed 5 specs after {cal.evaluations} dev evaluations")


def rebuild(specs_file: Path, out_root: Path) -> None:
    """Rebuild the damaged checkpoints from published specs. No calibration, no model runs."""
    sealed = json.loads(specs_file.read_text(encoding="utf-8"))
    n_head = GPT.from_pretrained(ORIGINAL).cfg.n_head
    for name, spec in sealed["checkpoints"].items():
        sha = save(name, spec, out_root, n_head)
        status = "matches" if sha == spec["sha256"] else "DIFFERS from"
        print(f"{name}: rebuilt, {status} the recorded sha256")
        if out_root != SEALED:
            shutil.copyfile(specs_file, out_root / "specs.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plant the Exp 2 faults.")
    parser.add_argument("--from-specs", type=Path, help="rebuild from a specs file")
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "wsp" / "exp2_rebuilt")
    args = parser.parse_args()
    torch.set_num_threads(12)
    if args.from_specs:
        rebuild(args.from_specs, args.out)
    else:
        draw_and_seal()


if __name__ == "__main__":
    main()
