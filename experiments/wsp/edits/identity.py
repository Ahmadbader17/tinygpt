"""Identity edit: returns every original value unchanged.

A Phase 1 sanity check, not a study edit. It proves that the pipeline (rebuild, save, reload,
diff summary) changes nothing when nothing should change.

Rebuild from the original checkpoint:  uv run python experiments/wsp/edits/identity.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from tinygpt.wsp.edits import load_base_params, write_edit

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "runs" / "wsp" / "original"
OUT = REPO / "runs" / "wsp" / "edits" / "identity"


def build(base_dir: Path) -> dict[str, torch.Tensor]:
    """New values for every parameter. Here: the originals, cloned."""
    return load_base_params(base_dir)


def main() -> None:
    summary = write_edit(BASE, build(BASE), OUT, "identity", Path(__file__))
    print(
        f"identity: {summary['weights_changed']} weights changed, "
        f"sha256 {summary['edited_sha256'][:12]}"
    )


if __name__ == "__main__":
    main()
