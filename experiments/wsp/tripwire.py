"""Access-time tripwires for the files an editor must not read.

`arm` sets each watched file's access time to just before its modification time. On this
filesystem (ext4, relatime) the first read after that moves the access time forward, so
`check` can tell whether anything read the file in between. This catches a read; it does not
prevent one, and it cannot tell who read. Nothing else may read the watched files between
`arm` and `check`: no git, lint, type check or tests.

    uv run python experiments/wsp/tripwire.py arm exp2
    uv run python experiments/wsp/tripwire.py check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATE = REPO / "runs" / "wsp" / "tripwire_state.json"
CLEAN_SHA256 = "6d9615059a7d499ad645f2f310fef70f91388a4804714becb008c72d90cb86ec"
CLEAN_BYTES = 23419560
WEIGHT_ROOTS = (REPO / "runs", REPO / "export", Path.home() / ".cache" / "huggingface")


def clean_copies() -> list[Path]:
    """Every copy of the healthy checkpoint on this machine that we know how to find."""
    found = []
    for root in WEIGHT_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            candidate = path.is_file() and not path.is_symlink()
            if (
                candidate
                and path.stat().st_size == CLEAN_BYTES
                and hashlib.sha256(path.read_bytes()).hexdigest() == CLEAN_SHA256
            ):
                found.append(path)
    return found


def watched(profile: str) -> list[Path]:
    wsp = REPO / "experiments" / "wsp"
    files = [wsp / "PLAN.md"]
    sealed = REPO / "runs" / "wsp" / "sealed"
    if sealed.exists():
        files += [p for p in sealed.rglob("*") if p.is_file()]
    if profile == "exp2":
        # the healthy model, and everything that names the fault families or the specs
        files += clean_copies()
        files += [
            wsp / "inject_faults.py",
            REPO / "src" / "tinygpt" / "wsp" / "faults.py",
            REPO / "tests" / "test_wsp_faults_controls.py",
            wsp / "logs" / "phase2" / "journal.md",
        ]
    elif profile == "exp3":
        # other experiments' findings; attempts must be independent of them
        for folder in (wsp / "results", wsp / "logs" / "exp1", wsp / "logs" / "exp2"):
            if folder.exists():
                files += [p for p in folder.rglob("*") if p.is_file()]
        files += [p for p in (wsp / "analysis").rglob("*") if p.is_file()]
    return sorted({p for p in files if p.exists()})


def arm(profile: str) -> None:
    state = {}
    for path in watched(profile):
        st = path.stat()
        os.utime(path, ns=(st.st_mtime_ns - 10_000_000_000, st.st_mtime_ns))
        state[str(path)] = path.stat().st_atime_ns
    STATE.write_text(json.dumps({"profile": profile, "files": state}, indent=1))
    print(f"armed {len(state)} files ({profile})")


def check() -> int:
    data = json.loads(STATE.read_text())
    tripped = [p for p, atime in data["files"].items() if Path(p).stat().st_atime_ns != atime]
    missing = [p for p in data["files"] if not Path(p).exists()]
    print(json.dumps({"watched": len(data["files"]), "tripped": tripped, "missing": missing}))
    return 1 if tripped or missing else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Arm or check the access-time tripwires.")
    parser.add_argument("action", choices=("arm", "check"))
    parser.add_argument("profile", nargs="?", choices=("exp2", "exp3"), default="exp2")
    args = parser.parse_args()
    if args.action == "arm":
        arm(args.profile)
        return 0
    return check()


if __name__ == "__main__":
    sys.exit(main())
