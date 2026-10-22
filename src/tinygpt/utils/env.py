"""Environment probe, saved to docs/ENVIRONMENT.md and to every run directory."""

import contextlib
import os
import platform
import sys
from typing import Any

import torch


def probe_environment() -> dict[str, Any]:
    """Collect OS, Python, torch, cores, devices and RAM."""
    ram_gb: float | None = None
    with contextlib.suppress(ValueError, OSError, AttributeError):
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "logical_cores": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "ram_gb": round(ram_gb, 1) if ram_gb is not None else None,
    }
