"""Small run helpers: seeding, device choice, logging setup and the environment probe."""

import contextlib
import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch so the same config gives the same CPU results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    """cuda, then mps, then cpu. Nothing may require a GPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Log to stderr and, for training runs, to runs/<name>/train.log as well."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers
    )


def probe_environment() -> dict[str, Any]:
    """OS, Python, torch, cores, devices and RAM; saved with every run."""
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
