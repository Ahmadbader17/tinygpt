"""metrics.csv writer and the loss plot."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a headless run
import matplotlib.pyplot as plt  # noqa: E402

FIELDS = ["step", "train_loss", "val_loss", "lr", "tokens_seen", "tokens_per_sec", "elapsed"]


class MetricsLogger:
    """Appends rows, so a resumed run continues the same file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(FIELDS)

    def log(self, **row: float | int | None) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(["" if row[k] is None else row[k] for k in FIELDS])


def plot_loss(csv_path: Path, out_path: Path) -> None:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([int(r["step"]) for r in rows], [float(r["train_loss"]) for r in rows], label="train")
    val = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["val_loss"]]
    if val:
        ax.plot(*zip(*val, strict=True), marker="o", label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
