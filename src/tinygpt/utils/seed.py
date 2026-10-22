"""Seeding."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch so the same config gives the same CPU results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
