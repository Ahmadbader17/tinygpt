"""Planted faults for Exp 2: five families, each applied from a small spec.

`draw_fault` picks a location and a starting magnitude from a random generator; `apply_fault`
turns a spec into damaged parameters, deterministically. The injector script calibrates the
magnitude on dev data and seals the specs. Nothing here runs the model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

FAMILIES = ("head_scale", "rank_one", "head_zero", "ln_gain", "noise")
NOISE_TARGETS = ("attn.qkv.weight", "attn.proj.weight", "mlp.fc.weight", "mlp.proj.weight")
MAGNITUDE_KEY = {  # the one number calibration may scale, per family
    "head_scale": "factor",
    "rank_one": "alpha_rel",
    "ln_gain": "factor",
    "noise": "std_rel",
}


def draw_fault(
    family: str, rng: np.random.Generator, n_layer: int, n_head: int, d_model: int
) -> dict[str, Any]:
    """A fault spec with a random location and a magnitude from the preregistered range."""
    layer = int(rng.integers(n_layer))
    spec: dict[str, Any] = {"family": family, "layer": layer, "head": None}
    if family in ("head_scale", "head_zero"):
        spec["tensor"] = f"blocks.{layer}.attn.proj.weight"
        spec["head"] = int(rng.integers(n_head))
        if family == "head_scale":
            spec["factor"] = float(rng.uniform(3.0, 8.0))
    elif family == "rank_one":
        spec["tensor"] = f"blocks.{layer}.mlp.proj.weight"
        spec["alpha_rel"] = float(rng.uniform(1.0, 3.0))  # times the matrix's spectral norm
        spec["direction_seed"] = int(rng.integers(2**31))
    elif family == "ln_gain":
        which = "ln1" if rng.random() < 0.5 else "ln2"
        spec["tensor"] = f"blocks.{layer}.{which}.weight"
        spec["indices"] = sorted(int(i) for i in rng.choice(d_model, d_model // 4, replace=False))
        spec["factor"] = float(rng.uniform(4.0, 8.0))
    elif family == "noise":
        spec["tensor"] = f"blocks.{layer}.{NOISE_TARGETS[int(rng.integers(len(NOISE_TARGETS)))]}"
        spec["std_rel"] = float(rng.uniform(0.5, 1.5))  # times the matrix's own std
        spec["noise_seed"] = int(rng.integers(2**31))
    else:
        raise ValueError(f"unknown fault family {family!r}")
    return spec


def head_columns(head: int, n_head: int, d_model: int) -> slice:
    """Input columns of `attn.proj.weight` that belong to one head (heads are concatenated)."""
    head_dim = d_model // n_head
    return slice(head * head_dim, (head + 1) * head_dim)


def apply_fault(params: dict[str, Tensor], spec: dict[str, Any], n_head: int) -> dict[str, Tensor]:
    """Damaged copies of `params`. Only `spec["tensor"]` changes."""
    out = {name: value.clone() for name, value in params.items()}
    w = out[spec["tensor"]]
    family = spec["family"]
    if family == "head_scale":
        w[:, head_columns(spec["head"], n_head, w.shape[0])] *= spec["factor"]
    elif family == "head_zero":
        w[:, head_columns(spec["head"], n_head, w.shape[0])] = 0.0
    elif family == "rank_one":
        gen = torch.Generator().manual_seed(spec["direction_seed"])
        u = torch.randn(w.shape[0], generator=gen)
        v = torch.randn(w.shape[1], generator=gen)
        u, v = u / u.norm(), v / v.norm()
        alpha = spec["alpha_rel"] * torch.linalg.matrix_norm(params[spec["tensor"]], ord=2).item()
        w += alpha * torch.outer(u, v)
    elif family == "ln_gain":
        w[spec["indices"]] *= spec["factor"]
    elif family == "noise":
        gen = torch.Generator().manual_seed(spec["noise_seed"])
        std = spec["std_rel"] * params[spec["tensor"]].std().item()
        w += std * torch.randn(w.shape, generator=gen)
    else:
        raise ValueError(f"unknown fault family {family!r}")
    return out


def score_diagnosis(diagnosis: dict[str, Any], spec: dict[str, Any]) -> dict[str, bool | None]:
    """Did the editor name the damaged tensor, and the head where the fault has one?"""
    tensor = diagnosis.get("tensor") == spec["tensor"]
    head: bool | None = None
    if spec["head"] is not None:
        head = tensor and diagnosis.get("head") == spec["head"]
    return {"tensor_correct": bool(tensor), "sublocation_correct": head}
