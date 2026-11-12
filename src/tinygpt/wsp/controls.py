"""Random edits matched to a real edit: the control that asks "would any change this size do?"

A control touches the same tensors, at the same entries, with the same Frobenius norm per
tensor as the edit it is matched to. Only the direction is random.
"""

from __future__ import annotations

import torch
from torch import Tensor

MAX_LOW_RANK = 4  # PLAN.md section 7: edits of rank 4 or less also get a rank-matched control


def _generator(seed: int, index: int) -> torch.Generator:
    return torch.Generator().manual_seed(1_000_003 * seed + index)


def _scaled(noise: Tensor, norm: float) -> Tensor:
    size = noise.double().norm().item()
    return noise * (norm / size) if size > 0 else noise


def matched_gaussian(
    base: dict[str, Tensor], edited: dict[str, Tensor], seed: int
) -> dict[str, Tensor]:
    """Gaussian noise on exactly the entries the edit changed, same norm per tensor."""
    out: dict[str, Tensor] = {}
    for index, name in enumerate(sorted(base)):
        old, new = base[name], edited[name]
        mask = old != new
        if not bool(mask.any()):
            out[name] = old.clone()
            continue
        norm = (new.double() - old.double()).norm().item()
        noise = torch.randn(old.shape, generator=_generator(seed, index)) * mask
        out[name] = old + _scaled(noise, norm)
    return out


def edit_ranks(base: dict[str, Tensor], edited: dict[str, Tensor]) -> dict[str, int]:
    """Numerical rank of the change to each touched matrix (values above 1e-5 of the top one)."""
    ranks: dict[str, int] = {}
    for name in sorted(base):
        delta = edited[name].double() - base[name].double()
        if delta.ndim == 2 and bool((delta != 0).any()):
            sv = torch.linalg.svdvals(delta)
            ranks[name] = int((sv > 1e-5 * sv[0]).sum())
    return ranks


def matched_low_rank(
    base: dict[str, Tensor], edited: dict[str, Tensor], seed: int
) -> dict[str, Tensor] | None:
    """A random change of the same rank, rows, columns and norm; None if the edit is not low rank.

    Applies when every touched matrix changed by rank `MAX_LOW_RANK` or less. The random
    change lives on the rows and columns the edit touched. Touched vectors (biases, gains) get
    Gaussian noise on their changed entries, as in `matched_gaussian`.
    """
    ranks = edit_ranks(base, edited)
    if not ranks or max(ranks.values()) > MAX_LOW_RANK:
        return None
    out: dict[str, Tensor] = {}
    for index, name in enumerate(sorted(base)):
        old, new = base[name], edited[name]
        mask = old != new
        if not bool(mask.any()):
            out[name] = old.clone()
            continue
        norm = (new.double() - old.double()).norm().item()
        gen = _generator(seed, index)
        if name in ranks:
            rows = mask.any(dim=1).nonzero().squeeze(1)
            cols = mask.any(dim=0).nonzero().squeeze(1)
            left = torch.randn((len(rows), ranks[name]), generator=gen)
            right = torch.randn((ranks[name], len(cols)), generator=gen)
            change = torch.zeros_like(old)
            change[rows.unsqueeze(1), cols.unsqueeze(0)] = left @ right
        else:
            change = torch.randn(old.shape, generator=gen) * mask
        out[name] = old + _scaled(change, norm)
    return out
