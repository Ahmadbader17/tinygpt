"""Exp 1: measuring what attention heads do, and scoring predictions made from weights alone.

Classes, scores and thresholds are the ones fixed in experiments/wsp/PLAN.md section 7 before
any weight was inspected. Scores are functions of attention patterns (B, H, T, T); row t is
where position t looks.
"""

from __future__ import annotations

from math import comb
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

CLASSES = (
    "previous_token",
    "start_anchor",
    "induction",
    "duplicate_token",
    "positional",
    "inactive",
)
THRESHOLDS = {
    "previous_token": 0.40,  # mean attention to position t-1
    "start_anchor": 0.40,  # mean attention to position 0, for t >= 16
    "induction": 0.20,  # repeated random tokens: attention to the token after the earlier copy
    "duplicate_token": 0.20,  # repeated random tokens: attention to the earlier copy itself
    "positional": 0.80,  # cosine similarity of patterns on different text
    "inactive": 0.002,  # |change in dev loss| when the head is zeroed, in nats: BELOW this
}
SKIP_ROWS = 16  # start-anchor and positional scores ignore the first rows, which have few choices
N_PERMUTATIONS = 10_000
PERMUTATION_SEED = 4242

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def previous_token_score(probs: Tensor) -> Tensor:
    """(B, H, T, T) -> (H,): mean attention from t to t-1, over t >= 1."""
    return probs.diagonal(offset=-1, dim1=-2, dim2=-1).mean(dim=(0, 2))


def start_anchor_score(probs: Tensor) -> Tensor:
    """(B, H, T, T) -> (H,): mean attention to position 0 from rows t >= SKIP_ROWS."""
    return probs[:, :, SKIP_ROWS:, 0].mean(dim=(0, 2))


def offset_score(probs: Tensor, first: int, offset: int) -> Tensor:
    """(B, H, T, T) -> (H,): mean attention from t to t - offset, for rows t >= first."""
    t = torch.arange(first, probs.shape[-1])
    return probs[:, :, t, t - offset].mean(dim=(0, 2))


def pattern_cosine(a: Tensor, b: Tensor) -> Tensor:
    """Cosine similarity of two windows' patterns, (H, T, T) each -> (H,), rows >= SKIP_ROWS."""
    x = a[:, SKIP_ROWS:].reshape(a.shape[0], -1).double()
    y = b[:, SKIP_ROWS:].reshape(b.shape[0], -1).double()
    cos: Tensor = (x * y).sum(dim=1) / (x.norm(dim=1) * y.norm(dim=1))
    return cos


def ground_truth(scores: dict[str, FloatArray]) -> dict[str, BoolArray]:
    """Class membership of every head from its measured scores."""
    truth: dict[str, BoolArray] = {}
    for name in CLASSES:
        if name == "inactive":
            truth[name] = np.abs(scores[name]) < THRESHOLDS[name]
        else:
            truth[name] = scores[name] >= THRESHOLDS[name]
    return truth


def hypergeom_tail(population: int, positives: int, drawn: int, hits: int) -> float:
    """P(at least `hits` positives) when `drawn` heads are picked at random."""
    total = comb(population, drawn)
    top = min(positives, drawn)
    return (
        sum(
            comb(positives, k) * comb(population - positives, drawn - k)
            for k in range(hits, top + 1)
        )
        / total
    )


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"precision": precision, "recall": recall, "f1": f1}


def micro(pred: BoolArray, truth: BoolArray) -> dict[str, Any]:
    """Counts pooled over the given (heads, classes) matrices."""
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    return {"tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)}


def _micro_f1(pred: BoolArray, truth: BoolArray) -> float:
    tp = (pred & truth).sum()
    denom = 2 * tp + (pred & ~truth).sum() + (~pred & truth).sum()
    return float(2 * tp / denom) if denom else 0.0


def permutation_test(
    pred: BoolArray,
    truth: BoolArray,
    layer_of: NDArray[np.int64],
    within_layer: bool,
    n: int = N_PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> dict[str, float]:
    """Is micro-F1 better than the same predictions handed to shuffled heads?

    Rows of `pred` (one per head) are permuted, either across all heads or only among the
    heads of the same layer. The within-layer null removes credit for knowing which layer
    tends to hold which kind of head.
    """
    rng = np.random.default_rng(seed)
    observed = _micro_f1(pred, truth)
    null = np.empty(n)
    heads = np.arange(len(pred))
    for i in range(n):
        if within_layer:
            order = heads.copy()
            for layer in np.unique(layer_of):
                members = heads[layer_of == layer]
                order[members] = rng.permutation(members)
        else:
            order = rng.permutation(heads)
        null[i] = _micro_f1(pred[order], truth)
    return {
        "observed_f1": observed,
        "null_mean_f1": float(null.mean()),
        "null_95th_f1": float(np.quantile(null, 0.95)),
        "p_value": float((1 + (null >= observed - 1e-12).sum()) / (1 + n)),
    }


def _ranks(values: FloatArray) -> FloatArray:
    """Ranks with ties sharing their average rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    for v in np.unique(values):
        tied = values == v
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x: FloatArray, y: FloatArray, n: int = N_PERMUTATIONS) -> dict[str, float | None]:
    """Rank correlation with a one-sided permutation p-value (positive association)."""
    rx, ry = _ranks(np.asarray(x, dtype=np.float64)), _ranks(np.asarray(y, dtype=np.float64))
    if rx.std() == 0 or ry.std() == 0:
        return {"rho": None, "p_value": None}
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(PERMUTATION_SEED)
    null = np.array([np.corrcoef(rng.permutation(rx), ry)[0, 1] for _ in range(n)])
    return {"rho": rho, "p_value": float((1 + (null >= rho - 1e-12).sum()) / (1 + n))}


def evaluate_predictions(
    confidence: dict[str, FloatArray], scores: dict[str, FloatArray], layer_of: NDArray[np.int64]
) -> dict[str, Any]:
    """Every number reported for Exp 1, and the preregistered verdict on H1."""
    truth = ground_truth(scores)
    n_heads = len(layer_of)
    per_class: dict[str, Any] = {}
    for name in CLASSES:
        conf = np.asarray(confidence[name], dtype=np.float64)
        pred, real = conf >= 0.5, truth[name]
        counts = micro(pred, real)
        base_rate = float(real.mean())
        brier = float(((conf - real) ** 2).mean())
        brier_ref = float(((base_rate - real) ** 2).mean())
        # for "inactive" a smaller measured effect means more inactive, so flip the sign
        measured = -np.abs(scores[name]) if name == "inactive" else scores[name]
        per_class[name] = {
            **counts,
            "positives": int(real.sum()),
            "predicted": int(pred.sum()),
            "base_rate": base_rate,
            "chance_precision": base_rate,
            "hypergeometric_p": hypergeom_tail(
                n_heads, int(real.sum()), int(pred.sum()), counts["tp"]
            )
            if real.any() and pred.any()
            else None,
            "spearman": spearman(conf, measured),
            "brier": brier,
            "brier_base_rate": brier_ref,
            "brier_skill": 1 - brier / brier_ref if brier_ref > 0 else None,
        }
    scored = [name for name in CLASSES if truth[name].any()]
    pred_all = np.stack([np.asarray(confidence[c]) >= 0.5 for c in CLASSES], axis=1)
    truth_all = np.stack([truth[c] for c in CLASSES], axis=1)
    keep = [CLASSES.index(c) for c in scored]
    pred_s, truth_s = pred_all[:, keep], truth_all[:, keep]
    micro_scored = micro(pred_s, truth_s)
    null_global = permutation_test(pred_s, truth_s, layer_of, within_layer=False)
    null_layer = permutation_test(pred_s, truth_s, layer_of, within_layer=True)
    precision = micro_scored["precision"]
    h1 = bool(null_layer["p_value"] < 0.05 and precision is not None and precision >= 0.5)
    return {
        "thresholds": THRESHOLDS,
        "per_class": per_class,
        "classes_with_positives": scored,
        "micro_over_classes_with_positives": micro_scored,
        "micro_over_all_classes": micro(pred_all, truth_all),  # stricter: empty classes count
        "permutation_global": null_global,
        "permutation_within_layer": null_layer,
        "h1_met": h1,
        "h1_rule": "within-layer permutation p < 0.05 and micro-precision >= 0.5",
    }
