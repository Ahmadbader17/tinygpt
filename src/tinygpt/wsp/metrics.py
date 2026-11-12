"""Repetition metrics on token sequences, and bootstrap confidence intervals.

All metrics are computed on generated token ids only (the prompt and the stop token are
excluded by the caller). Definitions are fixed in experiments/wsp/PLAN.md section 4.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

LOOP_N = 4  # a "loop" is some 4-gram ...
LOOP_MIN_COUNT = 3  # ... that occurs at least 3 times in one generation
LONG_LOOP_N = 8  # secondary: an 8-gram three times is rare in real stories, so it isolates loops
N_BOOT = 10_000
BOOT_SEED = 20261111


def ngrams(tokens: Sequence[int], n: int) -> list[tuple[int, ...]]:
    """All contiguous n-grams, in order. Empty if the sequence is shorter than n."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def repeated_ngram_fraction(tokens: Sequence[int], n: int) -> float:
    """Share of n-gram positions that repeat an n-gram seen earlier in the same sequence.

    Equals 1 - unique / total (seq-rep-n of Welleck et al. 2020). A sequence with no n-grams
    scores 0.0: it cannot repeat, and the length guardrail catches edits that shorten output.
    """
    grams = ngrams(tokens, n)
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def max_ngram_count(tokens: Sequence[int], n: int) -> int:
    """How often the most frequent n-gram occurs (0 if there are no n-grams)."""
    grams = ngrams(tokens, n)
    if not grams:
        return 0
    return max(Counter(grams).values())


def has_loop(tokens: Sequence[int], n: int = LOOP_N, min_count: int = LOOP_MIN_COUNT) -> bool:
    """True if some n-gram occurs at least `min_count` times."""
    return max_ngram_count(tokens, n) >= min_count


def distinct_n(generations: Sequence[Sequence[int]], n: int) -> float:
    """Corpus-level distinct-n (Li et al. 2016): unique n-grams / total n-grams, pooled."""
    unique: set[tuple[int, ...]] = set()
    total = 0
    for tokens in generations:
        grams = ngrams(tokens, n)
        unique.update(grams)
        total += len(grams)
    return len(unique) / total if total else 0.0


@dataclass(frozen=True)
class Interval:
    """A point estimate with a percentile bootstrap interval."""

    estimate: float
    low: float
    high: float

    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def as_dict(self) -> dict[str, float]:
        return {"estimate": self.estimate, "low": self.low, "high": self.high}


def _resample_indices(n: int, n_boot: int, seed: int) -> NDArray[np.int64]:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_mean(
    per_cluster: FloatArray, n_boot: int = N_BOOT, seed: int = BOOT_SEED, alpha: float = 0.05
) -> Interval:
    """Mean over clusters with a percentile CI; resamples clusters with replacement.

    `per_cluster` holds one value per cluster (a prompt, or a story). For a paired
    comparison pass the per-cluster difference: resampling the difference is the paired
    bootstrap, because both models are evaluated on the same resampled clusters.
    """
    values = np.asarray(per_cluster, dtype=np.float64)
    idx = _resample_indices(len(values), n_boot, seed)
    means = values[idx].mean(axis=1)  # (n_boot, n) -> (n_boot,)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Interval(float(values.mean()), float(low), float(high))


def bootstrap_ratio(
    numerator: FloatArray,
    denominator: FloatArray,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    alpha: float = 0.05,
) -> Interval:
    """sum(numerator) / sum(denominator) with a cluster bootstrap CI.

    Used for token-weighted loss: numerator is the summed negative log likelihood of each
    story and denominator its token count, so long stories weigh more, as in training.
    """
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    idx = _resample_indices(len(num), n_boot, seed)
    ratios = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    low, high = np.quantile(ratios, [alpha / 2, 1 - alpha / 2])
    return Interval(float(num.sum() / den.sum()), float(low), float(high))


def bootstrap_ratio_diff(
    numerator_a: FloatArray,
    numerator_b: FloatArray,
    denominator: FloatArray,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    alpha: float = 0.05,
) -> Interval:
    """Paired difference of two token-weighted losses (a minus b) on the same stories."""
    diff = np.asarray(numerator_a, dtype=np.float64) - np.asarray(numerator_b, dtype=np.float64)
    return bootstrap_ratio(diff, denominator, n_boot, seed, alpha)


PER_GENERATION_METRICS = ("loop_rate", "long_loop_rate", "rep2", "rep3", "rep4", "length")


def per_prompt_stats(generations: Sequence[Sequence[Sequence[int]]]) -> dict[str, FloatArray]:
    """Per-generation metrics averaged within each prompt; `generations[p][s]` is a token list.

    Returns one array of shape (n_prompts,) per metric in `PER_GENERATION_METRICS`. Prompts
    are the bootstrap clusters, because generations of the same prompt are correlated.
    """
    rows: dict[str, list[float]] = {name: [] for name in PER_GENERATION_METRICS}
    for runs in generations:
        rows["loop_rate"].append(float(np.mean([has_loop(t) for t in runs])))
        rows["long_loop_rate"].append(float(np.mean([has_loop(t, LONG_LOOP_N) for t in runs])))
        rows["rep2"].append(float(np.mean([repeated_ngram_fraction(t, 2) for t in runs])))
        rows["rep3"].append(float(np.mean([repeated_ngram_fraction(t, 3) for t in runs])))
        rows["rep4"].append(float(np.mean([repeated_ngram_fraction(t, 4) for t in runs])))
        rows["length"].append(float(np.mean([len(t) for t in runs])))
    return {name: np.asarray(values, dtype=np.float64) for name, values in rows.items()}


def bootstrap_recovery(
    nll_damaged: FloatArray,
    nll_repaired: FloatArray,
    nll_clean: FloatArray,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    alpha: float = 0.05,
) -> Interval:
    """Share of the loss a fault cost that a repair won back, with a bootstrap over stories.

    recovery = (loss damaged - loss repaired) / (loss damaged - loss clean). All three models
    are scored on the same stories, so the token counts cancel and the ratio of summed NLL
    differences is the ratio of loss differences. 1 is a full repair, 0 is no change, and a
    negative value means the "repair" made things worse.
    """
    won = np.asarray(nll_damaged, dtype=np.float64) - np.asarray(nll_repaired, dtype=np.float64)
    lost = np.asarray(nll_damaged, dtype=np.float64) - np.asarray(nll_clean, dtype=np.float64)
    idx = _resample_indices(len(won), n_boot, seed)
    ratios = won[idx].sum(axis=1) / lost[idx].sum(axis=1)
    low, high = np.quantile(ratios, [alpha / 2, 1 - alpha / 2])
    return Interval(float(won.sum() / lost.sum()), float(low), float(high))
