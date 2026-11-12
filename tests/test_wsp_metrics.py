import numpy as np

from tinygpt.wsp.metrics import (
    bootstrap_mean,
    bootstrap_ratio,
    bootstrap_ratio_diff,
    distinct_n,
    has_loop,
    max_ngram_count,
    per_prompt_stats,
    repeated_ngram_fraction,
)


def test_repeated_ngram_fraction_counts_repeats_of_earlier_ngrams() -> None:
    assert repeated_ngram_fraction([1, 2, 3, 4], 2) == 0.0
    # bigrams: (1,2) (2,1) (1,2) (2,1) (1,2) -> 2 unique of 5
    assert repeated_ngram_fraction([1, 2, 1, 2, 1, 2], 2) == 1 - 2 / 5
    assert repeated_ngram_fraction([7, 7, 7, 7, 7], 4) == 1 - 1 / 2


def test_sequences_too_short_for_an_ngram_score_zero() -> None:
    assert repeated_ngram_fraction([], 4) == 0.0
    assert repeated_ngram_fraction([1, 2, 3], 4) == 0.0
    assert max_ngram_count([1, 2, 3], 4) == 0
    assert not has_loop([1, 2, 3])


def test_loop_needs_a_4gram_three_times() -> None:
    phrase = [5, 6, 7, 8]
    assert not has_loop(phrase + [1] + phrase)  # twice is not a loop
    assert has_loop(phrase + [1] + phrase + [2] + phrase)
    assert has_loop([9] * 6)  # (9,9,9,9) occurs 3 times, overlapping counts
    assert not has_loop([9] * 5)


def test_distinct_n_pools_all_generations() -> None:
    gens = [[1, 2, 3], [1, 2, 4]]
    assert distinct_n(gens, 1) == 4 / 6
    assert distinct_n(gens, 2) == 3 / 4  # (1,2) is shared between the two generations
    assert distinct_n([], 2) == 0.0


def test_bootstrap_mean_is_deterministic_and_brackets_the_mean() -> None:
    values = np.random.default_rng(0).normal(1.0, 2.0, size=400)
    a = bootstrap_mean(values, n_boot=2000)
    b = bootstrap_mean(values, n_boot=2000)
    assert a == b
    assert a.low < a.estimate < a.high
    assert a.estimate == float(values.mean())
    # the interval is about +-1.96 standard errors wide
    assert abs((a.high - a.low) / 2 - 1.96 * values.std() / 20) < 0.03


def test_paired_difference_of_identical_models_is_exactly_zero() -> None:
    values = np.random.default_rng(1).random(50)
    ci = bootstrap_mean(values - values, n_boot=500)
    assert (ci.estimate, ci.low, ci.high) == (0.0, 0.0, 0.0)
    assert not ci.excludes_zero()


def test_bootstrap_ratio_weights_by_tokens() -> None:
    nll = np.array([10.0, 300.0])
    count = np.array([10.0, 100.0])
    ci = bootstrap_ratio(nll, count, n_boot=500)
    assert ci.estimate == 310.0 / 110.0  # not the mean of 1.0 and 3.0
    diff = bootstrap_ratio_diff(nll + count * 0.5, nll, count, n_boot=500)
    assert abs(diff.estimate - 0.5) < 1e-12 and abs(diff.low - 0.5) < 1e-12
    assert diff.excludes_zero()


def test_per_prompt_stats_average_over_the_seeds_of_each_prompt() -> None:
    loop = [5, 6, 7, 8] * 3
    stats = per_prompt_stats([[loop, [1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5, 6]]])
    assert stats["loop_rate"].tolist() == [0.5, 0.0]
    # (5,6,7,8) x 3 has only 5 8-grams and none occurs three times: a short loop, not a long one
    assert stats["long_loop_rate"].tolist() == [0.0, 0.0]
    assert per_prompt_stats([[[5, 6, 7, 8] * 5]])["long_loop_rate"].tolist() == [1.0]
    assert stats["length"].tolist() == [(12 + 5) / 2, 6.0]
    assert stats["rep4"][1] == 0.0
    # 9 4-grams in the looping run, 4 of them unique; the other run has no repeats
    assert stats["rep4"][0] == (1 - 4 / 9) / 2
