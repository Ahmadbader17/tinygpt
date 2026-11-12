from __future__ import annotations

import numpy as np
import torch

from tinygpt.wsp.headscore import (
    CLASSES,
    SKIP_ROWS,
    evaluate_predictions,
    ground_truth,
    hypergeom_tail,
    offset_score,
    pattern_cosine,
    permutation_test,
    previous_token_score,
    spearman,
    start_anchor_score,
)

T = 40


def patterns() -> torch.Tensor:
    """(1, 3, T, T): head 0 looks one back, head 1 at position 0, head 2 spreads evenly."""
    probs = torch.zeros(1, 3, T, T)
    for t in range(T):
        probs[0, 0, t, max(t - 1, 0)] = 1.0
        probs[0, 1, t, 0] = 1.0
        probs[0, 2, t, : t + 1] = 1.0 / (t + 1)
    return probs


def test_previous_token_and_start_scores_pick_the_right_heads() -> None:
    probs = patterns()
    prev = previous_token_score(probs)
    start = start_anchor_score(probs)
    assert prev[0] == 1.0 and prev[1] < 0.05 and prev[2] < 0.2
    assert start[1] == 1.0 and start[0] == 0.0


def test_offset_score_finds_attention_to_a_fixed_distance() -> None:
    probs = torch.zeros(1, 1, T, T)
    period = 10
    for t in range(period, T):
        probs[0, 0, t, t - period + 1] = 0.7  # the token after the earlier copy
        probs[0, 0, t, t - period] = 0.3  # the earlier copy itself
    assert torch.allclose(offset_score(probs, period, period - 1), torch.tensor([0.7]))
    assert torch.allclose(offset_score(probs, period, period), torch.tensor([0.3]))


def test_pattern_cosine_is_one_for_position_only_heads() -> None:
    a = patterns()[0]
    assert torch.allclose(pattern_cosine(a, a), torch.ones(3, dtype=torch.float64))
    shuffled = a.clone()
    # a content-driven head would look somewhere else on other text: move head 2's mass
    shuffled[2] = 0.0
    for t in range(SKIP_ROWS, T):
        shuffled[2, t, t] = 1.0
    cos = pattern_cosine(a, shuffled)
    assert abs(cos[0] - 1.0) < 1e-9 and cos[2] < 0.5


def test_ground_truth_uses_below_threshold_for_inactive() -> None:
    scores = {c: np.array([0.0, 0.5]) for c in CLASSES}
    scores["inactive"] = np.array([0.0005, -0.01])  # tiny effect, clear effect
    truth = ground_truth(scores)
    assert truth["previous_token"].tolist() == [False, True]
    assert truth["inactive"].tolist() == [True, False]


def test_hypergeometric_tail_known_values() -> None:
    assert hypergeom_tail(48, 6, 6, 0) == 1.0
    # all 2 picks are among the 2 positives of 4: 1 / C(4,2)
    assert abs(hypergeom_tail(4, 2, 2, 2) - 1 / 6) < 1e-12
    assert hypergeom_tail(48, 3, 3, 3) < 1e-4


def _layers() -> np.ndarray:
    return np.repeat(np.arange(6), 8)


def test_perfect_predictions_beat_both_nulls_and_random_ones_do_not() -> None:
    rng = np.random.default_rng(0)
    truth = rng.random((48, 3)) < 0.2
    good = permutation_test(truth.copy(), truth, _layers(), within_layer=True, n=500)
    assert good["observed_f1"] == 1.0 and good["p_value"] < 0.01
    random_pred = rng.random((48, 3)) < 0.2
    bad = permutation_test(random_pred, truth, _layers(), within_layer=False, n=500)
    assert bad["p_value"] > 0.05


def test_layer_knowledge_alone_does_not_beat_the_within_layer_null() -> None:
    # truth: one class, 4 of the 8 heads of layer 0. Prediction: "all of layer 0".
    truth = np.zeros((48, 1), dtype=bool)
    truth[:4, 0] = True
    pred = np.zeros((48, 1), dtype=bool)
    pred[:8, 0] = True
    assert permutation_test(pred, truth, _layers(), within_layer=False, n=500)["p_value"] < 0.01
    # shuffling heads inside layers cannot change this prediction, so it earns no credit
    assert permutation_test(pred, truth, _layers(), within_layer=True, n=500)["p_value"] == 1.0


def test_spearman_handles_ties_and_constants() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    up, down = spearman(x, x * 2, n=200)["rho"], spearman(x, -x, n=200)["rho"]
    assert up is not None and abs(up - 1.0) < 1e-9
    assert down is not None and abs(down + 1.0) < 1e-9
    assert spearman(np.ones(5), x, n=200) == {"rho": None, "p_value": None}


def test_evaluate_predictions_reports_h1() -> None:
    rng = np.random.default_rng(1)
    scores = {c: rng.random(48) * 0.3 for c in CLASSES}
    scores["previous_token"][:5] = 0.9
    scores["start_anchor"][8:12] = 0.8
    scores["inactive"] = rng.random(48) * 0.05 + 0.01  # no head is inactive
    truth = ground_truth(scores)
    exact = {c: truth[c].astype(float) for c in CLASSES}
    report = evaluate_predictions(exact, scores, _layers())
    assert report["h1_met"] is True
    assert "inactive" not in report["classes_with_positives"]
    assert report["micro_over_classes_with_positives"]["f1"] == 1.0
    assert report["per_class"]["previous_token"]["hypergeometric_p"] < 1e-5
    blank = {c: np.zeros(48) for c in CLASSES}
    assert evaluate_predictions(blank, scores, _layers())["h1_met"] is False
