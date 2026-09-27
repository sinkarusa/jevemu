"""Calibration metrics on ragged probability rows, weighted ECE and the paired cluster
bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from jevemu.eval.metrics import ece
from jevemu.eval.metrics_calib import ItemMetrics, bootstrap_replicates, interval, weighted_ece


def test_item_metrics_score_rows_of_different_lengths_on_their_own_keys() -> None:
    metrics = ItemMetrics.from_probs([np.array([0.7, 0.3]), np.array([0.1, 0.2, 0.3, 0.4])], [1, 3])
    assert metrics.correct.tolist() == [False, True]
    assert metrics.confidence.tolist() == pytest.approx([0.7, 0.4])
    assert metrics.nll.tolist() == pytest.approx([-np.log(0.3), -np.log(0.4)])
    # Brier over each row's own keys: padding adds nothing.
    assert metrics.brier.tolist() == pytest.approx(
        [0.7**2 + 0.7**2, 0.1**2 + 0.2**2 + 0.3**2 + 0.6**2]
    )


def test_weighted_ece_equals_ece_of_the_expanded_sample() -> None:
    rng = np.random.default_rng(0)
    n = 57
    confidence = np.sort(rng.uniform(0.2, 1.0, n))  # distinct, ascending
    correct = (rng.uniform(size=n) < confidence).astype(np.float64)
    weights = rng.integers(0, 4, size=(25, n)).astype(np.float64)
    weights[:, 0] += 1  # every replicate non-empty
    got = weighted_ece(confidence, correct, weights, n_bins=10)
    for row, w in enumerate(weights.astype(np.int64)):
        expanded = ece(np.repeat(confidence, w), np.repeat(correct, w), n_bins=10)
        assert got[row] == pytest.approx(expanded, abs=1e-12)


def test_identical_series_have_zero_paired_differences() -> None:
    rng = np.random.default_rng(1)
    probs = list(rng.dirichlet(np.ones(3), size=80))
    gold = list(rng.integers(0, 3, size=80))
    a = ItemMetrics.from_probs(probs, gold)
    reps = bootstrap_replicates(
        [f"q{i // 2}" for i in range(80)], {"a": a, "b": a}, n_resamples=300
    )
    for metric in ("accuracy", "nll", "brier", "ece"):
        diff = reps["a"][metric] - reps["b"][metric]
        assert interval(0.0, diff).to_json() == [0.0, 0.0, 0.0]


def test_percentile_interval_of_accuracy_covers_the_truth_about_95_percent_of_the_time() -> None:
    rng = np.random.default_rng(2)
    p_true, n, sims = 0.8, 200, 600
    covered = 0
    for sim in range(sims):
        correct = rng.uniform(size=n) < p_true
        confidence = np.full(n, 0.5)
        metrics = ItemMetrics(correct, confidence, np.zeros(n), np.zeros(n))
        reps = bootstrap_replicates(
            [str(i) for i in range(n)], {"s": metrics}, n_resamples=500, seed=sim
        )
        ci = interval(float(correct.mean()), reps["s"]["accuracy"])
        covered += ci.low <= p_true <= ci.high
    assert 0.93 <= covered / sims <= 0.97
