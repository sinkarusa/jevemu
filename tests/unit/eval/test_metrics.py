from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from jevemu.eval.metrics import (
    accuracy,
    brier,
    ece,
    kl_divergence,
    mcnemar_exact,
    nll,
    paired_bootstrap_ci,
    total_variation,
)


def test_accuracy_counts_exact_key_matches() -> None:
    assert accuracy(["A", "B", "C", "E"], ["A", "C", "C", "D"]) == 0.5
    assert math.isnan(accuracy([], []))


def test_nll_clips_zero_probability_to_eps() -> None:
    got = nll([1.0, 0.5, 0.0], eps=1e-6)
    np.testing.assert_allclose(got, [0.0, math.log(2.0), -math.log(1e-6)])
    assert nll([0.0], eps=1e-3)[0] == pytest.approx(-math.log(1e-3))


def test_brier_sums_squared_error_over_all_classes() -> None:
    probs = [[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]]
    np.testing.assert_allclose(brier(probs, np.array([0, 2])), [0.14, 0.78])


def test_brier_rejects_out_of_range_gold() -> None:
    with pytest.raises(ValueError, match="gold indices"):
        brier([[0.5, 0.5]], np.array([2]))


def test_ece_equal_width_bins() -> None:
    conf = [0.95, 0.95, 0.65, 0.62, 0.15]
    correct = [1, 0, 1, 1, 0]
    # bin 9: acc 0.5 vs conf 0.95; bin 6: acc 1 vs conf 0.635; bin 1: acc 0 vs conf 0.15
    expected = (2 * 0.45 + 2 * 0.365 + 0.15) / 5
    assert ece(conf, correct, binning="equal_width") == pytest.approx(expected)


def test_ece_equal_width_puts_edges_in_the_upper_bin() -> None:
    # 1.0 belongs to the last bin, 0.1 to [0.1, 0.2), not [0, 0.1)
    assert ece([1.0, 0.1], [1, 0], binning="equal_width", n_bins=10) == pytest.approx(0.05)
    assert ece([0.1, 0.1], [1, 0], binning="equal_width", n_bins=10) == pytest.approx(0.4)


def test_ece_equal_mass_splits_sorted_confidences() -> None:
    conf = [0.9, 0.6, 0.8, 0.7]
    correct = [1, 0, 1, 0]
    # bins {0.6, 0.7}: acc 0 vs 0.65; {0.8, 0.9}: acc 1 vs 0.85
    assert ece(conf, correct, n_bins=2) == pytest.approx((2 * 0.65 + 2 * 0.15) / 4)
    # 5 items in 2 bins: sizes 3 and 2
    conf5 = [0.1, 0.2, 0.3, 0.8, 0.9]
    hit5 = [0, 0, 1, 1, 1]
    expected = (3 * abs(1 / 3 - 0.2) + 2 * abs(1.0 - 0.85)) / 5
    assert ece(conf5, hit5, n_bins=2) == pytest.approx(expected)


def test_ece_of_perfectly_calibrated_bins_is_zero() -> None:
    conf = [0.5] * 4 + [1.0] * 2
    correct = [1, 0, 1, 0, 1, 1]
    assert ece(conf, correct, n_bins=10, binning="equal_width") == pytest.approx(0.0)


def test_kl_divergence_matches_hand_computation() -> None:
    got = kl_divergence([[0.5, 0.5], [0.2, 0.8]], [[0.25, 0.75], [0.2, 0.8]])
    expected = 0.5 * math.log(2.0) + 0.5 * math.log(2.0 / 3.0)
    np.testing.assert_allclose(got, [expected, 0.0], atol=1e-10)


def test_kl_divergence_is_finite_where_q_is_zero() -> None:
    got = kl_divergence([[1.0, 0.0]], [[0.0, 1.0]], eps=1e-12)[0]
    assert math.isfinite(got)
    assert got == pytest.approx(-math.log(1e-12), rel=1e-6)
    # a zero in p costs nothing (0 log 0 = 0)
    assert kl_divergence([[0.0, 1.0]], [[0.5, 0.5]])[0] == pytest.approx(math.log(2.0))


def test_total_variation_is_half_the_l1_distance() -> None:
    got = total_variation([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]], [[0.25, 0.25, 0.5], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(got, [0.5, 1.0])


def test_mcnemar_exact_p_value() -> None:
    # 1 item only a got right, 5 only b got right, 4 concordant: p = 2 * P(X <= 1 | 6, 1/2)
    a = [1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    b = [0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    result = mcnemar_exact(a, b)
    assert (result.only_a, result.only_b) == (1, 5)
    assert result.p_value == pytest.approx(14 / 64)


def test_mcnemar_exact_caps_at_one_and_handles_no_discordance() -> None:
    assert mcnemar_exact([1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]).p_value == 1.0
    assert mcnemar_exact([1, 0], [1, 0]).p_value == 1.0


def test_paired_bootstrap_is_seeded() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=80), rng.normal(size=80)
    first = paired_bootstrap_ci(a, b, n_resamples=2_000, seed=7)
    assert paired_bootstrap_ci(a, b, n_resamples=2_000, seed=7) == first
    assert paired_bootstrap_ci(a, b, n_resamples=2_000, seed=8) != first
    assert first.estimate == pytest.approx(float(np.mean(a - b)))
    assert first.low < first.estimate < first.high
    assert first.method == "percentile"


def test_paired_bootstrap_of_a_constant_difference_is_a_point() -> None:
    a = np.linspace(0.0, 1.0, 50)
    ci = paired_bootstrap_ci(a + 0.25, a, n_resamples=500)
    assert (ci.low, ci.estimate, ci.high) == pytest.approx((0.25, 0.25, 0.25))
    assert ci.excludes_zero


def test_paired_bootstrap_resamples_whole_clusters() -> None:
    # every question contributes +1 and -1: item resampling varies, cluster resampling cannot
    diffs = np.tile([1.0, -1.0], 40)
    questions = np.repeat(np.arange(40), 2).tolist()
    zeros = np.zeros_like(diffs)
    by_item = paired_bootstrap_ci(diffs, zeros, n_resamples=1_000)
    by_question = paired_bootstrap_ci(diffs, zeros, clusters=questions, n_resamples=1_000)
    assert by_item.high - by_item.low > 0.1
    assert (by_question.low, by_question.high) == (0.0, 0.0)
    assert by_question.n_clusters == 40


def test_paired_bootstrap_bca_matches_scipy() -> None:
    rng = np.random.default_rng(3)
    a = rng.exponential(1.0, size=600)  # skewed, so BCa differs from percentile
    b = a * 0.5 + rng.normal(0.0, 0.3, size=600)
    ours = paired_bootstrap_ci(a, b, n_resamples=20_000, seed=0)
    assert ours.method == "bca"
    ref = stats.bootstrap(
        (a, b),
        lambda x, y, axis=-1: np.mean(x, axis=axis) - np.mean(y, axis=axis),
        paired=True,
        n_resamples=20_000,
        method="BCa",
        random_state=np.random.default_rng(1),
    ).confidence_interval
    width = ref.high - ref.low
    assert ours.low == pytest.approx(ref.low, abs=0.03 * width)
    assert ours.high == pytest.approx(ref.high, abs=0.03 * width)
