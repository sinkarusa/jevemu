from __future__ import annotations

import json
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jevemu.calibrate import (
    Calibrator,
    FloatArray,
    HistogramCalibrator,
    IdentityCalibrator,
    IntArray,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    VectorCalibrator,
    probs_to_logits,
)

ALL_CALIBRATORS: tuple[type[Calibrator], ...] = (
    IdentityCalibrator,
    TemperatureCalibrator,
    VectorCalibrator,
    PlattCalibrator,
    IsotonicCalibrator,
    HistogramCalibrator,
)
NEG_INF = -math.inf


def log_softmax(z: FloatArray) -> FloatArray:
    shifted = z - z.max(axis=1, keepdims=True)
    out: FloatArray = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return out


def log(probs: list[list[float]] | FloatArray) -> FloatArray:
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(probs, dtype=np.float64))


def sample_labels(rng: np.random.Generator, true_logits: FloatArray) -> IntArray:
    """Correct keys drawn from ``softmax(true_logits)``; zero-probability keys are never drawn."""
    cumulative = np.exp(log_softmax(true_logits)).cumsum(axis=1)
    u = rng.random((cumulative.shape[0], 1)) * cumulative[:, -1:]
    return (cumulative <= u).sum(axis=1).astype(np.int64)


def synthetic(
    seed: int, n: int = 4000, k: int = 4, t_true: float = 2.0, mask_rate: float = 0.0
) -> tuple[FloatArray, IntArray]:
    """A model whose reported logits are ``t_true`` times the truth; some keys masked to -inf."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.5, size=(n, k))
    masked = rng.random((n, k)) < mask_rate
    masked[np.arange(n), rng.integers(0, k, size=n)] = False  # keep one key per row
    z[masked] = NEG_INF
    return log_softmax(z * t_true), sample_labels(rng, z)


def assert_distributions(probs: FloatArray, shape: tuple[int, ...]) -> None:
    assert probs.shape == shape
    assert np.isfinite(probs).all()
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=0, atol=1e-9)


FINITE_LOGITS = st.floats(-50.0, 50.0, allow_nan=False)


@st.composite
def logit_rows(draw: st.DrawFn, k: int | None = None) -> FloatArray:
    """(N, K) logits in [-50, 50] with some -inf entries, at least one finite entry per row."""
    n = draw(st.integers(1, 12))
    k = draw(st.integers(2, 6)) if k is None else k
    cells = st.one_of(FINITE_LOGITS, st.just(NEG_INF))
    z = np.array(draw(st.lists(cells, min_size=n * k, max_size=n * k))).reshape(n, k)
    for row in z:
        if not np.isfinite(row).any():
            row[draw(st.integers(0, k - 1))] = draw(FINITE_LOGITS)
    return z


@st.composite
def labelled_rows(draw: st.DrawFn) -> tuple[FloatArray, IntArray]:
    """Logits plus a correct key per row whose logit is finite (NLL fits need that)."""
    z = draw(logit_rows())
    y = [draw(st.sampled_from(np.flatnonzero(np.isfinite(row)).tolist())) for row in z]
    return z, np.array(y, dtype=np.int64)


# --- temperature ----------------------------------------------------------------------------


@pytest.mark.parametrize("t_true", [0.5, 1.0, 2.0, 3.5])
@pytest.mark.parametrize("mask_rate", [0.0, 0.3])
def test_temperature_recovers_known_t_within_5_percent(t_true: float, mask_rate: float) -> None:
    logp, y = synthetic(seed=7, n=20_000, t_true=t_true, mask_rate=mask_rate)
    fitted = TemperatureCalibrator().fit(logp, y).temperature
    assert fitted == pytest.approx(t_true, rel=0.05)


@given(logit_rows())
def test_temperature_one_is_the_identity(z: FloatArray) -> None:
    logp = log_softmax(z)
    out = TemperatureCalibrator(1.0).transform(logp)
    np.testing.assert_allclose(out, np.exp(logp), rtol=0, atol=1e-12)
    assert (out[np.isneginf(logp)] == 0.0).all()


def test_temperature_fit_ignores_per_row_logit_offsets() -> None:
    # probs_to_logits rows are not normalized; only within-row differences may matter.
    logp, y = synthetic(seed=3, n=2000, t_true=2.0)
    offsets = np.random.default_rng(0).normal(0.0, 20.0, size=(logp.shape[0], 1))
    t_normalized = TemperatureCalibrator().fit(logp, y).temperature
    t_offset = TemperatureCalibrator().fit(logp + offsets, y).temperature
    assert t_offset == pytest.approx(t_normalized, rel=1e-6)


@pytest.mark.parametrize("cls", [TemperatureCalibrator, VectorCalibrator])
def test_nll_fit_rejects_a_correct_key_with_zero_probability(cls: type[Calibrator]) -> None:
    logp = log([[0.7, 0.3, 0.0], [0.2, 0.5, 0.3]])
    with pytest.raises(ValueError, match="probs_to_logits"):
        cls().fit(logp, np.array([2, 1]))


# --- vector ---------------------------------------------------------------------------------


def test_vector_recovers_label_bias_and_temperature() -> None:
    rng = np.random.default_rng(11)
    truth = rng.normal(0.0, 1.5, size=(20_000, 4))
    y = sample_labels(rng, truth)
    bias = np.array([1.0, 0.0, -0.5, -0.5])  # the model under-rates key 0, over-rates 2 and 3
    fitted = VectorCalibrator().fit(2.0 * (truth - bias), y)
    assert fitted.temperature == pytest.approx(2.0, rel=0.05)
    assert fitted.bias is not None
    np.testing.assert_allclose(fitted.bias, bias - bias.mean(), atol=0.1)


def test_vector_bias_stays_finite_for_a_key_that_is_never_correct() -> None:
    logp, y = synthetic(seed=5, n=500, k=3)
    fitted = VectorCalibrator().fit(logp, np.where(y == 2, 0, y))
    assert fitted.bias is not None
    assert np.isfinite(fitted.bias).all()
    assert fitted.transform(logp)[:, 2].max() < 0.1


# --- platt ----------------------------------------------------------------------------------


def test_platt_binary_recovers_known_slope_and_intercept() -> None:
    rng = np.random.default_rng(2)
    s = rng.normal(0.0, 2.0, size=50_000)
    p_first = 1.0 / (1.0 + np.exp(-(0.6 * s + 0.4)))
    y = (rng.random(s.shape[0]) >= p_first).astype(np.int64)  # y == 0: the first key
    fitted = PlattCalibrator().fit(np.stack([s / 2.0, -s / 2.0], axis=1), y)
    assert fitted.a == pytest.approx(0.6, abs=0.05)
    assert fitted.b == pytest.approx(0.4, abs=0.05)


def test_platt_binary_corrects_a_yes_bias_across_one_half() -> None:
    # The model says P(yes) = 0.6 on every item, but only 20 of 100 are yes.
    logp = log(np.tile([0.6, 0.4], (100, 1)))
    y = np.array([0] * 20 + [1] * 80)
    out = PlattCalibrator().fit(logp, y).transform(logp)
    smoothed_yes_rate = (20 * 21 / 22 + 80 * 1 / 82) / 100  # Platt's targets
    assert out[0, 0] == pytest.approx(smoothed_yes_rate, abs=1e-4)
    assert out[0, 0] < out[0, 1]


def test_platt_binary_maps_zero_probability_to_a_finite_calibrated_value() -> None:
    logp, y = synthetic(seed=1, n=500, k=2)
    out = PlattCalibrator().fit(logp, y).transform(np.array([[0.0, NEG_INF], [NEG_INF, 0.0]]))
    assert_distributions(out, (2, 2))
    assert out[0, 0] > out[1, 0]


# --- top-label calibrators ------------------------------------------------------------------


def test_isotonic_replaces_p_max_by_accuracy_and_pools_violators() -> None:
    # 100 rows at p_max 0.9 with 60% correct and 100 rows at 0.6 with 80% correct violate
    # monotonicity, so both pool to 70%.
    high = log(np.tile([0.9, 0.05, 0.05], (100, 1)))
    low = log(np.tile([0.6, 0.2, 0.2], (100, 1)))
    y = np.array([0] * 60 + [1] * 40 + [0] * 80 + [2] * 20)
    fitted = IsotonicCalibrator().fit(np.concatenate([high, low]), y)
    out = fitted.transform(log([[0.9, 0.05, 0.05], [0.6, 0.2, 0.2]]))
    np.testing.assert_allclose(out, [[0.7, 0.15, 0.15], [0.7, 0.15, 0.15]], atol=1e-12)


@settings(deadline=None)
@given(
    st.integers(2, 6),
    st.lists(st.tuples(st.floats(0.0, 1.0), st.booleans()), min_size=1, max_size=60),
)
def test_isotonic_calibrated_confidence_is_non_decreasing(
    k: int, rows: list[tuple[float, bool]]
) -> None:
    def top_then_uniform(confidence: FloatArray) -> FloatArray:
        probs = np.repeat(((1.0 - confidence) / (k - 1))[:, None], k, axis=1)
        probs[:, 0] = confidence
        return log(probs)

    confidence = 1.0 / k + (1.0 - 1.0 / k) * np.array([c for c, _ in rows])
    y = np.array([0 if correct else 1 for _, correct in rows])
    fitted = IsotonicCalibrator().fit(top_then_uniform(confidence), y)
    calibrated = fitted.transform(top_then_uniform(np.linspace(1.0 / k, 1.0, 201)))[:, 0]
    assert (np.diff(calibrated) >= -1e-12).all()


def test_histogram_bins_hold_equal_mass_and_report_bin_accuracy() -> None:
    confidence = np.linspace(0.5, 0.99, 100)
    logp = log(np.stack([confidence, 1.0 - confidence], axis=1))
    y = np.zeros(100, dtype=np.int64)
    y[np.arange(100) % 5 == 0] = 1  # 80% correct in each of the three lower quarters ...
    y[75:] = 0  # ... and 100% in the top quarter
    out = HistogramCalibrator(n_bins=4).fit(logp, y).transform(logp)[:, 0]
    np.testing.assert_allclose(out[:75], 0.8, atol=1e-12)
    np.testing.assert_allclose(out[75:], 1.0, atol=1e-12)


def test_histogram_never_splits_tied_confidences_and_leaves_no_empty_bin() -> None:
    # Jev-like data: 70 answers at p_max = 1.0 (floored zeros) and 30 spread below.
    top = np.concatenate([np.ones(70), np.linspace(0.4, 0.95, 30)])
    logp = probs_to_logits(np.stack([top, (1.0 - top) / 2, (1.0 - top) / 2], axis=1))
    y = np.array([0] * 49 + [1] * 21 + [0] * 15 + [2] * 15)
    fitted = HistogramCalibrator(n_bins=10).fit(logp, y)
    out = fitted.transform(logp)[:, 0]
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[:70], 0.7, atol=1e-12)
    beyond = fitted.transform(np.array([[0.0, NEG_INF, NEG_INF], log([0.34, 0.33, 0.33])]))
    assert beyond[0, 0] == pytest.approx(0.7)  # above the largest fitted p_max: last bin
    assert beyond[1, 0] == pytest.approx(out[70])  # below the smallest: first bin


def test_top_label_redistributes_the_rest_proportionally() -> None:
    fitted = IsotonicCalibrator().fit(
        log(np.tile([0.9, 0.08, 0.02], (10, 1))), np.array([0] * 7 + [1] * 3)
    )
    out = fitted.transform(log([[0.9, 0.08, 0.02]]))
    np.testing.assert_allclose(out, [[0.7, 0.24, 0.06]], atol=1e-12)


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        # A calibrated 0.1 would drop key 0 below key 1: floored at 0.45 / (0.45 + 0.5).
        ([0.5, 0.45, 0.05], [0.45 / 0.95, 0.45 / 0.95, 0.05 / 0.95]),
        # No other key had mass: the remainder is shared uniformly, floored at 1 / K.
        ([1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3]),
    ],
)
def test_top_label_never_demotes_the_top_key(row: list[float], expected: list[float]) -> None:
    fitted = IsotonicCalibrator().fit(
        log(np.tile([0.5, 0.45, 0.05], (10, 1))), np.array([0] + [1] * 9)
    )
    np.testing.assert_allclose(fitted.transform(log([row])), [expected], atol=1e-12)


# --- invariants shared by every calibrator --------------------------------------------------


@settings(deadline=None, max_examples=60)
@given(labelled_rows(), st.data())
def test_fitted_calibrators_emit_distributions_without_nan(
    data: tuple[FloatArray, IntArray], draw: st.DataObject
) -> None:
    logp, y = data
    k = logp.shape[1]
    unseen = draw.draw(logit_rows(k=k))
    rows = np.arange(logp.shape[0])
    top_logit = logp.argmax(axis=1)
    top_prob = IdentityCalibrator().transform(logp).argmax(axis=1)
    for cls in ALL_CALIBRATORS:
        fitted = cls().fit(logp, y)
        out = fitted.transform(logp)
        assert_distributions(out, logp.shape)
        assert_distributions(fitted.transform(unseen), unseen.shape)
        if cls in (IdentityCalibrator, TemperatureCalibrator):
            assert (out[rows, top_logit] >= out.max(axis=1) - 1e-12).all()
            assert (out[np.isneginf(logp)] == 0.0).all()
        elif cls in (IsotonicCalibrator, HistogramCalibrator) or (cls is PlattCalibrator and k > 2):
            assert (out[rows, top_prob] >= out.max(axis=1) - 1e-12).all()


def fitted_examples() -> list[Calibrator]:
    logp4, y4 = synthetic(seed=21, n=300, k=4, t_true=2.5, mask_rate=0.2)
    logp2, y2 = synthetic(seed=22, n=300, k=2, t_true=0.7, mask_rate=0.2)
    return [
        IdentityCalibrator(),
        TemperatureCalibrator().fit(logp4, y4),
        VectorCalibrator(l2=0.01).fit(logp4, y4),
        PlattCalibrator().fit(logp4, y4),
        PlattCalibrator().fit(logp2, y2),
        IsotonicCalibrator().fit(logp4, y4),
        HistogramCalibrator(n_bins=7).fit(logp4, y4),
    ]


@pytest.mark.parametrize("calibrator", fitted_examples(), ids=lambda c: c.name)
def test_json_round_trip_is_exact(calibrator: Calibrator) -> None:
    document = calibrator.to_json()
    restored = type(calibrator).from_json(json.loads(json.dumps(document, allow_nan=False)))
    assert restored.to_json() == document
    k = 2 if document.get("n_classes") == 2 else 4
    probe, _ = synthetic(seed=99, n=50, k=k, t_true=1.3, mask_rate=0.3)
    assert np.array_equal(restored.transform(probe), calibrator.transform(probe))


def test_from_json_rejects_another_calibrators_document() -> None:
    with pytest.raises(ValueError, match="temperature"):
        TemperatureCalibrator.from_json({"name": "vector", "temperature": 2.0})


@pytest.mark.parametrize(
    "cls", [VectorCalibrator, PlattCalibrator, IsotonicCalibrator, HistogramCalibrator]
)
def test_key_count_calibrators_reject_a_different_k(cls: type[Calibrator]) -> None:
    logp, y = synthetic(seed=4, n=200, k=4)
    fitted = cls().fit(logp, y)
    with pytest.raises(ValueError, match="K=4"):
        fitted.transform(synthetic(seed=4, n=5, k=3)[0])


@pytest.mark.parametrize(
    ("logp", "message"),
    [
        ([[0.0, NEG_INF], [NEG_INF, NEG_INF]], "finite entry"),  # softmax would give NaN
        ([[0.0, math.nan]], "NaN"),
        ([[0.0, math.inf]], r"\+inf"),
        ([0.0, -1.0], r"shape \(N, K\)"),
        ([[0.0], [0.0]], "K >= 2"),
    ],
)
@pytest.mark.parametrize("cls", ALL_CALIBRATORS)
def test_invalid_logits_are_rejected(
    cls: type[Calibrator], logp: list[list[float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cls().fit(np.asarray(logp), np.zeros(np.asarray(logp).shape[0], dtype=np.int64))


@pytest.mark.parametrize(
    ("y", "message"),
    [([0, 2], r"\[0, 2\)"), ([0, -1], r"\[0, 2\)"), ([0], r"shape \(2,\)"), ([0.0, 1.0], "int")],
)
@pytest.mark.parametrize("cls", ALL_CALIBRATORS)
def test_invalid_labels_are_rejected(cls: type[Calibrator], y: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cls().fit(np.log([[0.6, 0.4], [0.3, 0.7]]), np.asarray(y))


# --- probs_to_logits ------------------------------------------------------------------------


def test_probs_to_logits_floors_exact_zeros_at_eps() -> None:
    np.testing.assert_array_equal(
        probs_to_logits([[0.88, 0.12, 0.0]]), np.log([[0.88, 0.12, 1e-4]])
    )
    assert probs_to_logits([0.0, 0.5], eps=0.01)[0] == pytest.approx(math.log(0.01))


def test_jev_probabilities_calibrate_without_infinite_nll() -> None:
    # Two-decimal Jev answers where the correct key was reported as exactly 0.
    probs = np.array([[0.88, 0.12, 0.0], [1.0, 0.0, 0.0], [0.61, 0.35, 0.04]])
    fitted = TemperatureCalibrator().fit(probs_to_logits(probs), np.array([2, 1, 0]))
    assert math.isfinite(fitted.temperature)
    assert fitted.temperature > 1.0


@pytest.mark.parametrize("bad", [[0.5, math.nan], [-0.1, 1.1], [0.2, 1.5]])
def test_probs_to_logits_rejects_non_probabilities(bad: list[float]) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        probs_to_logits(bad)
