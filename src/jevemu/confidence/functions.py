"""Confidence functions: collapse an answer's probability vector into one number in [0, 1].

Jev returns ``confidence`` on Choice and Score answers without specifying the formula; its docs
give ``(K * p_max - 1) / (K - 1)`` clamped to [0, 1] as an approximation. A fit on 18,206 Jev
answers (:mod:`jevemu.confidence.fit`, ``docs/research/calibration.md``) recovered the formula:
:func:`mode_distance` reproduces it for both question types (for Choice it equals
``peak_linear``; for Score it measures how far the probability mass sits from the modal level).

Every function takes the probabilities in key order, uses them as given (no renormalization),
needs ``K >= 2`` non-negative entries, and accepts ``ordinal``: ``True`` for Score answers,
whose levels are ordered, ``False`` for Choice. Only :func:`mode_distance` uses it; the others
are invariant to the order of the probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol


class ConfidenceFn(Protocol):
    """Probabilities in key order -> confidence in [0, 1]; ``ordinal`` for Score levels."""

    def __call__(self, probs: Sequence[float], *, ordinal: bool = False) -> float: ...


def _checked(probs: Sequence[float]) -> list[float]:
    values = [float(p) for p in probs]
    if len(values) < 2:
        raise ValueError(f"confidence needs at least 2 probabilities, got {len(values)}")
    if any(not (p >= 0.0 and math.isfinite(p)) for p in values):
        raise ValueError(f"probabilities must be finite and >= 0, got {values}")
    return values


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


def peak_linear(probs: Sequence[float], *, ordinal: bool = False) -> float:
    """``clip((K * p_max - 1) / (K - 1), 0, 1)``: 0 for a uniform vector, 1 for one-hot.

    The clip matters for Jev's two-decimal probabilities, whose maximum can fall below 1/K
    (e.g. ``[0.33, 0.33, 0.33]``).
    """
    values = _checked(probs)
    k = len(values)
    return _clip01((k * max(values) - 1.0) / (k - 1))


def max_prob(probs: Sequence[float], *, ordinal: bool = False) -> float:
    """``p_max``."""
    return _clip01(max(_checked(probs)))


def margin(probs: Sequence[float], *, ordinal: bool = False) -> float:
    """``p1 - p2``: the gap between the two largest probabilities."""
    first, second = sorted(_checked(probs), reverse=True)[:2]
    return _clip01(first - second)


def one_minus_norm_entropy(probs: Sequence[float], *, ordinal: bool = False) -> float:
    """``1 - H(p) / log K`` with ``0 log 0 = 0``: 0 for a uniform vector, 1 for one-hot."""
    values = _checked(probs)
    entropy = -sum(p * math.log(p) for p in values if p > 0.0)
    return _clip01(1.0 - entropy / math.log(len(values)))


def uniform_spread(k: int, *, ordinal: bool) -> float:
    """``min_c E|X - c|`` for ``X`` uniform on ``K`` positions: ``(K - 1) / K`` under the 0/1
    distance, ``K / 4`` (even ``K``) or ``(K^2 - 1) / (4K)`` (odd ``K``) under ``|i - j|``."""
    if k < 2:
        raise ValueError(f"K must be >= 2, got {k}")
    if not ordinal:
        return (k - 1) / k
    return k / 4 if k % 2 == 0 else (k * k - 1) / (4 * k)


def mode_distance(probs: Sequence[float], *, ordinal: bool = False) -> float:
    """``clip(1 - E[d(X, mode)] / uniform_spread(K), 0, 1)``: the fit of Jev's confidence.

    ``d`` is the 0/1 distance for Choice (then this is exactly :func:`peak_linear`) and the
    level distance ``|i - j|`` for Score (``ordinal=True``), so mass one level from the mode
    costs less than mass four levels away. The mode is the first maximum. The normalizer is
    the spread of a uniform distribution around its best center, so a uniform vector scores
    about 0 (exactly 0 for Choice and for Score with the mode at a median) and one-hot scores 1.
    """
    values = _checked(probs)
    k = len(values)
    mode = values.index(max(values))
    if ordinal:
        spread = math.fsum(p * abs(i - mode) for i, p in enumerate(values))
    else:
        spread = 1.0 - values[mode]  # E[d] for a distribution summing to 1
    return _clip01(1.0 - spread / uniform_spread(k, ordinal=ordinal))


DEFAULT_CONFIDENCE: ConfidenceFn = mode_distance
"""The emulator's default: the fit of Jev's ``confidence`` (``docs/research/calibration.md``)."""

CONFIDENCE_FUNCTIONS: dict[str, ConfidenceFn] = {
    "peak_linear": peak_linear,
    "max_prob": max_prob,
    "margin": margin,
    "one_minus_norm_entropy": one_minus_norm_entropy,
    "mode_distance": mode_distance,
}
"""Confidence functions by name (for configs)."""
