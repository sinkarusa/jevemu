"""Fit Jev's confidence as a function of its probability vector.

Samples are Jev's Choice and Score answers (Noul answers carry no confidence). Probabilities
are renormalized to sum to 1 (Jev rounds each to 0.01); a candidate's prediction is rounded to
0.01, as Jev rounds ``confidence``, before it is compared with the reported value.

Candidates are every function of :data:`~jevemu.confidence.CONFIDENCE_FUNCTIONS` plus a small
fitted monotone family, :class:`PowerConfidence`: ``base(p) ** gamma`` with one ``gamma`` per
question kind, fitted by least squares on the unrounded prediction. :func:`evaluate` reports
the mean absolute error per (kind, K), the share of exact matches and of matches within 0.01,
and the mean signed error.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize_scalar

from jevemu.confidence.functions import CONFIDENCE_FUNCTIONS, ConfidenceFn
from jevemu.types import Answer, ChoiceAnswer, ScoreAnswer

__all__ = [
    "ConfidenceSample",
    "FitRow",
    "Kind",
    "PowerConfidence",
    "evaluate",
    "fit_power",
    "sample_from_answer",
]

Kind = Literal["choice", "score"]
KINDS: tuple[Kind, ...] = ("choice", "score")
ROUND_TO = 0.01
_EXACT = 0.005
"""Absolute errors below this are exact matches (both values are multiples of 0.01)."""


@dataclass(frozen=True)
class ConfidenceSample:
    benchmark: str
    kind: Kind
    probabilities: tuple[float, ...]
    """Renormalized, in key order (Score: levels ``0``..``K-1``)."""
    reported: float

    @property
    def k(self) -> int:
        return len(self.probabilities)


def sample_from_answer(benchmark: str, answer: Answer) -> ConfidenceSample | None:
    """The sample of a Choice or Score answer; ``None`` for Noul or an all-zero vector."""
    if isinstance(answer, ChoiceAnswer):
        kind: Kind = "choice"
        values = list(answer.probabilities.values())
    elif isinstance(answer, ScoreAnswer):
        kind = "score"
        values = [answer.probabilities[str(i)] for i in range(len(answer.legend))]
    else:
        return None
    total = math.fsum(values)
    if total <= 0.0:
        return None
    return ConfidenceSample(
        benchmark, kind, tuple(v / total for v in values), float(answer.confidence)
    )


def _round(value: float) -> float:
    return round(round(value / ROUND_TO) * ROUND_TO, 12)


def predict(fn: ConfidenceFn, sample: ConfidenceSample) -> float:
    """``fn``'s confidence for ``sample``, rounded to 0.01 like Jev's."""
    return _round(fn(sample.probabilities, ordinal=sample.kind == "score"))


class PowerConfidence:
    """``base(p) ** gamma[kind]``: a monotone reshaping of ``base`` that keeps 0 and 1."""

    def __init__(self, base: str, gammas: Mapping[Kind, float]) -> None:
        if base not in CONFIDENCE_FUNCTIONS:
            raise ValueError(f"unknown base {base!r}; known: {sorted(CONFIDENCE_FUNCTIONS)}")
        if any(not (math.isfinite(g) and g > 0.0) for g in gammas.values()):
            raise ValueError(f"gammas must be finite and > 0, got {dict(gammas)}")
        self.base = base
        self.gammas = dict(gammas)
        self.__name__ = f"power({base})"

    def __call__(self, probs: Sequence[float], *, ordinal: bool = False) -> float:
        value = CONFIDENCE_FUNCTIONS[self.base](probs, ordinal=ordinal)
        return float(value ** self.gammas.get("score" if ordinal else "choice", 1.0))


def fit_power(samples: Sequence[ConfidenceSample], base: str) -> PowerConfidence:
    """Fit :class:`PowerConfidence` on ``samples``: per kind, the ``gamma`` in [0.2, 5]
    minimizing the squared error of the unrounded prediction."""
    fn = CONFIDENCE_FUNCTIONS[base]
    gammas: dict[Kind, float] = {}
    for kind in KINDS:
        rows = [s for s in samples if s.kind == kind]
        if not rows:
            continue
        x = np.array([fn(s.probabilities, ordinal=kind == "score") for s in rows])
        y = np.array([s.reported for s in rows])

        def loss(log_gamma: float, x: np.ndarray = x, y: np.ndarray = y) -> float:
            return float(np.mean((x ** math.exp(log_gamma) - y) ** 2))

        result = minimize_scalar(loss, bounds=(math.log(0.2), math.log(5.0)), method="bounded")
        gammas[kind] = math.exp(float(result.x))
    return PowerConfidence(base, gammas)


@dataclass(frozen=True)
class FitRow:
    candidate: str
    kind: Kind | Literal["all"]
    k: int | None
    """``None`` for a total over every K."""
    n: int
    mae: float
    exact: float
    """Share of predictions equal to the reported confidence."""
    within_001: float
    """Share within 0.01."""
    bias: float
    """Mean of reported minus predicted."""


def _row(candidate: str, kind: Kind | Literal["all"], k: int | None, err: np.ndarray) -> FitRow:
    return FitRow(
        candidate,
        kind,
        k,
        int(err.shape[0]),
        float(np.mean(np.abs(err))),
        float(np.mean(np.abs(err) < _EXACT)),
        float(np.mean(np.abs(err) < ROUND_TO + _EXACT)),
        float(np.mean(err)),
    )


def evaluate(
    candidates: Mapping[str, ConfidenceFn], samples: Iterable[ConfidenceSample]
) -> list[FitRow]:
    """Per candidate: one row per (kind, K), one per kind and one over all samples."""
    rows = list(samples)
    if not rows:
        raise ValueError("evaluate needs at least one sample")
    out: list[FitRow] = []
    for name, fn in candidates.items():
        errors = np.array([s.reported - predict(fn, s) for s in rows])
        groups: defaultdict[tuple[Kind, int], list[int]] = defaultdict(list)
        for index, sample in enumerate(rows):
            groups[(sample.kind, sample.k)].append(index)
        for kind, k in sorted(groups):
            out.append(_row(name, kind, k, errors[groups[(kind, k)]]))
        for kind in KINDS:
            mask = np.array([s.kind == kind for s in rows])
            if mask.any():
                out.append(_row(name, kind, None, errors[mask]))
        out.append(_row(name, "all", None, errors))
    return out
