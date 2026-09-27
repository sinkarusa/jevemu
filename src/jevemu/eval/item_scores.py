"""Per-item scores of typed answers, per-benchmark summaries and macro averages.

Every answer's probabilities are first renormalized to sum to 1 (Jev rounds each to 0.01, so
its rows do not sum to exactly 1). Per question type:

- **choice**: correct iff ``choice == gold``; choosing the "I don't know" option (``idk_key``)
  is wrong and also counted in the IDK rate. ``p_gold``, NLL ``-log max(p_gold, eps)`` and the
  multiclass Brier ``sum_k (p_k - [k == gold])^2`` over the answer's keys; the confidence for
  ECE is the probability of ``choice``.
- **noul**: predicts yes iff ``P(yes) >= 0.5``; correct iff that matches gold. ``p_gold`` is
  ``P(yes)`` or ``1 - P(yes)``; Brier over the two outcomes is ``2 (P(yes) - y)^2``; the
  confidence is the probability of the predicted outcome.
- **score**: predicts the argmax level (the lowest on ties); correct iff it is the gold level.
  NLL and Brier over the levels; confidence is the predicted level's probability. Also the
  absolute error of the reported expected score ``score`` against the gold level, and whether
  the predicted level is within one of gold.

Summaries use :mod:`jevemu.eval.metrics`: ECE is top-label with 10 equal-mass bins; a
benchmark's accuracy interval is :func:`~jevemu.eval.metrics.paired_bootstrap_ci` against zero
(BCa from 500 items, percentile below). Macro averages weigh every benchmark equally; their
intervals come from a stratified bootstrap (items resampled within each benchmark, benchmark
means averaged, percentile interval), which also serves paired differences when fed per-item
differences.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jevemu.bench.spec import Gold, QuestionType
from jevemu.eval.metrics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    NLL_EPS,
    ece,
    paired_bootstrap_ci,
)
from jevemu.types import ChoiceAnswer, NoulAnswer, ScoreAnswer

__all__ = [
    "Interval",
    "ItemScore",
    "ScoreSummary",
    "ScoreTable",
    "macro_intervals",
    "score_answer",
    "summarize_scores",
]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ItemScore:
    correct: bool
    p_gold: float
    """Renormalized probability of the gold outcome."""
    nll: float
    brier: float
    confidence: float
    """Renormalized probability of the predicted outcome (ECE's confidence)."""
    abstained: bool
    """Choice only: the answer is the "I don't know" option."""
    abs_error: float | None = None
    """Score only: ``|score - gold|``."""
    within_one: bool | None = None
    """Score only: the predicted level is at most one level from gold."""


def _normalized(probabilities: Mapping[str, float]) -> dict[str, float]:
    total = math.fsum(probabilities.values())
    if total <= 0.0:
        return {key: 0.0 for key in probabilities}
    return {key: value / total for key, value in probabilities.items()}


def _log_loss(p_gold: float, eps: float) -> float:
    return -math.log(min(1.0, max(p_gold, eps)))


def _brier(probabilities: Mapping[str, float], gold: str) -> float:
    return math.fsum((p - (key == gold)) ** 2 for key, p in probabilities.items()) + (
        0.0 if gold in probabilities else 1.0
    )


def score_answer(
    answer: ChoiceAnswer | ScoreAnswer | NoulAnswer,
    gold: Gold,
    *,
    idk_key: str | None = None,
    eps: float = NLL_EPS,
) -> ItemScore:
    """Score one answer against ``gold`` (module docstring). ``gold`` is a criteria key for
    choice, a bool for noul and a level index for score."""
    if isinstance(answer, NoulAnswer):
        if not isinstance(gold, bool):
            raise TypeError(f"noul gold must be a bool, got {gold!r}")
        p_yes = answer.noul
        p_gold = p_yes if gold else 1.0 - p_yes
        predicted = p_yes >= 0.5
        return ItemScore(
            correct=predicted == gold,
            p_gold=p_gold,
            nll=_log_loss(p_gold, eps),
            brier=2.0 * (p_yes - float(gold)) ** 2,
            confidence=p_yes if predicted else 1.0 - p_yes,
            abstained=False,
        )
    probabilities = _normalized(answer.probabilities)
    if isinstance(answer, ChoiceAnswer):
        if not isinstance(gold, str):
            raise TypeError(f"choice gold must be a criteria key, got {gold!r}")
        p_gold = probabilities.get(gold, 0.0)
        return ItemScore(
            correct=answer.choice == gold,
            p_gold=p_gold,
            nll=_log_loss(p_gold, eps),
            brier=_brier(probabilities, gold),
            confidence=probabilities[answer.choice],
            abstained=idk_key is not None and answer.choice == idk_key,
        )
    if isinstance(gold, bool) or not isinstance(gold, int):
        raise TypeError(f"score gold must be a level index, got {gold!r}")
    levels = [probabilities[str(i)] for i in range(len(answer.legend))]
    level = int(np.argmax(levels))
    key = str(gold)
    p_gold = probabilities.get(key, 0.0)
    return ItemScore(
        correct=level == gold,
        p_gold=p_gold,
        nll=_log_loss(p_gold, eps),
        brier=_brier(probabilities, key),
        confidence=levels[level],
        abstained=False,
        abs_error=abs(answer.score - gold),
        within_one=abs(level - gold) <= 1,
    )


@dataclass(frozen=True)
class ScoreTable:
    """The item scores of one benchmark as aligned arrays (``abs_error``/``within_one`` are
    NaN/False outside score questions)."""

    question_type: QuestionType
    idk: bool
    """The benchmark offers an "I don't know" option."""
    item_ids: tuple[str, ...]
    correct: BoolArray
    p_gold: FloatArray
    nll: FloatArray
    brier: FloatArray
    confidence: FloatArray
    abstained: BoolArray
    abs_error: FloatArray
    within_one: BoolArray

    @classmethod
    def from_scores(
        cls,
        question_type: QuestionType,
        item_ids: Sequence[str],
        scores: Sequence[ItemScore],
        *,
        idk: bool = False,
    ) -> ScoreTable:
        if len(item_ids) != len(scores):
            raise ValueError(f"{len(item_ids)} item ids for {len(scores)} scores")
        return cls(
            question_type=question_type,
            idk=idk,
            item_ids=tuple(item_ids),
            correct=np.array([s.correct for s in scores], dtype=np.bool_),
            p_gold=np.array([s.p_gold for s in scores], dtype=np.float64),
            nll=np.array([s.nll for s in scores], dtype=np.float64),
            brier=np.array([s.brier for s in scores], dtype=np.float64),
            confidence=np.array([s.confidence for s in scores], dtype=np.float64),
            abstained=np.array([s.abstained for s in scores], dtype=np.bool_),
            abs_error=np.array(
                [math.nan if s.abs_error is None else s.abs_error for s in scores],
                dtype=np.float64,
            ),
            within_one=np.array([bool(s.within_one) for s in scores], dtype=np.bool_),
        )

    def __len__(self) -> int:
        return len(self.item_ids)

    def take(self, item_ids: Sequence[str]) -> ScoreTable:
        """The rows of ``item_ids``, in that order."""
        position = {item_id: i for i, item_id in enumerate(self.item_ids)}
        rows = np.array([position[item_id] for item_id in item_ids], dtype=np.int64)
        return ScoreTable(
            question_type=self.question_type,
            idk=self.idk,
            item_ids=tuple(item_ids),
            correct=self.correct[rows],
            p_gold=self.p_gold[rows],
            nll=self.nll[rows],
            brier=self.brier[rows],
            confidence=self.confidence[rows],
            abstained=self.abstained[rows],
            abs_error=self.abs_error[rows],
            within_one=self.within_one[rows],
        )


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.estimate:.4f} [{self.low:.4f}, {self.high:.4f}]"


@dataclass(frozen=True)
class ScoreSummary:
    n: int
    accuracy: Interval
    idk_rate: float | None
    """IDK benchmarks only: share of answers that chose "I don't know"."""
    mean_p_gold: float
    nll: float
    brier: float
    ece: float
    mae: float | None
    """Score only: mean ``|score - gold|``."""
    within_one: float | None
    """Score only: share of predicted levels at most one from gold."""


def _mean(values: FloatArray | BoolArray) -> float:
    return float(np.mean(values)) if len(values) else math.nan


def summarize_scores(
    table: ScoreTable,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 0,
    confidence_level: float = 0.95,
) -> ScoreSummary:
    """Accuracy with its bootstrap interval and the mean metrics of one benchmark."""
    n = len(table)
    if n:
        correct = table.correct.astype(np.float64)
        ci = paired_bootstrap_ci(
            correct,
            np.zeros(n),
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            seed=seed,
        )
        accuracy = Interval(ci.estimate, ci.low, ci.high)
    else:
        accuracy = Interval(math.nan, math.nan, math.nan)
    is_score = table.question_type == "score"
    return ScoreSummary(
        n=n,
        accuracy=accuracy,
        idk_rate=_mean(table.abstained) if table.idk else None,
        mean_p_gold=_mean(table.p_gold),
        nll=_mean(table.nll),
        brier=_mean(table.brier),
        ece=ece(table.confidence, table.correct),
        mae=_mean(table.abs_error) if is_score else None,
        within_one=_mean(table.within_one) if is_score else None,
    )


def _resampled_means(values: FloatArray, n_resamples: int, rng: np.random.Generator) -> FloatArray:
    """``(n_resamples, m)`` column means of ``values`` (``(n, m)``) over bootstrap resamples."""
    n, m = values.shape
    out = np.empty((n_resamples, m), dtype=np.float64)
    step = max(1, 2_000_000 // max(1, n * m))  # bounds each gathered block to ~16 MB
    for start in range(0, n_resamples, step):
        idx = rng.integers(0, n, size=(min(step, n_resamples - start), n))
        out[start : start + len(idx)] = values[idx].mean(axis=1)
    return out


def macro_intervals(
    columns: Mapping[str, FloatArray],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 0,
    confidence_level: float = 0.95,
) -> list[Interval]:
    """Equal-weight mean over benchmarks of each column's per-benchmark mean, with a stratified
    percentile bootstrap interval.

    ``columns`` maps benchmark -> ``(n_b, m)`` per-item values (one column per metric, e.g.
    correctness and NLL, or paired differences of them). Returns ``m`` intervals. The same
    ``seed`` gives the same intervals.
    """
    if not columns:
        raise ValueError("macro_intervals needs at least one benchmark")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must lie in (0, 1), got {confidence_level}")
    widths = {np.shape(values)[1] for values in columns.values()}
    if len(widths) != 1:
        raise ValueError(f"every benchmark needs the same number of columns, got {widths}")
    (m,) = widths
    rng = np.random.default_rng(seed)
    boot = np.zeros((n_resamples, m), dtype=np.float64)
    estimate = np.zeros(m, dtype=np.float64)
    for name in sorted(columns):
        values = np.asarray(columns[name], dtype=np.float64)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError(f"{name}: values must be a non-empty (n, m) array")
        estimate += values.mean(axis=0)
        boot += _resampled_means(values, n_resamples, rng)
    estimate /= len(columns)
    boot /= len(columns)
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(boot, [alpha, 1.0 - alpha], axis=0)
    return [
        Interval(float(e), float(lo), float(hi))
        for e, lo, hi in zip(estimate, low, high, strict=True)
    ]
