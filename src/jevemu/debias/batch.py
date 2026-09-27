"""Batch calibration (Zhou et al., 2023): the prior is the mean distribution over a batch of
questions that share an answer space; dividing it out removes the model's average lean."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np

from jevemu.debias.base import (
    DEFAULT_FIXED_TEXTS,
    DebiasOutcome,
    LabelPriors,
    Scorer,
    apply_prior,
    free_positions,
    mean_prior,
    passthrough,
    prior_by_key,
    prior_key,
)
from jevemu.scoring import ScoreResult
from jevemu.types import JSONish, Question

__all__ = ["BatchDebiaser", "fit_batch_priors"]


def fit_batch_priors(
    items: Iterable[tuple[str, Sequence[float], Sequence[bool]]], *, min_count: int = 1
) -> LabelPriors:
    """Batch priors from scored questions: ``(prior key, probabilities, free)`` each. A key's
    prior is the mean probability of each free position over its questions, renormalized;
    keys with fewer than ``min_count`` questions get no prior."""
    per_key: defaultdict[str, list[list[float]]] = defaultdict(list)
    for key, probs, free in items:
        mask = np.asarray(free, dtype=np.bool_)
        per_key[key].append(np.asarray(probs, dtype=np.float64)[mask].tolist())
    kept = {key: rows for key, rows in per_key.items() if len(rows) >= min_count}
    return LabelPriors(
        "batch",
        {key: tuple(mean_prior(rows).tolist()) for key, rows in kept.items()},
        {key: len(rows) for key, rows in kept.items()},
    )


class BatchDebiaser:
    """Divide each question by the batch prior of its prior key (no extra backend calls).

    The emulator answers one request at a time, so the batch is fitted beforehand from a
    recorded run (:func:`fit_batch_priors` on the strategy's raw probabilities, e.g. via
    ``scripts/calibrate.py priors``) and passed in. Questions without a prior pass through with
    a warning. Applies to every question type.
    """

    name = "batch"

    def __init__(
        self, priors: LabelPriors, *, fixed_texts: Sequence[str] = DEFAULT_FIXED_TEXTS
    ) -> None:
        self.priors = priors
        self.fixed_texts = tuple(fixed_texts)

    @property
    def id(self) -> str:
        return f"batch(fixed={list(self.fixed_texts)},priors=sha256:{self.priors.digest()})"

    async def debias(
        self, state: JSONish, question: Question, base: ScoreResult, score: Scorer
    ) -> DebiasOutcome:
        free = free_positions(question, self.fixed_texts)
        if sum(free) < 2:
            return passthrough(base)
        key = prior_key(question, free)
        prior = self.priors.priors.get(key)
        if prior is None:
            return passthrough(base, f"batch: no prior for {key}")
        logp = apply_prior(list(base.logprobs), prior, free)
        return DebiasOutcome(
            logprobs=tuple(logp.tolist()), prior=prior_by_key(question, free, prior)
        )
