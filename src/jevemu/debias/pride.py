"""PriDe (Zheng et al., ICLR 2024): estimate a prior over answer positions from cyclically
permuted presentations of a few questions, then divide it out of every question."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence

from jevemu.debias.base import (
    DEFAULT_FIXED_TEXTS,
    DebiasOutcome,
    LabelPriors,
    Presentation,
    Scorer,
    apply_prior,
    average_presentations,
    cyclic_orders,
    free_positions,
    item_prior,
    mean_prior,
    passthrough,
    prior_by_key,
    prior_key,
    score_orders,
)
from jevemu.debias.permutation import DEFAULT_MAX_OPTIONS
from jevemu.scoring import ScoreResult
from jevemu.types import ChoiceQuestion, JSONish, Question

__all__ = ["DEFAULT_ALPHA", "PriDeDebiaser", "fit_pride_priors", "selected_for_estimation"]

DEFAULT_ALPHA = 0.05
"""Share of questions (per prior key) permuted to estimate the prior."""


def selected_for_estimation(state: JSONish, question: Question, alpha: float, seed: int) -> bool:
    """Whether a question is one of the ``alpha`` share used for estimation: a seeded
    SHA-256 of (state, question) below ``alpha``, independent of arrival order."""
    payload = f"{seed}\x1f{question.model_dump_json()}\x1f{_state_json(state)}"
    value = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
    return value < alpha * 2.0**64


def _state_json(state: JSONish) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fit_pride_priors(
    items: Iterable[tuple[str, Sequence[Presentation], Sequence[bool]]], *, min_count: int = 1
) -> LabelPriors:
    """PriDe priors from recorded permuted questions: ``(prior key, presentations, free)``
    per question, the presentations being all cyclic shifts of its free options. Each key's
    prior is the mean of its questions' :func:`~jevemu.debias.base.item_prior`; keys with
    fewer than ``min_count`` questions get no prior."""
    per_key: defaultdict[str, list[Sequence[float]]] = defaultdict(list)
    for key, presentations, free in items:
        per_key[key].append(item_prior(presentations, free).tolist())
    kept = {key: priors for key, priors in per_key.items() if len(priors) >= min_count}
    return LabelPriors(
        "pride",
        {key: tuple(mean_prior(priors).tolist()) for key, priors in kept.items()},
        {key: len(priors) for key, priors in kept.items()},
    )


class PriDeDebiaser:
    """Divide each Choice question's distribution by a prior over its free positions.

    - With ``priors`` (fitted offline, e.g. :func:`fit_pride_priors` on a recorded run), a
      question whose :func:`~jevemu.debias.base.prior_key` has a prior is debiased at no extra
      cost.
    - Otherwise the prior is estimated online: the ``alpha`` share of questions picked by
      :func:`selected_for_estimation`, and the first question of each prior key, are scored
      under all ``F`` cyclic shifts of their ``F`` free options (``F - 1`` extra passes). Such a
      question contributes its :func:`~jevemu.debias.base.item_prior` and is answered with the
      permutation average; every other question is divided by the mean of the estimates so
      far. Online estimates depend on arrival order, so for reproducible answers fit priors
      first and pass them in. Expected cost: about ``x(1 + alpha (F - 1))``.

    Score and Noul questions, and Choice questions with fewer than 2 or more than
    ``max_options`` free options, pass through.
    """

    name = "pride"

    def __init__(
        self,
        *,
        alpha: float = DEFAULT_ALPHA,
        priors: LabelPriors | None = None,
        seed: int = 0,
        max_options: int | None = DEFAULT_MAX_OPTIONS,
        fixed_texts: Sequence[str] = DEFAULT_FIXED_TEXTS,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
        if max_options is not None and max_options < 2:
            raise ValueError(f"max_options must be >= 2, got {max_options}")
        self.alpha = alpha
        self.priors = priors
        self.seed = seed
        self.max_options = max_options
        self.fixed_texts = tuple(fixed_texts)
        self._estimates: defaultdict[str, list[Sequence[float]]] = defaultdict(list)

    @property
    def id(self) -> str:
        frozen = "none" if self.priors is None else f"sha256:{self.priors.digest()}"
        return (
            f"pride(alpha={self.alpha},seed={self.seed},max_options={self.max_options},"
            f"fixed={list(self.fixed_texts)},priors={frozen})"
        )

    def estimated_priors(self) -> LabelPriors:
        """The online estimates so far (keys with fitted ``priors`` are not estimated)."""
        return LabelPriors(
            "pride",
            {key: tuple(mean_prior(est).tolist()) for key, est in self._estimates.items() if est},
            {key: len(est) for key, est in self._estimates.items() if est},
        )

    async def debias(
        self, state: JSONish, question: Question, base: ScoreResult, score: Scorer
    ) -> DebiasOutcome:
        if not isinstance(question, ChoiceQuestion):
            return passthrough(base)
        free = free_positions(question, self.fixed_texts)
        n_free = sum(free)
        if n_free < 2:
            return passthrough(base)
        if self.max_options is not None and n_free > self.max_options:
            return passthrough(
                base, f"pride: {n_free} free options > max_options={self.max_options}"
            )
        key = prior_key(question, free)
        fitted = None if self.priors is None else self.priors.priors.get(key)
        if fitted is not None:
            return self._divide(question, base, free, fitted)
        estimates = self._estimates[key]
        if estimates and not selected_for_estimation(state, question, self.alpha, self.seed):
            return self._divide(question, base, free, mean_prior(estimates).tolist())
        presentations, calls, tokens = await score_orders(
            state, question, cyclic_orders(free), base, score
        )
        estimates.append(item_prior(presentations, free).tolist())
        return DebiasOutcome(
            logprobs=tuple(average_presentations(presentations).tolist()),
            extra_calls=calls,
            extra_prompt_tokens=tokens,
            presentations=tuple(presentations),
        )

    def _divide(
        self,
        question: ChoiceQuestion,
        base: ScoreResult,
        free: Sequence[bool],
        prior: Sequence[float],
    ) -> DebiasOutcome:
        logp = apply_prior(list(base.logprobs), prior, free)
        return DebiasOutcome(
            logprobs=tuple(logp.tolist()), prior=prior_by_key(question, free, prior)
        )
