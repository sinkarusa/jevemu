"""Contextual calibration (Zhao et al., ICML 2021): score each question once with a
content-free state, and divide that distribution out of every answer to the question."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

import numpy as np

from jevemu.debias.base import (
    DEFAULT_FIXED_TEXTS,
    DebiasOutcome,
    Scorer,
    apply_prior,
    free_positions,
    mean_prior,
    passthrough,
    prior_by_key,
)
from jevemu.scoring import ScoreResult
from jevemu.types import JSONish, Question

__all__ = ["DEFAULT_CONTENT_FREE", "ContextualDebiaser"]

DEFAULT_CONTENT_FREE: tuple[JSONish, ...] = ("N/A",)


class ContextualDebiaser:
    """Prior = mean distribution of the question over the ``content_free`` states.

    Applies to every question type (Choice free options, Score levels, Noul yes/no). The prior
    is computed once per distinct question (instructions and criteria; the state is what
    varies) and cached for the debiaser's lifetime: ``len(content_free)`` extra scoring passes
    for the first request with a question, none afterwards. A question whose options change
    per item (a multiple-choice bank) therefore costs one extra pass per item.
    """

    name = "contextual"

    def __init__(
        self,
        *,
        content_free: Sequence[JSONish] = DEFAULT_CONTENT_FREE,
        fixed_texts: Sequence[str] = DEFAULT_FIXED_TEXTS,
    ) -> None:
        if not content_free:
            raise ValueError("content_free needs at least one state")
        self.content_free = tuple(content_free)
        self.fixed_texts = tuple(fixed_texts)
        self._priors: dict[str, list[float]] = {}
        self._pending: dict[str, asyncio.Task[tuple[list[float], int, int]]] = {}

    @property
    def id(self) -> str:
        states = json.dumps(list(self.content_free), ensure_ascii=False, separators=(",", ":"))
        return f"contextual(content_free={states},fixed={list(self.fixed_texts)})"

    async def debias(
        self, state: JSONish, question: Question, base: ScoreResult, score: Scorer
    ) -> DebiasOutcome:
        free = free_positions(question, self.fixed_texts)
        if sum(free) < 2:
            return passthrough(base)
        cache_key = question.model_dump_json()
        calls = tokens = 0
        cached = self._priors.get(cache_key)
        if cached is None:
            task = self._pending.get(cache_key)
            owner = task is None
            if task is None:
                task = asyncio.ensure_future(self._estimate(question, free, score))
                self._pending[cache_key] = task
            try:
                # Shielded: a cancelled request must not cancel the estimate others await.
                prior, spent_calls, spent_tokens = await asyncio.shield(task)
            finally:
                if owner:
                    self._pending.pop(cache_key, None)
            self._priors[cache_key] = prior
            if owner:  # the request that paid for the estimate reports its cost
                calls, tokens = spent_calls, spent_tokens
        else:
            prior = cached
        logp = apply_prior(list(base.logprobs), prior, free)
        return DebiasOutcome(
            logprobs=tuple(logp.tolist()),
            extra_calls=calls,
            extra_prompt_tokens=tokens,
            prior=prior_by_key(question, free, prior),
        )

    async def _estimate(
        self, question: Question, free: Sequence[bool], score: Scorer
    ) -> tuple[list[float], int, int]:
        results = await asyncio.gather(*(score(state, question) for state in self.content_free))
        mask = np.asarray(free, dtype=np.bool_)
        prior = mean_prior(np.exp(np.asarray(r.logprobs))[mask].tolist() for r in results)
        calls = sum(r.n_backend_calls for r in results)
        tokens = sum(r.prompt_tokens for r in results)
        return prior.tolist(), calls, tokens
