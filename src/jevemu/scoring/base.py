"""Scoring contract: a strategy turns (state, question) into a distribution over answer keys.

A strategy owns its label scheme and prompt (echo scoring, for example, scores option text
rather than letters), so it receives the renderer rather than a pre-rendered prompt. All
probability maths is client-side and in log space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from jevemu.backends.base import Backend, Capabilities
from jevemu.render.renderer import PromptRenderer
from jevemu.types import JSONish, Question


@dataclass(frozen=True)
class ScoreResult:
    """Distribution over a question's answer keys plus how it was obtained.

    ``keys`` are in question order: Choice option keys, Score levels ``"0"``..``"K-1"``, Noul
    ``("true", "false")``.
    """

    keys: tuple[str, ...]
    logprobs: tuple[float, ...]
    """Log-probabilities renormalized over ``keys`` (logsumexp == 0); ``-inf`` allowed."""
    raw_logprobs: tuple[float, ...]
    """Before renormalization, as observed (merged over surface variants)."""
    observed_mass: float
    """Sum of valid-label probability before renormalization (1.0 when the mask is reflected)."""
    missing: tuple[str, ...]
    """Keys whose label was absent from the backend output."""
    truncated: bool
    """Some key's logprob is an upper-bound estimate rather than an observation."""
    strategy: str
    label_scheme: str
    n_backend_calls: int
    prompt_tokens: int
    cached_tokens: int | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not (len(self.keys) == len(self.logprobs) == len(self.raw_logprobs)):
            raise ValueError("keys, logprobs and raw_logprobs must have equal length")

    @property
    def probabilities(self) -> dict[str, float]:
        return {k: math.exp(lp) for k, lp in zip(self.keys, self.logprobs, strict=True)}


class ScoringStrategy(Protocol):
    name: str

    def supports(self, capabilities: Capabilities, question: Question) -> bool:
        """Whether this strategy can score ``question`` exactly on a backend with these caps."""
        ...

    async def score(
        self, backend: Backend, renderer: PromptRenderer, state: JSONish, question: Question
    ) -> ScoreResult: ...
