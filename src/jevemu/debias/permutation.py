"""Permutation debiasing: average the option probabilities over reordered presentations."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import Literal

from jevemu.debias.base import (
    DEFAULT_FIXED_TEXTS,
    DebiasOutcome,
    Scorer,
    average_presentations,
    cyclic_orders,
    free_positions,
    passthrough,
    score_orders,
)
from jevemu.scoring import ScoreResult
from jevemu.types import ChoiceQuestion, JSONish, Question

__all__ = ["DEFAULT_MAX_OPTIONS", "PermutationDebiaser", "random_orders"]

DEFAULT_MAX_OPTIONS = 32
"""Permuting debiasers skip Choice questions with more free options than this (each extra
presentation is a full scoring pass; banking77 and CLINC150 have 77 and 150 options)."""

PermutationKind = Literal["cyclic", "random"]


def random_orders(free: Sequence[bool], n: int, rng: random.Random) -> list[tuple[int, ...]]:
    """The identity, then up to ``n - 1`` distinct random shuffles of the free positions."""
    slots = [i for i, is_free in enumerate(free) if is_free]
    identity = tuple(range(len(free)))
    orders = [identity]
    seen = {identity}
    for _ in range(64 * n):  # the free positions may have fewer than n orders
        if len(orders) >= n:
            break
        shuffled = slots[:]
        rng.shuffle(shuffled)
        order = list(identity)
        for slot, source in zip(slots, shuffled, strict=True):
            order[slot] = source
        candidate = tuple(order)
        if candidate not in seen:
            seen.add(candidate)
            orders.append(candidate)
    return orders


class PermutationDebiaser:
    """Score a Choice question under several orders of its free options and average each
    option's probability over them (fixed options such as "I don't know" stay last).

    ``kind="cyclic"`` (default) uses the ``F`` cyclic shifts of the ``F`` free options, or
    ``n_permutations`` evenly spaced ones; ``kind="random"`` uses the identity plus
    ``n_permutations - 1`` random orders, seeded per question from ``seed`` and the question.
    Cost: ``n - 1`` extra scoring passes per Choice question. Score and Noul questions, and
    Choice questions with fewer than 2 or more than ``max_options`` free options, pass through.
    """

    name = "permutation"

    def __init__(
        self,
        *,
        kind: PermutationKind = "cyclic",
        n_permutations: int | None = None,
        seed: int = 0,
        max_options: int | None = DEFAULT_MAX_OPTIONS,
        fixed_texts: Sequence[str] = DEFAULT_FIXED_TEXTS,
    ) -> None:
        if kind not in ("cyclic", "random"):
            raise ValueError(f"kind must be 'cyclic' or 'random', got {kind!r}")
        if n_permutations is not None and n_permutations < 1:
            raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")
        if kind == "random" and n_permutations is None:
            raise ValueError("kind='random' needs n_permutations")
        if max_options is not None and max_options < 2:
            raise ValueError(f"max_options must be >= 2, got {max_options}")
        self.kind: PermutationKind = kind
        self.n_permutations = n_permutations
        self.seed = seed
        self.max_options = max_options
        self.fixed_texts = tuple(fixed_texts)

    @property
    def id(self) -> str:
        n = "all" if self.n_permutations is None else str(self.n_permutations)
        return (
            f"permutation(kind={self.kind},n={n},seed={self.seed},"
            f"max_options={self.max_options},fixed={list(self.fixed_texts)})"
        )

    def orders(self, question: ChoiceQuestion) -> list[tuple[int, ...]]:
        """The presentations scored for ``question``, the identity first."""
        free = free_positions(question, self.fixed_texts)
        if self.kind == "cyclic":
            return cyclic_orders(free, self.n_permutations)
        assert self.n_permutations is not None
        digest = hashlib.sha256(f"{self.seed}\x1f{question.model_dump_json()}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        return random_orders(free, self.n_permutations, rng)

    async def debias(
        self, state: JSONish, question: Question, base: ScoreResult, score: Scorer
    ) -> DebiasOutcome:
        if not isinstance(question, ChoiceQuestion):
            return passthrough(base)
        n_free = sum(free_positions(question, self.fixed_texts))
        if n_free < 2:
            return passthrough(base)
        if self.max_options is not None and n_free > self.max_options:
            return passthrough(
                base, f"permutation: {n_free} free options > max_options={self.max_options}"
            )
        presentations, calls, tokens = await score_orders(
            state, question, self.orders(question), base, score
        )
        return DebiasOutcome(
            logprobs=tuple(average_presentations(presentations).tolist()),
            extra_calls=calls,
            extra_prompt_tokens=tokens,
            presentations=tuple(presentations),
        )
