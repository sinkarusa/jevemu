"""``auto``: pick a scoring strategy from the backend's measured capabilities.

The design's selection table, applied to every question type (Noul gets ``Yes``/``No`` and
Score gets digits from the same rows, since both always fit a probed top-k):

| Condition | Strategy |
| --- | --- |
| K x variants <= ``top_logprobs_max`` | S2 constrained if structured choice works, else S1 |
| K x variants > cap, explicit tokens available | S5 (not implemented yet: skipped) |
| echo available (K <= 255 always holds) | S4 echo |
| otherwise | S3 trie |

``variants`` is the number of surfaces a label competes with in the top-k: ``"A"`` and
``" A"`` after the ``Answer:`` prefill, one without a prefill
(:func:`~jevemu.scoring.first_token.surface_variants`). Choice above 52 options has no letter
labels and goes down the table like an overflow.

A backend without ``assistant_prefill`` (a hosted chat API) only has the first row: S2 as a
structured read of a new assistant turn (S1 on its first token without structured choice).
Questions that do not fit its top-k raise ``CapabilityError``, since echo and trie reads continue
the prefill.
Every fallback is recorded in the result's ``warnings``; the result's ``strategy`` names the
strategy that scored it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from jevemu.backends.base import Backend, Capabilities, CapabilityError
from jevemu.render.labels import LabelCapacityError, scheme_for
from jevemu.render.renderer import PromptRenderer
from jevemu.scoring.base import ScoreResult, ScoringStrategy
from jevemu.scoring.constrained import ConstrainedStrategy
from jevemu.scoring.echo import EchoMode, EchoStrategy
from jevemu.scoring.first_token import FirstTokenStrategy, surface_variants
from jevemu.scoring.trie import DEFAULT_TAU, TrieStrategy
from jevemu.types import JSONish, Question

__all__ = ["AutoStrategy", "Selection", "select", "select_strategy"]


@dataclass(frozen=True)
class Selection:
    strategy: ScoringStrategy
    fallbacks: tuple[str, ...]
    """Why the preferred strategy was not used, in table order; empty on the default path."""


def select(
    capabilities: Capabilities,
    question: Question,
    *,
    echo_mode: EchoMode = "sum",
    tau: float = DEFAULT_TAU,
) -> Selection:
    """The strategy the design's table picks for ``question`` on a backend with these caps.

    Raises ``CapabilityError`` when the backend cannot continue an assistant prefill and the
    question's labels do not fit one first-token top-k.
    """
    variants = surface_variants(capabilities)
    cap = capabilities.top_logprobs_max
    try:
        n_labels = len(scheme_for(question))
    except LabelCapacityError as exc:
        overflow = f"{exc.needed} options exceed the {exc.capacity} letter labels"
    else:
        needed = n_labels * variants
        if needed <= cap:
            if capabilities.structured_choice:
                return Selection(ConstrainedStrategy(), ())
            if not capabilities.assistant_prefill:
                return Selection(
                    FirstTokenStrategy(),
                    ("no assistant prefill: S1 first-token on a new assistant turn",),
                )
            return Selection(
                FirstTokenStrategy(),
                ("structured choice unavailable: S1 first-token without a constraint",),
            )
        overflow = f"{n_labels} labels x {variants} variants = {needed} > top_logprobs_max {cap}"
    if not capabilities.assistant_prefill:
        raise CapabilityError(
            f"{overflow}, and without assistant_prefill no other strategy applies "
            "(echo and trie reads continue the 'Answer:' prefill)"
        )
    fallbacks = [
        f"{overflow}: first-token scoring cannot see every label",
        "explicit-token scoring (S5) is not implemented yet: skipped"
        if capabilities.explicit_token_logprobs
        else "explicit-token logprobs unavailable: S5 skipped",
    ]
    if capabilities.echo_prompt_logprobs:
        fallbacks.append("using S4 echo")
        return Selection(EchoStrategy(echo_mode), tuple(fallbacks))
    fallbacks.append("echo unavailable: using S3 trie")
    return Selection(TrieStrategy(tau=tau), tuple(fallbacks))


def select_strategy(capabilities: Capabilities, question: Question) -> ScoringStrategy:
    """The strategy ``auto`` uses for ``question`` on a backend with these capabilities."""
    return select(capabilities, question).strategy


class AutoStrategy:
    """Scores each question with the strategy :func:`select` picks for the backend."""

    name = "auto"

    def __init__(self, *, echo_mode: EchoMode = "sum", tau: float = DEFAULT_TAU) -> None:
        self.echo_mode: EchoMode = echo_mode
        self.tau = tau

    def supports(self, capabilities: Capabilities, question: Question) -> bool:
        try:
            select(capabilities, question)
        except CapabilityError:
            return False
        return True

    async def score(
        self, backend: Backend, renderer: PromptRenderer, state: JSONish, question: Question
    ) -> ScoreResult:
        selection = select(backend.capabilities, question, echo_mode=self.echo_mode, tau=self.tau)
        result = await selection.strategy.score(backend, renderer, state, question)
        notes = tuple(f"auto: {note}" for note in selection.fallbacks)
        return dataclasses.replace(result, warnings=notes + result.warnings)
