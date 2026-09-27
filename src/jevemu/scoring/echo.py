"""S4 echo sequence scoring: the logprob of each option's text written after the prefill.

Each label is scored as the continuation ``" " + label`` (the form the model writes after
``Answer:``): Choice options by key (the ``"keys"`` scheme, so any K up to 255 and semantic
option text), Score levels by digit (``" 7"`` = ``" "`` + ``"7"``, the model's own path) and
Noul by ``Yes``/``No``. One backend request per option; on vLLM 0.30.0 each is a full prefill,
because requests with prompt logprobs never read the prefix cache (``pmi`` doubles that).

Modes:

- ``sum`` (default): total logprob of the continuation.
- ``mean``: total divided by its token count (length-normalized).
- ``pmi``: total minus the total of the same continuation under a content-free state
  (``"N/A"`` by default), removing the options' prior ("surface form competition").

``raw_logprobs`` holds the per-option scores that are renormalized (mean or PMI scores in
those modes; PMI can be positive, and ``+inf`` when the content-free prompt gives an option
zero probability, in which case those options share all the mass). ``observed_mass`` sums the
continuations' probabilities (the chance the answer starts with one of the options).

Limitation: continuations carry no end marker, so under ``sum`` an option that is a prefix of
another (``"cat"``, ``"cat food"``) absorbs the longer option's probability. The trie strategy
scores the end of a label explicitly.

``prompt_tokens`` sums the tokens each echo request processed, as the backend reports them
(``SeqScore.prompt_tokens``; vLLM's ``usage.prompt_tokens``). A backend that does not report
them gets an estimate: per request, the tokenized prompt text (without chat-template tokens)
plus the continuation.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Literal, get_args

from jevemu.backends.base import Backend, Capabilities, RenderedPrompt, SeqScore
from jevemu.render.labels import sequence_scheme_for
from jevemu.render.renderer import PromptRenderer
from jevemu.scoring.aggregate import (
    low_mass_warning,
    observed_mass,
    renormalize,
    sequence_logprob,
)
from jevemu.scoring.base import ScoreResult
from jevemu.scoring.first_token import GAP
from jevemu.types import JSONish, Question

__all__ = ["DEFAULT_NULL_STATE", "EchoMode", "EchoStrategy"]

EchoMode = Literal["sum", "mean", "pmi"]
DEFAULT_NULL_STATE = "N/A"
"""Content-free state for ``pmi``."""


class EchoStrategy:
    """S4: score every option's text by echo (prompt logprobs); modes ``sum``/``mean``/``pmi``."""

    def __init__(self, mode: EchoMode = "sum", *, null_state: JSONish = DEFAULT_NULL_STATE) -> None:
        if mode not in get_args(EchoMode):
            raise ValueError(f"unknown echo mode {mode!r}; expected one of {get_args(EchoMode)}")
        self.mode: EchoMode = mode
        self.null_state = null_state
        self.name = f"echo_{mode}"

    def supports(self, capabilities: Capabilities, question: Question) -> bool:
        return capabilities.assistant_prefill and capabilities.echo_prompt_logprobs

    async def score(
        self, backend: Backend, renderer: PromptRenderer, state: JSONish, question: Question
    ) -> ScoreResult:
        scheme = sequence_scheme_for(question)
        continuations = [GAP + label for label in scheme.labels]
        prompts = [renderer.render(state, question, scheme)]
        if self.mode == "pmi":
            prompts.append(renderer.render(self.null_state, question, scheme))
        scored = await asyncio.gather(
            *(backend.sequence_logprobs(p, continuations) for p in prompts)
        )
        totals = _totals(scored[0], continuations)
        warnings: list[str] = []
        if self.mode == "sum":
            scores = totals
        elif self.mode == "mean":
            scores = [t / len(s.tokens) for t, s in zip(totals, scored[0], strict=True)]
        else:
            null = _totals(scored[1], continuations)
            scores = [_pmi(c, n) for c, n in zip(totals, null, strict=True)]
            unbounded = [k for k, s in zip(scheme.keys, scores, strict=True) if s == math.inf]
            if unbounded:
                warnings.append(
                    f"options {unbounded} have zero probability under the content-free state; "
                    "their PMI is unbounded and they share all the mass"
                )
        mass = observed_mass(totals)
        low = low_mass_warning(mass, backend.capabilities.logprobs_mode)
        if low is not None:
            warnings.append(low)
        prompt_tokens = await _prompt_tokens(backend, prompts, scored)
        return ScoreResult(
            keys=scheme.keys,
            logprobs=renormalize(scores),
            raw_logprobs=tuple(scores),
            observed_mass=mass,
            missing=(),
            truncated=False,
            strategy=self.name,
            label_scheme=scheme.name,
            n_backend_calls=len(continuations) * len(prompts),
            prompt_tokens=prompt_tokens,
            cached_tokens=None,
            warnings=tuple(warnings),
        )


def _prompt_text(prompt: RenderedPrompt) -> str:
    return "".join(message.content for message in prompt.messages)


def _totals(scores: Sequence[SeqScore], continuations: Sequence[str]) -> list[float]:
    got = [s.continuation for s in scores]
    if got != list(continuations):
        raise ValueError(f"echo returned continuations {got}, expected {list(continuations)}")
    return [sequence_logprob(s) for s in scores]


def _pmi(conditional: float, null: float) -> float:
    if conditional == -math.inf:
        return -math.inf
    if null == -math.inf:
        return math.inf
    return conditional - null


async def _prompt_tokens(
    backend: Backend, prompts: Sequence[RenderedPrompt], scored: Sequence[Sequence[SeqScore]]
) -> int:
    """Reported prompt tokens of every echo request, else the estimate (module docstring)."""
    reported = [s.prompt_tokens for batch in scored for s in batch]
    if all(n is not None for n in reported):
        return sum(n for n in reported if n is not None)
    prefixes = await asyncio.gather(*(backend.tokenize(_prompt_text(p)) for p in prompts))
    return sum(
        len(prefix) * len(batch) + sum(len(s.tokens) for s in batch)
        for prefix, batch in zip(prefixes, scored, strict=True)
    )
