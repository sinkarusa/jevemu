"""``auto`` with echo (S4) switched off: the selection runs' scoring policy.

On vLLM 0.30.0 each echo request is a full prefill that never reads the prefix cache, one per
option, so ``auto`` on a question with more than 32 options (banking77, CLINC150) costs tens of
seconds per item (docs/research/selection.md, "Stage 1 screen"). :class:`NoEchoAutoStrategy` runs
``auto`` on a view of the backend without echo prompt logprobs:

- questions whose labels fit the top-k get what ``auto`` gives them (S2 constrained on vLLM);
- the ones ``auto`` would echo get the S3 trie over their option keys, pruned at ``tau``;
- a label missing from a top-k gets the upper bound instead of an echo fill.

    Emulator(backend, strategy=NoEchoAutoStrategy())
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Sequence

from jevemu.backends.base import (
    Backend,
    BackendInfo,
    Capabilities,
    CapabilityError,
    NextTokenDist,
    RenderedPrompt,
    SeqScore,
)
from jevemu.render.renderer import PromptRenderer
from jevemu.scoring.auto import AutoStrategy
from jevemu.scoring.base import ScoreResult
from jevemu.types import JSONish, Question

__all__ = ["NO_ECHO_TAU", "NoEchoAutoStrategy", "echo_off", "without_echo"]

NO_ECHO_TAU = 1e-3
"""Trie pruning threshold of the selection runs (docs/research/selection.md, "Stage 1 screen")."""


class _WithoutEcho:
    """``inner`` with echo prompt logprobs switched off; every other call passes through."""

    def __init__(self, inner: Backend) -> None:
        self._inner = inner

    @property
    def capabilities(self) -> Capabilities:
        return dataclasses.replace(self._inner.capabilities, echo_prompt_logprobs=False)

    @capabilities.setter
    def capabilities(self, value: Capabilities) -> None:
        self._inner.capabilities = value

    async def next_token_logprobs(
        self, prompt: RenderedPrompt, *, allowed: Sequence[str] | None, top_k: int
    ) -> NextTokenDist:
        return await self._inner.next_token_logprobs(prompt, allowed=allowed, top_k=top_k)

    async def sequence_logprobs(
        self, prefix: RenderedPrompt, continuations: Sequence[str]
    ) -> list[SeqScore]:
        raise CapabilityError("echo is switched off by the auto_noecho policy")

    async def tokenize(self, text: str) -> list[int]:
        return await self._inner.tokenize(text)

    async def detokenize(self, ids: Sequence[int]) -> str:
        return await self._inner.detokenize(ids)

    async def health(self) -> BackendInfo:
        return await self._inner.health()


_VIEWS: weakref.WeakKeyDictionary[Backend, _WithoutEcho] = weakref.WeakKeyDictionary()


def without_echo(backend: Backend) -> Backend:
    """``backend`` with echo prompt logprobs switched off.

    One view per backend, so the tokenizer caches keyed by backend object stay warm.
    """
    view = _VIEWS.get(backend)
    if view is None:
        view = _VIEWS[backend] = _WithoutEcho(backend)
    return view


def echo_off(result: ScoreResult) -> ScoreResult:
    """``result`` with warnings saying ``echo off`` where ``auto`` says ``echo unavailable``."""
    warnings = tuple(w.replace("echo unavailable", "echo off") for w in result.warnings)
    return dataclasses.replace(result, warnings=warnings)


class NoEchoAutoStrategy:
    """``auto`` on the backend with echo (S4) switched off, for cost.

    ``name`` is ``auto_noecho_tau<tau>`` (``auto_noecho_tau0.001`` by default), the strategy
    identity run manifests record. Warnings say ``echo off`` where ``auto`` would say ``echo
    unavailable``.
    """

    def __init__(self, *, tau: float = NO_ECHO_TAU) -> None:
        self.name = f"auto_noecho_tau{tau:g}"
        self._auto = AutoStrategy(tau=tau)

    def supports(self, capabilities: Capabilities, question: Question) -> bool:
        return capabilities.assistant_prefill

    async def score(
        self, backend: Backend, renderer: PromptRenderer, state: JSONish, question: Question
    ) -> ScoreResult:
        return echo_off(await self._auto.score(without_echo(backend), renderer, state, question))
