"""Scripted, deterministic backend for tests: no network, no model.

``FakeBackend`` returns logprobs from tables you supply, and simulates the capability-dependent
behavior of vLLM (``--max-logprobs`` truncation, whether a structured-choice mask is reflected in
the reported logprobs, missing echo support) so strategies can be tested against each variant.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

from jevemu.backends.base import (
    BackendInfo,
    Capabilities,
    CapabilityError,
    NextTokenDist,
    RenderedPrompt,
    SeqScore,
    TokenLogprob,
)

__all__ = [
    "DEFAULT_CAPABILITIES",
    "FakeBackend",
    "NextTokenScript",
    "SequenceScript",
]

DEFAULT_CAPABILITIES = Capabilities(
    top_logprobs_max=64,
    structured_choice=True,
    logprobs_mode="raw_logprobs",
    mask_reflected_in_logprobs=False,
    echo_prompt_logprobs=True,
    explicit_token_logprobs=False,
    assistant_prefill=True,
    prefix_caching=True,
)

NextTokenScript: TypeAlias = (
    Mapping[str, float] | Callable[[RenderedPrompt, Sequence[str] | None], Mapping[str, float]]
)
"""Full next-token table (token text -> logprob), or a function of (prompt, allowed) giving one."""

SequenceScript: TypeAlias = (
    Mapping[str, Sequence[tuple[str, float]]]
    | Callable[[RenderedPrompt, str], Sequence[tuple[str, float]]]
)
"""Continuation -> [(token, logprob)], or a function of (prefix, continuation) giving that list."""


def _logsumexp(values: Sequence[float]) -> float:
    peak = max(values)
    if peak == -math.inf:
        return -math.inf
    return peak + math.log(math.fsum(math.exp(v - peak) for v in values))


class FakeBackend:
    """Backend driven by scripted logprob tables. Pure and deterministic; logs every call.

    ``prompt_tokens`` is the character count of all prompt messages (one token per character);
    ``cached_tokens`` is always ``None`` (not reported).
    """

    def __init__(
        self,
        *,
        capabilities: Capabilities = DEFAULT_CAPABILITIES,
        next_token: NextTokenScript | None = None,
        sequences: SequenceScript | None = None,
        vocab: Sequence[str] | None = None,
        model: str = "fake-model",
    ) -> None:
        self.capabilities = capabilities
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._next_token = next_token
        self._sequences = sequences
        self._model = model
        self._vocab: tuple[str, ...] | None = None
        self._token_ids: dict[str, int] = {}
        self._max_token_len = 0
        if vocab is not None:
            self._vocab = tuple(vocab)
            if any(not token for token in self._vocab):
                raise ValueError("FakeBackend vocab entries must be non-empty strings")
            for index, token in enumerate(self._vocab):
                self._token_ids.setdefault(token, index)
            self._max_token_len = max((len(token) for token in self._vocab), default=0)

    async def next_token_logprobs(
        self, prompt: RenderedPrompt, *, allowed: Sequence[str] | None, top_k: int
    ) -> NextTokenDist:
        allowed_tuple = None if allowed is None else tuple(allowed)
        self.calls.append(
            ("next_token_logprobs", {"prompt": prompt, "allowed": allowed_tuple, "top_k": top_k})
        )
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if allowed_tuple is not None and not self.capabilities.structured_choice:
            raise CapabilityError(
                "FakeBackend: `allowed` given but capabilities.structured_choice is False"
            )
        if self._next_token is None:
            raise ValueError("FakeBackend: no `next_token` table scripted")
        table = (
            self._next_token(prompt, allowed_tuple)
            if callable(self._next_token)
            else self._next_token
        )
        if not table:
            raise ValueError("FakeBackend: scripted `next_token` table is empty")

        entries = list(table.items())
        sampled_pool = entries
        if allowed_tuple is not None:
            allowed_set = set(allowed_tuple)
            sampled_pool = [(token, lp) for token, lp in entries if token in allowed_set]
            if not sampled_pool or max(lp for _, lp in sampled_pool) == -math.inf:
                raise ValueError(
                    f"FakeBackend: no allowed token {sorted(allowed_set)!r} has finite "
                    "probability in the scripted table"
                )
            if self.capabilities.mask_reflected_in_logprobs:
                # As vLLM 0.30.0: allowed tokens renormalized over the allowed set; masked
                # tokens still fill the top-k at the floor, which the HTTP adapter maps to -inf.
                norm = _logsumexp([lp for _, lp in sampled_pool])
                sampled_pool = [(token, lp - norm) for token, lp in sampled_pool]
                entries = [
                    (token, lp - norm if token in allowed_set else -math.inf)
                    for token, lp in entries
                ]

        # Stable sort: ties (e.g. floored -inf entries) keep table order, so results are
        # deterministic.
        ranked = sorted(entries, key=lambda item: -item[1])
        k = min(top_k, self.capabilities.top_logprobs_max)
        best_token, best_lp = max(sampled_pool, key=lambda item: item[1])
        return NextTokenDist(
            top=tuple(self._token_logprob(token, lp) for token, lp in ranked[:k]),
            sampled=self._token_logprob(best_token, best_lp),
            constrained=allowed_tuple is not None,
            prompt_tokens=sum(len(message.content) for message in prompt.messages),
        )

    async def sequence_logprobs(
        self, prefix: RenderedPrompt, continuations: Sequence[str]
    ) -> list[SeqScore]:
        continuations_tuple = tuple(continuations)
        self.calls.append(
            ("sequence_logprobs", {"prefix": prefix, "continuations": continuations_tuple})
        )
        if not self.capabilities.echo_prompt_logprobs:
            raise CapabilityError(
                "FakeBackend: sequence_logprobs needs capabilities.echo_prompt_logprobs"
            )
        if self._sequences is None:
            raise ValueError("FakeBackend: no `sequences` table scripted")
        scores: list[SeqScore] = []
        for continuation in continuations_tuple:
            if callable(self._sequences):
                pairs = self._sequences(prefix, continuation)
            else:
                try:
                    pairs = self._sequences[continuation]
                except KeyError:
                    raise ValueError(
                        f"FakeBackend: continuation {continuation!r} not in `sequences` table"
                    ) from None
            scores.append(
                SeqScore(
                    continuation=continuation,
                    tokens=tuple(token for token, _ in pairs),
                    token_logprobs=tuple(lp for _, lp in pairs),
                )
            )
        return scores

    async def tokenize(self, text: str) -> list[int]:
        self.calls.append(("tokenize", {"text": text}))
        vocab = self._require_vocab()
        ids: list[int] = []
        pos = 0
        while pos < len(text):
            for length in range(min(self._max_token_len, len(text) - pos), 0, -1):
                token_id = self._token_ids.get(text[pos : pos + length])
                if token_id is not None:
                    ids.append(token_id)
                    pos += length
                    break
            else:
                raise ValueError(
                    f"FakeBackend: cannot tokenize {text[pos]!r} at offset {pos}: "
                    f"not covered by vocab of {len(vocab)} tokens"
                )
        return ids

    async def detokenize(self, ids: Sequence[int]) -> str:
        ids_tuple = tuple(ids)
        self.calls.append(("detokenize", {"ids": ids_tuple}))
        vocab = self._require_vocab()
        for token_id in ids_tuple:
            if not 0 <= token_id < len(vocab):
                raise ValueError(f"FakeBackend: token id {token_id} outside vocab")
        return "".join(vocab[token_id] for token_id in ids_tuple)

    async def health(self) -> BackendInfo:
        self.calls.append(("health", {}))
        return BackendInfo(
            backend="fake", model=self._model, model_revision=None, engine_version="fake"
        )

    def _require_vocab(self) -> tuple[str, ...]:
        if self._vocab is None:
            raise ValueError("FakeBackend: no `vocab` given; tokenize/detokenize unavailable")
        return self._vocab

    def _token_logprob(self, token: str, logprob: float) -> TokenLogprob:
        return TokenLogprob(token=token, token_id=self._token_ids.get(token), logprob=logprob)
