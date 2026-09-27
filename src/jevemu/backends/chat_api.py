"""Shared transport of the Chat Completions backends: answer logprobs from a hosted model.

:class:`ChatCompletionsBackend` is subclassed per API (:mod:`jevemu.backends.openai_chat`,
:mod:`jevemu.backends.openrouter_chat`); a subclass builds the request body, prices a response,
checks a response's identity and classifies out-of-credit errors. The rest is shared:

- **Scoring surface.** Prompts end in the user turn (``PromptRenderer(prefill=False)``); only
  :meth:`~ChatCompletionsBackend.next_token_logprobs` works, every other scoring call raises
  ``CapabilityError``. Without ``allowed`` it reads the first token of a free reply. With
  ``allowed`` (the **structured read**) the reply is constrained by a Structured Outputs JSON
  schema whose only value is an ``enum`` over the allowed labels (:class:`AnswerFormat`), and
  the distribution is read where that value starts (:func:`parse_structured_response`).
- **Reasoning off.** A response that bills reasoning tokens is rejected
  (:class:`~jevemu.errors.ChatAPIResponseError`) and not cached.
- **Cache first.** :class:`~jevemu.cache.ResponseCache` is keyed by the model and the canonical
  request body, so reruns and resumes are free and need no API key.
- **Resilience.** 429 (except out-of-credit), 5xx and connection failures are retried with
  :class:`~jevemu.jev_client.RetryPolicy`, never sooner than ``retry-after-ms`` /
  ``retry-after``. A :class:`~jevemu.jev_client.TokenBucket` keeps request starts at or below
  ``max_rpm`` and at most ``max_concurrency`` requests are in flight.
- **Accounting.** Every call's spend (the subclass's :meth:`~ChatCompletionsBackend.call_cost`)
  is reported on ``NextTokenDist.api`` (:class:`~jevemu.backends.base.ApiCall`). ``max_usd``
  refuses a request before sending it if a pessimistic estimate at :attr:`price` could push
  spend past the budget (:class:`~jevemu.errors.ChatAPIBudgetExceeded`).
- **Empty replies.** A reply with no answer to read (no token logprobs, or a structured reply
  that never reaches its value) is not cached and is sent again, up to
  :data:`EMPTY_REPLY_ATTEMPTS` requests; after a structured reply stuck before its value the
  resends raise the output-token cap to :data:`STUCK_REPLY_MAX_TOKENS`.
- **Masked tokens.** Logprobs at or below :data:`MASKED_LOGPROB` (a Structured Outputs mask's
  floor) are read as ``-inf``.
- **Identity.** Every distinct :class:`ResponseIdentity` seen is counted in
  :attr:`~ChatCompletionsBackend.identities` and a new one is logged.

The API key comes from the argument or the subclass's environment variable and is never logged,
cached, recorded or included in exception messages.
"""

from __future__ import annotations

import asyncio
import bisect
import dataclasses
import itertools
import json
import logging
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, ClassVar, NamedTuple, TypeVar

import httpx

from jevemu.backends.base import (
    PRICE_ID_FLAG,
    ApiCall,
    BackendInfo,
    Capabilities,
    CapabilityError,
    NextTokenDist,
    RenderedPrompt,
    SeqScore,
    TokenLogprob,
)
from jevemu.cache import CacheEntry, ResponseCache, cache_key, canonical_json
from jevemu.errors import (
    ChatAPIAuthError,
    ChatAPIBudgetExceeded,
    ChatAPIConnectionError,
    ChatAPIEmptyReply,
    ChatAPIHTTPError,
    ChatAPIKeyMissing,
    ChatAPIQuotaExceeded,
    ChatAPIResponseError,
    ChatAPIStuckReply,
)
from jevemu.eval.costs import TokenPrice, TokenUsage
from jevemu.jev_client import Clock, RetryPolicy, Sleep, TokenBucket, parse_retry_after

__all__ = [
    "ANSWER_KEY",
    "CHAT_PATH",
    "EMPTY_REPLY_ATTEMPTS",
    "MASKED_LOGPROB",
    "STUCK_REPLY_MAX_TOKENS",
    "AnswerFormat",
    "ApiReply",
    "ChatCompletionsBackend",
    "ResponseIdentity",
    "estimate_usage",
    "parse_chat_response",
    "parse_structured_response",
    "reasoning_tokens",
    "response_usage",
]

logger = logging.getLogger(__name__)

CHAT_PATH = "/v1/chat/completions"

ESTIMATE_BYTES_PER_TOKEN = 2
ESTIMATE_OVERHEAD_TOKENS = 64
"""Budget-guard estimate: ``64 + ceil(bytes / 2)`` input tokens for the compact request JSON, all
billed at the cache-write rate (the highest input rate), plus the output-token cap. English text
runs about 4 bytes per token, so this is deliberately pessimistic."""

EMPTY_REPLY_ATTEMPTS = 3
"""Requests :meth:`ChatCompletionsBackend.complete` sends before giving up on empty replies
(:class:`~jevemu.errors.ChatAPIEmptyReply`)."""

STUCK_REPLY_MAX_TOKENS = 64
"""Output-token cap of the resends after a structured reply that never reached its value
(:class:`~jevemu.errors.ChatAPIStuckReply`). The cap does not condition the distribution read at
the value token; it only leaves room for a reply that runs past its leading whitespace
(measured on Makora, 2026-09-27: ``\\r\\n`` repeated after ``"answer": `` until the cap)."""

POSITIVE_LOGPROB_TOLERANCE = 1e-3
"""Measured 2026-09-25: gpt-6-luna reports a near-certain token at slightly positive logprobs
(``3.814697265625e-06`` = 2^-18, float32 rounding). Values up to this are clamped to 0."""

MASKED_LOGPROB = -9999.0
"""Logprobs at or below this read as ``-inf``: where Structured Outputs masks tokens, providers
still list them, at ``-9999`` (Makora via OpenRouter, 2026-09-27) or the float32 minimum
``-3.4028234663852886e+38`` (Wafer)."""

SAME_LOGPROB = 1e-3
"""How close a listed alternative's logprob must be to the generated token's own logprob for the
list to count as that position's (see :func:`parse_structured_response`)."""

ANSWER_KEY = "answer"
"""Property holding the answer in an :class:`AnswerFormat` reply, and the schema's name."""

_VALUE_OPENING = re.compile(r'\s*\{?\s*"?' + ANSWER_KEY + r'"?\s*:\s*"?')
"""Everything a structured reply's token stream holds before its value's first character. The
brace and quotes may be missing: measured on Makora (2026-09-27), tokens the grammar forces are
left out of the logprobs (``{\\n\\n`` then ``answer``, or ``\\t`` then ``4`` for ``\\t"4``)
although the message text has them; a value read after a forced quote is conditioned on it."""

_ERROR_TEXT_LIMIT = 2000
_REDACTED = "***"

_Backend = TypeVar("_Backend", bound="ChatCompletionsBackend")


# --- pure helpers --------------------------------------------------------------------------------


def _count(mapping: Mapping[str, Any] | None, name: str) -> int | None:
    value = (mapping or {}).get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ChatAPIResponseError(f"usage field {name!r} is not a token count", value)
    return value


def response_usage(payload: Mapping[str, Any]) -> TokenUsage:
    """Billed tokens of a chat-completions response. ``input_tokens`` includes the cached and
    cache-write tokens (``usage.prompt_tokens_details``); output includes reasoning tokens."""
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        raise ChatAPIResponseError("response has no usage", payload)
    details = usage.get("prompt_tokens_details")
    prompt_tokens = _count(usage, "prompt_tokens")
    completion_tokens = _count(usage, "completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        raise ChatAPIResponseError("usage lacks prompt_tokens/completion_tokens", usage)
    return TokenUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        cached_input_tokens=_count(details, "cached_tokens"),
        cache_write_tokens=_count(details, "cache_write_tokens") or 0,
    )


def estimate_usage(canonical_request: str, max_output_tokens: int) -> TokenUsage:
    """Pessimistic usage for the budget guard (see ``ESTIMATE_*``)."""
    tokens = ESTIMATE_OVERHEAD_TOKENS + math.ceil(
        len(canonical_request.encode()) / ESTIMATE_BYTES_PER_TOKEN
    )
    return TokenUsage(
        input_tokens=tokens, output_tokens=max_output_tokens, cache_write_tokens=tokens
    )


def reasoning_tokens(payload: Mapping[str, Any]) -> int | None:
    """``usage.completion_tokens_details.reasoning_tokens``, or None if not reported."""
    usage = payload.get("usage")
    details = usage.get("completion_tokens_details") if isinstance(usage, Mapping) else None
    return _count(details if isinstance(details, Mapping) else None, "reasoning_tokens")


def _logprob(entry: Mapping[str, Any]) -> TokenLogprob:
    value = float(entry["logprob"])
    if math.isnan(value) or value > POSITIVE_LOGPROB_TOLERANCE:
        raise ChatAPIResponseError("response holds a logprob that is not a log-probability", entry)
    logprob = -math.inf if value <= MASKED_LOGPROB else min(value, 0.0)
    return TokenLogprob(token=str(entry["token"]), token_id=None, logprob=logprob)


def _log_sum(logprobs: Sequence[float]) -> float:
    peak = max(logprobs)
    return peak + math.log(math.fsum(math.exp(v - peak) for v in logprobs))


def _token_logprobs(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """``choices[0].logprobs.content``: one entry per generated token (``ChatAPIEmptyReply`` if
    there is none)."""
    try:
        choice = payload["choices"][0]
    except (KeyError, IndexError, TypeError):
        raise ChatAPIResponseError("response has no choices", payload) from None
    content: list[Mapping[str, Any]] = (choice.get("logprobs") or {}).get("content") or []
    if not content:
        text = (choice.get("message") or {}).get("content")
        raise ChatAPIEmptyReply(
            "chat completion returned no token logprobs "
            f"(finish_reason={choice.get('finish_reason')!r}, text={text!r})",
            payload,
        )
    return content


def parse_chat_response(payload: Mapping[str, Any]) -> NextTokenDist:
    """The first generated token's top-k of a chat-completions response (``api`` unset; the
    backend adds its accounting)."""
    first = _token_logprobs(payload)[0]
    top = sorted(
        (_logprob(alt) for alt in first.get("top_logprobs") or []),
        key=lambda t: t.logprob,
        reverse=True,
    )
    usage = response_usage(payload)
    return NextTokenDist(
        top=tuple(top),
        sampled=_logprob(first),
        constrained=False,
        prompt_tokens=usage.input_tokens,
        cached_tokens=usage.cached_input_tokens,
    )


@dataclass(frozen=True)
class AnswerFormat:
    """A Structured Outputs ``response_format`` whose reply is ``{"answer": "<value>"}`` with
    ``<value>`` one of ``values`` (OpenAI accepts only object roots).

    Values are plain labels: non-empty, no quote, backslash or control character (so the reply
    text is the JSON text), and distinct first characters (so any piece of a value names one
    value).
    """

    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("an answer format needs at least one value")
        for value in self.values:
            if not value or not value.isprintable() or '"' in value or "\\" in value:
                raise ValueError(f"answer value {value!r} is not a plain label")
        if len({value[0] for value in self.values}) != len(self.values):
            raise ValueError(f"answer values {self.values} must start with distinct characters")

    @classmethod
    def of(cls, body: Mapping[str, Any]) -> AnswerFormat | None:
        """The format a request ``body`` asks for, or None for a free reply."""
        spec = body.get("response_format")
        if spec is None:
            return None
        try:
            return cls(tuple(spec["json_schema"]["schema"]["properties"][ANSWER_KEY]["enum"]))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"not an answer format: {spec!r}") from exc

    def response_format(self) -> dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {ANSWER_KEY: {"type": "string", "enum": list(self.values)}},
            "required": [ANSWER_KEY],
            "additionalProperties": False,
        }
        return {
            "type": "json_schema",
            "json_schema": {"name": ANSWER_KEY, "strict": True, "schema": schema},
        }

    @staticmethod
    def value_start(text: str) -> int | None:
        """Offset of the value's first character in ``text`` (a reply's token texts joined; see
        ``_VALUE_OPENING``), or None."""
        match = _VALUE_OPENING.match(text)
        return None if match is None else match.end()

    def value_of(self, piece: str) -> str | None:
        """The value a reply piece starting at the value's first character begins (``B``,
        ``B"`` and ``B"}`` -> ``B``; ``Ye`` -> ``Yes``), or None."""
        head = piece.partition('"')[0]
        if not head:
            return None
        found = [value for value in self.values if value.startswith(head)]
        return found[0] if found and (head == found[0] or '"' not in piece) else None


def _listed(entry: Mapping[str, Any]) -> list[TokenLogprob]:
    """``entry``'s alternatives, checked to be its own position's: the generated token must be
    among them at its own logprob. Wafer returns, for a token generated in the same decoding
    step as the one before it, a copy of an earlier position's list."""
    top = [_logprob(alt) for alt in entry.get("top_logprobs") or []]
    own = _logprob(entry)
    if not any(t.token == own.token and abs(t.logprob - own.logprob) <= SAME_LOGPROB for t in top):
        raise ChatAPIResponseError(
            f"top_logprobs at the answer value do not list the generated token {own.token!r}: "
            "a stale list from another position",
            [(t.token, t.logprob) for t in top],
        )
    return top


def parse_structured_response(payload: Mapping[str, Any], answer: AnswerFormat) -> NextTokenDist:
    """The distribution over ``answer.values`` of a structured reply, read where its value
    starts (``api`` unset; the backend adds its accounting).

    The value's first piece may share a token with the JSON before it (``":"B``, `` "B``). So
    the read starts at the token ``j`` that holds the value's opening quote, with ``prefix`` =
    that token's text up to the value. The alternatives at ``j`` that start the value right there
    are the ones the read uses:

    - ``prefix`` plus a piece of a value counts for that value (``":"A`` -> ``A``);
    - ``prefix`` alone: the value starts at token ``j + 1``. If the reply took it, every value
      listed there counts with that alternative's logprob added (``P(":"A) + P(":") *
      P(A | ":")``). If the reply took a token carrying a value piece, that mass stays
      unattributed: the labels it may hold are bounded by the leftover mass
      (:mod:`jevemu.scoring.missing_mass`).

    Everything is normalized over those alternatives: the read is the distribution over the
    values given that the value starts where the reply's does. Other alternatives at ``j`` (more
    whitespace, another JSON layout; measured on Makora, 2026-09-27: up to 0.6 of the mass on
    whitespace before SST-5 values) are other paths to the value and are left out. ``top`` holds
    one entry per value found, ``sampled`` the generated value with the logprob of its path.
    ``ChatAPIStuckReply`` when the reply never reaches its value (measured on Makora,
    2026-09-27: ``\\r\\n`` repeated until ``max_tokens``, on about 3% of SST-5 items; a new request
    usually answers). ``ChatAPIResponseError`` when the value is not an allowed one or a list it
    reads is not its position's (:func:`_listed`).
    """
    content = _token_logprobs(payload)
    tokens = [str(entry["token"]) for entry in content]
    start = answer.value_start("".join(tokens))
    if start is None:
        raise ChatAPIStuckReply("structured reply does not reach its answer value", tokens)
    ends = list(itertools.accumulate(len(token) for token in tokens))
    j = bisect.bisect_right(ends, start - 1)
    prefix = tokens[j][: start - (ends[j] - len(tokens[j]))]
    found: defaultdict[str, list[float]] = defaultdict(list)
    bare: list[float] = []
    for alt in _listed(content[j]):
        if alt.logprob == -math.inf or not alt.token.startswith(prefix):
            continue
        piece = alt.token[len(prefix) :]
        value = answer.value_of(piece)
        if value is not None:
            found[value].append(alt.logprob)
        elif not piece:
            bare.append(alt.logprob)
    here = [*bare, *itertools.chain.from_iterable(found.values())]
    if not here:
        raise ChatAPIResponseError("structured reply's value is not an allowed one", tokens)
    starts_here = _log_sum(here)
    chosen = _logprob(content[j])
    sampled_value = answer.value_of(chosen.token[len(prefix) :])
    sampled_logprob = chosen.logprob
    if chosen.token == prefix:
        if j + 1 == len(content):
            raise ChatAPIStuckReply("structured reply ends before its answer value", tokens)
        through = _log_sum(bare)
        for alt in _listed(content[j + 1]):
            value = answer.value_of(alt.token)
            if value is not None and alt.logprob > -math.inf:
                found[value].append(through + alt.logprob)
        after = _logprob(content[j + 1])
        sampled_value = answer.value_of(after.token)
        sampled_logprob = chosen.logprob + after.logprob
    if sampled_value is None:
        raise ChatAPIResponseError("structured reply's value is not an allowed one", tokens)
    top = sorted(
        (
            TokenLogprob(value, None, min(_log_sum(lps) - starts_here, 0.0))
            for value, lps in found.items()
        ),
        key=lambda t: t.logprob,
        reverse=True,
    )
    usage = response_usage(payload)
    return NextTokenDist(
        top=tuple(top),
        sampled=TokenLogprob(sampled_value, None, min(sampled_logprob - starts_here, 0.0)),
        constrained=True,
        prompt_tokens=usage.input_tokens,
        cached_tokens=usage.cached_input_tokens,
    )


@dataclass(frozen=True)
class ApiReply:
    """One chat-completions response and what it cost."""

    json: Any
    latency_ms: float
    """HTTP round trip (the recorded one for a response-cache hit)."""
    cost_usd: float
    """Spend incurred: 0 for a response-cache hit."""
    cached: bool
    """Served from the response cache."""


class ResponseIdentity(NamedTuple):
    """Who answered: the reported model, ``system_fingerprint`` and serving provider."""

    model: str
    system_fingerprint: str | None
    provider: str | None


# --- backend -------------------------------------------------------------------------------------


class ChatCompletionsBackend:
    """``Backend`` over a Chat Completions API (see the module docstring); subclass per API.

    ``cache`` is owned (closed on :meth:`aclose`) when ``owns_cache``. One ``httpx.AsyncClient``
    and one semaphore exist per event loop. Use ``async with`` or :meth:`aclose`.
    """

    service: ClassVar[str]
    """API name in log lines and error messages."""
    backend_name: ClassVar[str]
    """``BackendInfo.backend``."""
    engine_version: ClassVar[str]
    """``BackendInfo.engine_version``."""
    api_key_env: ClassVar[str]
    request_id_header: ClassVar[str] = "x-request-id"
    retryable_status: ClassVar[frozenset[int]] = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        model: str,
        *,
        price: TokenPrice,
        capabilities: Capabilities,
        base_url: str,
        cache: ResponseCache,
        owns_cache: bool,
        api_key: str | None = None,
        max_usd: float | None = None,
        max_rpm: int = 450,
        max_concurrency: int = 32,
        retry: RetryPolicy | None = None,
        timeout: float | httpx.Timeout = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if max_rpm < 1:
            raise ValueError(f"max_rpm must be >= 1, got {max_rpm}")
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        if max_usd is not None and max_usd < 0:
            raise ValueError(f"max_usd must be >= 0, got {max_usd}")
        self.model = model
        self.price = price
        self.price_id = price.price_id
        self.capabilities = capabilities
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.cache = cache
        self.max_usd = max_usd
        self.retry = retry or RetryPolicy()
        self.identities: Counter[ResponseIdentity] = Counter()
        """Responses per :class:`ResponseIdentity` (network and cache)."""
        self._owns_cache = owns_cache
        self._api_key = api_key or os.environ.get(self.api_key_env) or None
        self._max_concurrency = max_concurrency
        self._timeout = timeout
        self._transport = transport
        self._clock = clock
        self._sleep = sleep
        self._bucket = TokenBucket(max_rpm, clock=clock, sleep=sleep)
        self._rng = random.Random()
        self._spent_usd = 0.0
        self._reserved_usd = 0.0
        self._network_calls = 0
        self._cache_hits = 0
        self._empty_replies = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: httpx.AsyncClient | None = None
        self._semaphore: asyncio.Semaphore | None = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.model!r}, price_id={self.price_id!r}, "
            f"cache={self.cache!r}, max_usd={self.max_usd!r}, spent_usd={self._spent_usd:.6f})"
        )

    # -- per-API hooks --

    def build_body(self, prompt: RenderedPrompt, top_logprobs: int) -> dict[str, Any]:
        """Request body for a new assistant turn with ``top_logprobs`` alternatives per token
        (no ``response_format``; :meth:`request_body` adds it)."""
        raise NotImplementedError

    def max_output_tokens(self, body: Mapping[str, Any]) -> int:
        """The output-token cap ``body`` sets (for the budget estimate)."""
        raise NotImplementedError

    def with_max_output_tokens(self, body: Mapping[str, Any], cap: int) -> dict[str, Any]:
        """``body`` with its output-token cap set to ``cap``."""
        raise NotImplementedError

    def health_flags(self) -> dict[str, str]:
        """``BackendInfo.flags`` besides the price id: settings that change the answers."""
        raise NotImplementedError

    def identify(self, payload: Mapping[str, Any]) -> ResponseIdentity:
        """Who answered ``payload``; raises ``ChatAPIResponseError`` for a response that must
        not be used (another model, an unpinned provider)."""
        raise NotImplementedError

    def is_out_of_credit(self, status: int, body: Any) -> bool:
        """Whether an error response says the account has no credit left."""
        raise NotImplementedError

    def call_cost(self, payload: Mapping[str, Any]) -> float:
        """$ billed for one network response: the price table applied to its usage."""
        return self.price.cost(response_usage(payload))

    def model_revision(self) -> str | None:
        """``BackendInfo.model_revision``."""
        return None

    def price_snapshot(self) -> Mapping[str, Any] | None:
        """``BackendInfo.price`` (a price not in ``TOKEN_PRICES``)."""
        return None

    # -- accounting --

    @property
    def spent_usd(self) -> float:
        """Spend so far: every response received over the network."""
        return self._spent_usd

    @property
    def network_calls(self) -> int:
        return self._network_calls

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def empty_replies(self) -> int:
        """Network replies with no answer to read (discarded and sent again)."""
        return self._empty_replies

    async def __aenter__(self: _Backend) -> _Backend:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        client, self._client, self._loop, self._semaphore = self._client, None, None, None
        if client is not None:
            await client.aclose()
        if self._owns_cache:
            self.cache.close()

    # -- Backend protocol --

    async def next_token_logprobs(
        self, prompt: RenderedPrompt, *, allowed: Sequence[str] | None, top_k: int
    ) -> NextTokenDist:
        """The first token of a free reply, or with ``allowed`` the structured read: the
        distribution over ``allowed`` where the value of a reply constrained to them starts."""
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if prompt.has_prefill:
            raise ValueError(
                f"the {self.service} chat backend does not continue an assistant prefill; "
                "render with PromptRenderer(prefill=False)"
            )
        answer = None if allowed is None else AnswerFormat(tuple(allowed))
        body = self.request_body(prompt, min(top_k, self.capabilities.top_logprobs_max), answer)
        return await self.complete(body)

    async def sequence_logprobs(
        self, prefix: RenderedPrompt, continuations: Sequence[str]
    ) -> list[SeqScore]:
        raise CapabilityError(f"the {self.service} chat API returns no prompt (echo) logprobs")

    async def tokenize(self, text: str) -> list[int]:
        raise CapabilityError(f"{self.model}'s tokenizer is not available locally")

    async def detokenize(self, ids: Sequence[int]) -> str:
        raise CapabilityError(f"{self.model}'s tokenizer is not available locally")

    async def health(self) -> BackendInfo:
        """Static identity (no network call): the pinned model and the request settings."""
        return BackendInfo(
            backend=self.backend_name,
            model=self.model,
            model_revision=self.model_revision(),
            engine_version=self.engine_version,
            flags={**self.health_flags(), PRICE_ID_FLAG: self.price_id},
            price=self.price_snapshot(),
        )

    # -- requests --

    def request_body(
        self, prompt: RenderedPrompt, top_logprobs: int, answer: AnswerFormat | None = None
    ) -> dict[str, Any]:
        """The body :meth:`next_token_logprobs` sends: :meth:`build_body`, plus ``answer``'s
        ``response_format`` for a structured read."""
        body = self.build_body(prompt, top_logprobs)
        if answer is not None:
            body["response_format"] = answer.response_format()
        return body

    async def complete(
        self, body: Mapping[str, Any], *, bypass_cache: bool = False
    ) -> NextTokenDist:
        """:meth:`request` ``body`` and parse it: its first generated token, or for a body with
        an :class:`AnswerFormat` its answer value (:func:`parse_structured_response`).

        A reply with no answer to read (:class:`~jevemu.errors.ChatAPIEmptyReply`: the model
        ended its turn at once, as gpt-6-luna sometimes does, or a structured reply never reached
        its value) is not cached and is sent again, up to :data:`EMPTY_REPLY_ATTEMPTS` requests
        in all; after a stuck structured reply (:class:`~jevemu.errors.ChatAPIStuckReply`) the
        resends carry :data:`STUCK_REPLY_MAX_TOKENS`. The returned ``api`` accounting includes
        the cost and latency of the discarded attempts; so does a ``ChatAPIResponseError`` raised
        after them (``cost_usd``, ``latency_ms``), so a failed item still records what it cost.
        """
        answer = AnswerFormat.of(body)

        def parse(payload: Any) -> NextTokenDist:
            if answer is None:
                return parse_chat_response(payload)
            return parse_structured_response(payload, answer)

        def check(payload: Any) -> None:
            parse(payload)
            self._verified(payload)

        wasted_usd = wasted_ms = 0.0
        send = body
        for attempt in range(1, EMPTY_REPLY_ATTEMPTS + 1):
            try:
                reply = await self.request(send, bypass_cache=bypass_cache, check=check)
            except ChatAPIEmptyReply as exc:
                self._empty_replies += 1
                wasted_usd += exc.cost_usd
                wasted_ms += exc.latency_ms
                if attempt == EMPTY_REPLY_ATTEMPTS:
                    exc.cost_usd, exc.latency_ms = wasted_usd, wasted_ms
                    raise
                if isinstance(exc, ChatAPIStuckReply):
                    send = self.with_max_output_tokens(body, STUCK_REPLY_MAX_TOKENS)
                logger.warning(
                    "%s %s: %s (attempt %d/%d); sending again",
                    self.service,
                    self.model,
                    "reply stuck before its value"
                    if isinstance(exc, ChatAPIStuckReply)
                    else "empty reply",
                    attempt,
                    EMPTY_REPLY_ATTEMPTS,
                )
                continue
            except ChatAPIResponseError as exc:
                exc.cost_usd += wasted_usd
                exc.latency_ms += wasted_ms
                raise
            return self._accounted(
                dataclasses.replace(
                    reply,
                    cost_usd=reply.cost_usd + wasted_usd,
                    latency_ms=reply.latency_ms + wasted_ms,
                ),
                parse(reply.json),
            )
        raise AssertionError("unreachable")  # pragma: no cover

    def _verified(self, payload: Mapping[str, Any]) -> ResponseIdentity:
        """:meth:`identify` ``payload``, which must bill no reasoning tokens."""
        reasoning = reasoning_tokens(payload)
        if reasoning:
            raise ChatAPIResponseError(
                f"response billed {reasoning} reasoning tokens although reasoning is off", payload
            )
        return self.identify(payload)

    async def request(
        self,
        body: Mapping[str, Any],
        *,
        bypass_cache: bool = False,
        check: Callable[[Any], object] | None = None,
    ) -> ApiReply:
        """Send (or replay from the response cache) one chat-completions ``body``.

        The budget guard, retries, rate limit and spend accounting apply to any body.
        ``bypass_cache`` neither reads nor writes the response cache (nondeterminism probes).
        ``check`` validates a network response before it is cached; its
        :class:`~jevemu.errors.ChatAPIResponseError` propagates with the call's ``cost_usd`` and
        ``latency_ms`` set, and the response is not cached.
        """
        payload = canonical_json(body)
        key = cache_key(payload, self.model)
        if not bypass_cache:
            entry = self.cache.get(key)
            if entry is not None:
                self._cache_hits += 1
                return ApiReply(entry.response, entry.latency_ms, 0.0, cached=True)

        estimate = self.price.cost(estimate_usage(payload, self.max_output_tokens(body)))
        self._reserve(estimate)
        try:
            json_body, latency_ms = await self._send(payload)
        finally:
            self._reserved_usd -= estimate
        self._network_calls += 1
        try:
            cost = self.call_cost(json_body)
        except ChatAPIResponseError:
            # Charged but unreadable: count the estimate so the budget stays conservative.
            self._spent_usd += estimate
            raise
        self._spent_usd += cost
        if check is not None:
            try:
                check(json_body)
            except ChatAPIResponseError as exc:
                exc.cost_usd, exc.latency_ms = cost, latency_ms
                raise
        if not bypass_cache:
            self.cache.put(
                CacheEntry(
                    key=key,
                    model=self.model,
                    request=dict(body),
                    response=json_body,
                    latency_ms=latency_ms,
                    created_at=datetime.now(timezone.utc),
                )
            )
        return ApiReply(json_body, latency_ms, cost, cached=False)

    def _accounted(self, reply: ApiReply, dist: NextTokenDist) -> NextTokenDist:
        raw = reply.json
        identity = self._verified(raw)
        if identity not in self.identities:
            logger.info("%s %s: new response identity %s", self.service, self.model, identity)
        self.identities[identity] += 1
        usage = response_usage(raw)
        details = raw["usage"].get("prompt_tokens_details")
        return dataclasses.replace(
            dist,
            api=ApiCall(
                latency_ms=reply.latency_ms,
                cost_usd=reply.cost_usd,
                response_cached=reply.cached,
                completion_tokens=usage.output_tokens,
                cache_write_tokens=_count(details, "cache_write_tokens"),
                model=identity.model,
                system_fingerprint=identity.system_fingerprint,
                provider=identity.provider,
            ),
        )

    def _reserve(self, estimate: float) -> None:
        committed = self._spent_usd + self._reserved_usd
        if self.max_usd is not None and committed + estimate > self.max_usd:
            raise ChatAPIBudgetExceeded(
                service=self.service,
                max_usd=self.max_usd,
                committed_usd=committed,
                estimated_usd=estimate,
            )
        self._reserved_usd += estimate

    def _session(self) -> tuple[httpx.AsyncClient, asyncio.Semaphore]:
        if not self._api_key:
            raise ChatAPIKeyMissing(
                f"no {self.service} API key: pass api_key or set {self.api_key_env}"
            )
        loop = asyncio.get_running_loop()
        if self._client is None or self._semaphore is None or self._loop is not loop:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=self._timeout,
                transport=self._transport,
                limits=httpx.Limits(
                    max_connections=self._max_concurrency,
                    max_keepalive_connections=self._max_concurrency,
                ),
            )
            self._semaphore = asyncio.Semaphore(self._max_concurrency)
            self._loop = loop
        return self._client, self._semaphore

    async def _send(self, payload: str) -> tuple[Any, float]:
        """One logical request, retrying per :attr:`retry` (module docstring); returns the JSON
        body and the HTTP latency of the successful attempt."""
        client, semaphore = self._session()
        data = payload.encode()
        attempts = self.retry.max_attempts
        for attempt in range(1, attempts + 1):
            await self._bucket.acquire()
            try:
                async with semaphore:
                    start = self._clock()
                    response = await client.post(CHAT_PATH, content=data)
                    latency_ms = (self._clock() - start) * 1000.0
            except httpx.TransportError as exc:
                if attempt == attempts:
                    raise ChatAPIConnectionError(
                        f"POST {self.base_url}{CHAT_PATH} failed after {attempt} attempt(s): "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                await self._sleep(self.retry.delay(attempt, None, self._rng))
                continue
            request_id = response.headers.get(self.request_id_header)
            if response.is_success:
                try:
                    return response.json(), latency_ms
                except ValueError as exc:
                    raise ChatAPIResponseError(
                        "response is not JSON", self._error_body(response)
                    ) from exc
            status = response.status_code
            body = self._error_body(response)
            where = {"attempts": attempt, "request_id": request_id, "service": self.service}
            if self.is_out_of_credit(status, body):
                raise ChatAPIQuotaExceeded(status, body, **where)
            if status in self.retryable_status and attempt < attempts:
                delay = self.retry.delay(attempt, _retry_after(response), self._rng)
                logger.warning(
                    "%s POST %s returned %d (attempt %d/%d, request %s); retrying in %.2f s",
                    self.service,
                    CHAT_PATH,
                    status,
                    attempt,
                    attempts,
                    request_id,
                    delay,
                )
                await self._sleep(delay)
                continue
            if status in (401, 403):
                raise ChatAPIAuthError(status, body, **where)
            raise ChatAPIHTTPError(status, body, **where)
        raise AssertionError("unreachable")  # pragma: no cover

    def _error_body(self, response: httpx.Response) -> Any:
        text = response.text
        if self._api_key:
            text = text.replace(self._api_key, _REDACTED)
        try:
            return json.loads(text)
        except ValueError:
            return text[:_ERROR_TEXT_LIMIT]


def _retry_after(response: httpx.Response) -> float | None:
    """``retry-after-ms`` (OpenAI's millisecond header), else ``retry-after``, in seconds."""
    millis = response.headers.get("retry-after-ms")
    if millis is not None:
        try:
            value = float(millis) / 1000.0
        except ValueError:
            value = math.nan
        if math.isfinite(value):
            return max(0.0, value)
    return parse_retry_after(response.headers.get("retry-after"))
