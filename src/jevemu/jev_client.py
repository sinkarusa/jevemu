"""Client for the real Jev System One API (``POST https://api.typesafe.ai/v1/systemone``).

- **Version pin.** Every request is sent with the client's versioned model id (default
  ``jev-1.13.0``), whatever ``request.model`` says; aliases such as ``jev-latest`` are refused.
  A response reporting another model raises :class:`~jevemu.errors.JevVersionDrift`
  (``strict_version=False`` logs a warning instead).
- **Cache first.** :class:`~jevemu.cache.ResponseCache` is read before any network call; a hit
  needs no API key, costs 0 and is flagged in the :class:`JevCallRecord`.
- **Resilience.** 429 and 529 are retried with exponential backoff and jitter, never sooner
  than ``retry-after``. A :class:`TokenBucket` keeps request starts at or below ``max_rpm``.
  401, 400/422 (validation) and every other error status raise immediately.
- **Accounting.** Per call: client-measured latency (monotonic clock), usage tokens and cost
  (``input_tokens`` x $0.042 / 1M). ``max_usd`` refuses a request before sending it if a
  pessimistic estimate (:func:`estimate_input_tokens`) could push spend past the budget.

The API key is read from the argument or ``TYPESAFE_API_KEY`` and is never logged, stored in
the cache or records, or included in exception messages. :func:`load_env_file` loads a local
``.env`` for scripts and tests.

Other servers of the same wire format reuse the client by overriding its class attributes
(:attr:`JevClient.service`, ``api_key_env``, ``api_key_required``, ``model_pattern``,
``usd_per_input_token``): :class:`jevemu.clm_client.ClmClient` drives a local CLM server.
"""

from __future__ import annotations

import asyncio
import email.utils
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from jevemu.cache import CacheEntry, ResponseCache, cache_key, canonical_json
from jevemu.errors import (
    JevAPIKeyMissing,
    JevAuthError,
    JevBudgetExceeded,
    JevConnectionError,
    JevHTTPError,
    JevOverloaded,
    JevRateLimited,
    JevResponseError,
    JevValidationError,
    JevVersionDrift,
)
from jevemu.types import (
    DEFAULT_MODEL,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
)

__all__ = [
    "API_KEY_ENV",
    "DEFAULT_BASE_URL",
    "USD_PER_INPUT_TOKEN",
    "AnswerDrift",
    "JevCallRecord",
    "JevClient",
    "JevModelInfo",
    "NondeterminismReport",
    "RetryPolicy",
    "TokenBucket",
    "cost_usd",
    "estimate_input_tokens",
    "load_env_file",
    "parse_retry_after",
    "probe_nondeterminism",
    "read_env_file",
    "wire_body",
]

logger = logging.getLogger(__name__)

API_KEY_ENV = "TYPESAFE_API_KEY"
DEFAULT_BASE_URL = "https://api.typesafe.ai"
SYSTEMONE_PATH = "/v1/systemone"
MODELS_PATH = "/v1/models"
REQUEST_ID_HEADER = "x-typesafe-request-id"

USD_PER_INPUT_TOKEN = 0.042 / 1_000_000
"""Jev price: $0.042 per 1M input tokens; output tokens are free."""

ESTIMATE_OVERHEAD_TOKENS = 512
ESTIMATE_BYTES_PER_TOKEN = 2
"""Budget-guard estimate ``512 + ceil(bytes / 2)`` of the compact request JSON. A least-squares
fit over the recorded jev-1.13.0 fixtures gives billed ``input_tokens`` ~ 229 + 5 per question +
0.32 per byte, so the estimate is 1.7-2.1x the billed count there (checked by the golden tests).
Deliberately pessimistic: the guard must refuse before the budget can be exceeded."""

_VERSIONED_MODEL = re.compile(r"jev-\d+\.\d+\.\d+")
_RETRYABLE: dict[int, type[JevRateLimited] | type[JevOverloaded]] = {
    429: JevRateLimited,
    529: JevOverloaded,
}
_ERROR_TEXT_LIMIT = 2000
_REDACTED = "***"

Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]


# --- pure helpers ------------------------------------------------------------------------------


def cost_usd(input_tokens: int, usd_per_input_token: float = USD_PER_INPUT_TOKEN) -> float:
    return input_tokens * usd_per_input_token


def estimate_input_tokens(canonical_request: str) -> int:
    """Pessimistic billed-token estimate for the budget guard (see ``ESTIMATE_*``)."""
    n_bytes = len(canonical_request.encode())
    return ESTIMATE_OVERHEAD_TOKENS + math.ceil(n_bytes / ESTIMATE_BYTES_PER_TOKEN)


def wire_body(request: SystemOneRequest, model: str) -> dict[str, Any]:
    """The JSON body sent to Jev: ``request`` pinned to ``model``, Noul ``criteria`` omitted
    when absent (Jev types it as an optional object, not nullable)."""
    body = request.model_dump(mode="json")
    body["model"] = model
    for question in body["questions"].values():
        if question["type"] == "noul" and question.get("criteria") is None:
            question.pop("criteria", None)
    return body


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Seconds to wait from a ``retry-after`` header (delta-seconds or HTTP-date)."""
    if value is None:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - (now or datetime.now(timezone.utc))).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def read_env_file(path: str | os.PathLike[str] = ".env") -> dict[str, str]:
    """Parse a dotenv file without touching ``os.environ``; a missing file gives ``{}``.

    Supports blank lines, ``#`` comments, an ``export`` prefix, single/double-quoted values and
    trailing `` # comments`` on unquoted values. Malformed lines are skipped.
    """
    file = Path(path)
    if not file.is_file():
        return {}
    values = {}
    for raw in file.read_text("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        name, sep, value = line.partition("=")
        name = name.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        values[name] = value
    return values


def load_env_file(path: str | os.PathLike[str] = ".env", *, override: bool = False) -> list[str]:
    """Copy :func:`read_env_file` values into ``os.environ`` (existing variables win unless
    ``override``). Returns the names that were set, never their values."""
    loaded = []
    for name, value in read_env_file(path).items():
        if override or name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


# --- resilience --------------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """Retries for 429/529. Attempt ``n`` (1-based) that fails waits
    ``max(retry_after, b / 2) + uniform(0, b / 2)`` with ``b = min(max_backoff_s,
    initial_backoff_s * 2 ** (n - 1))``: equal-jitter exponential backoff, never sooner than
    the server's ``retry-after``."""

    max_attempts: int = 8
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 60.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.initial_backoff_s < 0 or self.max_backoff_s < 0:
            raise ValueError("backoff times must be >= 0")

    def delay(self, attempt: int, retry_after: float | None, rng: random.Random) -> float:
        base = min(self.max_backoff_s, self.initial_backoff_s * 2.0 ** (attempt - 1))
        return max(retry_after or 0.0, base / 2) + rng.uniform(0.0, base / 2)


class TokenBucket:
    """Async rate limiter (token bucket in GCRA form): at most ``burst`` immediate starts,
    then one start every ``60 / rate_per_minute`` seconds. Slots are reserved without
    awaiting, so concurrent tasks on one event loop are spaced correctly."""

    def __init__(
        self,
        rate_per_minute: float,
        *,
        burst: int = 1,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if rate_per_minute <= 0:
            raise ValueError(f"rate_per_minute must be > 0, got {rate_per_minute}")
        if burst < 1:
            raise ValueError(f"burst must be >= 1, got {burst}")
        self.interval = 60.0 / rate_per_minute
        self._tolerance = (burst - 1) * self.interval
        self._clock = clock
        self._sleep = sleep
        self._tat = -math.inf
        """Theoretical arrival time of the next request."""

    async def acquire(self) -> float:
        """Wait for a slot; returns the seconds waited."""
        now = self._clock()
        tat = max(self._tat, now)
        wait = max(0.0, tat - self._tolerance - now)
        self._tat = tat + self.interval
        if wait > 0:
            await self._sleep(wait)
        return wait


# --- records -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class JevCallRecord:
    """Accounting for one :meth:`JevClient.system_one_with_meta` call."""

    cache_key: str
    model: str
    """Model id reported by the response."""
    cached: bool
    """Served from the cache: no network call, ``cost_usd`` is 0."""
    latency_ms: float
    """Round trip of the HTTP attempt that produced the response (monotonic clock); for a
    cache hit, the latency recorded when the response was first fetched."""
    wall_ms: float
    """This call's wall time, including rate-limit waits, retries and backoff."""
    input_tokens: int
    output_tokens: int
    cost_usd: float
    """Spend incurred by this call."""
    attempts: int
    """HTTP attempts made (0 for a cache hit)."""
    created_at: datetime
    """When Jev produced the response (UTC)."""
    request_id: str | None
    """Jev's ``x-typesafe-request-id`` (None for a cache hit)."""


@dataclass(frozen=True)
class _Reply:
    json: Any
    attempts: int
    latency_ms: float
    request_id: str | None


class JevModelInfo(BaseModel):
    """One entry of ``GET /v1/models``."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str
    release_date: str


class _ModelList(BaseModel):
    model_config = ConfigDict(extra="allow")

    models: list[JevModelInfo]


# --- client ------------------------------------------------------------------------------------


class JevClient:
    """``SystemOneClient`` for the real Jev API with pinning, caching, retries and accounting.

    ``cache=None`` uses the on-disk default (``~/.cache/jevemu/jev_cache.sqlite``); pass
    ``ResponseCache.in_memory()`` to keep nothing on disk. ``max_usd`` caps this client's
    spend; ``max_rpm=None`` turns the rate limit off. ``clock``/``sleep`` drive latency
    measurement, rate limiting and backoff and are injectable for tests and simulation.

    One ``httpx.AsyncClient`` exists per event loop, so the client survives repeated
    ``asyncio.run`` calls. Use ``async with`` or :meth:`aclose`.
    """

    service: ClassVar[str] = "Jev"
    """Server name in log lines and error messages."""
    api_key_env: ClassVar[str] = API_KEY_ENV
    api_key_required: ClassVar[bool] = True
    """``False``: without a key, requests go out with no ``Authorization`` header."""
    model_pattern: ClassVar[re.Pattern[str]] = _VERSIONED_MODEL
    """The model ids the client may pin (aliases such as ``jev-latest`` do not match)."""
    usd_per_input_token: ClassVar[float] = USD_PER_INPUT_TOKEN

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        cache: ResponseCache | None = None,
        max_rpm: int | None = 1200,
        strict_version: bool = True,
        max_usd: float | None = None,
        retry: RetryPolicy | None = None,
        timeout: float | httpx.Timeout = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if not self.model_pattern.fullmatch(model):
            raise ValueError(
                f"model must be a versioned {self.service} id like {DEFAULT_MODEL!r}, got {model!r}"
            )
        if max_rpm is not None and max_rpm < 1:
            raise ValueError(f"max_rpm must be >= 1, got {max_rpm}")
        if max_usd is not None and max_usd < 0:
            raise ValueError(f"max_usd must be >= 0, got {max_usd}")
        self.model = model
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.cache = cache if cache is not None else self._default_cache()
        self.strict_version = strict_version
        self.max_usd = max_usd
        self.retry = retry or RetryPolicy()
        self._owns_cache = cache is None
        self._api_key = api_key or os.environ.get(self.api_key_env) or None
        self._timeout = timeout
        self._transport = transport
        self._clock = clock
        self._sleep = sleep
        self._bucket = None if max_rpm is None else TokenBucket(max_rpm, clock=clock, sleep=sleep)
        self._rng = random.Random()
        self._spent_usd = 0.0
        self._reserved_usd = 0.0
        self._network_calls = 0
        self._cache_hits = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: httpx.AsyncClient | None = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.model!r}, base_url={self.base_url!r}, "
            f"cache={self.cache!r}, max_usd={self.max_usd!r}, spent_usd={self._spent_usd:.6f})"
        )

    @property
    def spent_usd(self) -> float:
        """Actual spend so far (billed ``input_tokens`` of every response received)."""
        return self._spent_usd

    @property
    def network_calls(self) -> int:
        """Responses received over the network (retried attempts not counted)."""
        return self._network_calls

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    async def __aenter__(self) -> JevClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        client, self._client, self._loop = self._client, None, None
        if client is not None:
            await client.aclose()
        if self._owns_cache:
            self.cache.close()

    # -- public API --

    async def system_one(self, request: SystemOneRequest) -> SystemOneResponse:
        response, _ = await self.system_one_with_meta(request)
        return response

    async def system_one_with_meta(
        self, request: SystemOneRequest, *, bypass_cache: bool = False
    ) -> tuple[SystemOneResponse, JevCallRecord]:
        """Answer ``request`` and report latency, tokens and cost.

        ``bypass_cache`` neither reads nor writes the cache (nondeterminism probe): the call
        always goes to the network and the stored answer is left untouched.
        """
        started = self._clock()
        body = wire_body(request, self.model)
        payload = canonical_json(body)
        key = cache_key(payload, self.model)

        if not bypass_cache:
            entry = self.cache.get(key)
            if entry is not None:
                response = self._validate(entry.response)
                self._cache_hits += 1
                record = JevCallRecord(
                    cache_key=key,
                    model=response.model,
                    cached=True,
                    latency_ms=entry.latency_ms,
                    wall_ms=(self._clock() - started) * 1000.0,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    cost_usd=0.0,
                    attempts=0,
                    created_at=entry.created_at,
                    request_id=None,
                )
                logger.debug("%s cache hit %s", self.service, key[:12])
                return response, record

        estimate = cost_usd(estimate_input_tokens(payload), self.usd_per_input_token)
        self._reserve(estimate)
        try:
            reply = await self._send("POST", SYSTEMONE_PATH, payload)
        finally:
            self._reserved_usd -= estimate
        created_at = datetime.now(timezone.utc)
        self._network_calls += 1
        call_cost = cost_usd(_billed_tokens(reply.json), self.usd_per_input_token)
        self._spent_usd += call_cost

        response = self._validate(reply.json)
        if not bypass_cache:
            self.cache.put(
                CacheEntry(
                    key=key,
                    model=self.model,
                    request=body,
                    response=reply.json,
                    latency_ms=reply.latency_ms,
                    created_at=created_at,
                )
            )
        record = JevCallRecord(
            cache_key=key,
            model=response.model,
            cached=False,
            latency_ms=reply.latency_ms,
            wall_ms=(self._clock() - started) * 1000.0,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=call_cost,
            attempts=reply.attempts,
            created_at=created_at,
            request_id=reply.request_id,
        )
        logger.debug(
            "%s call %s (%s): %.1f ms, %d attempt(s), %d input tokens, $%.8f",
            self.service,
            key[:12],
            reply.request_id,
            reply.latency_ms,
            reply.attempts,
            record.input_tokens,
            call_cost,
        )
        return response, record

    async def list_models(self) -> list[JevModelInfo]:
        """``GET /v1/models`` (lists aliases; versioned ids are accepted regardless)."""
        raw = (await self._send("GET", MODELS_PATH, None)).json
        try:
            return _ModelList.model_validate(raw).models
        except ValidationError as exc:
            raise JevResponseError("unexpected /v1/models body", raw) from exc

    # -- internals --

    def _default_cache(self) -> ResponseCache:
        """The cache used when ``cache=None`` (closed by :meth:`aclose`)."""
        return ResponseCache()

    def _validate(self, raw: Any) -> SystemOneResponse:
        try:
            response = SystemOneResponse.model_validate(raw)
        except ValidationError as exc:
            raise JevResponseError("response is not a valid SystemOneResponse", raw) from exc
        if response.model != self.model:
            if self.strict_version:
                raise JevVersionDrift(self.model, response.model, service=self.service)
            logger.warning(
                "%s answered with model %r, pinned %r", self.service, response.model, self.model
            )
        return response

    def _reserve(self, estimate: float) -> None:
        committed = self._spent_usd + self._reserved_usd
        if self.max_usd is not None and committed + estimate > self.max_usd:
            raise JevBudgetExceeded(
                max_usd=self.max_usd, committed_usd=committed, estimated_usd=estimate
            )
        self._reserved_usd += estimate

    def _session(self) -> httpx.AsyncClient:
        if self.api_key_required and not self._api_key:
            raise JevAPIKeyMissing(
                f"no {self.service} API key: pass api_key or set {self.api_key_env}"
            )
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self._timeout,
                transport=self._transport,
            )
            self._loop = loop
        return self._client

    async def _send(self, method: str, path: str, content: str | None) -> _Reply:
        """One logical request, retrying 429/529 per :attr:`retry`."""
        client = self._session()
        data = content.encode() if content is not None else None
        for attempt in range(1, self.retry.max_attempts + 1):
            if self._bucket is not None:
                await self._bucket.acquire()
            start = self._clock()
            try:
                response = await client.request(method, path, content=data)
            except httpx.TransportError as exc:
                raise JevConnectionError(
                    f"{method} {self.base_url}{path} failed: {type(exc).__name__}: {exc}"
                ) from exc
            latency_ms = (self._clock() - start) * 1000.0
            request_id = response.headers.get(REQUEST_ID_HEADER)
            if response.is_success:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise JevResponseError(
                        "response is not JSON", self._error_body(response)
                    ) from exc
                return _Reply(payload, attempt, latency_ms, request_id)
            status = response.status_code
            retry_after = parse_retry_after(response.headers.get("retry-after"))
            error_cls = _RETRYABLE.get(status)
            if error_cls is not None and attempt < self.retry.max_attempts:
                delay = self.retry.delay(attempt, retry_after, self._rng)
                logger.warning(
                    "%s %s %s returned %d (attempt %d/%d, request %s); retrying in %.2f s",
                    self.service,
                    method,
                    path,
                    status,
                    attempt,
                    self.retry.max_attempts,
                    request_id,
                    delay,
                )
                await self._sleep(delay)
                continue
            body = self._error_body(response)
            if error_cls is not None:
                raise error_cls(
                    status,
                    body,
                    attempts=attempt,
                    request_id=request_id,
                    retry_after=retry_after,
                )
            if status == 401:
                raise JevAuthError(status, body, attempts=attempt, request_id=request_id)
            if status in (400, 422):
                raise JevValidationError(status, body, attempts=attempt, request_id=request_id)
            raise JevHTTPError(status, body, attempts=attempt, request_id=request_id)
        raise AssertionError("unreachable")  # pragma: no cover

    def _error_body(self, response: httpx.Response) -> Any:
        text = response.text
        if self._api_key:
            text = text.replace(self._api_key, _REDACTED)
        try:
            return json.loads(text)
        except ValueError:
            return text[:_ERROR_TEXT_LIMIT]


def _billed_tokens(raw: Any) -> int:
    """``usage.input_tokens`` from a raw body, 0 if absent (the response is rejected later)."""
    try:
        tokens = raw["usage"]["input_tokens"]
    except (KeyError, TypeError):
        return 0
    return tokens if isinstance(tokens, int) and tokens >= 0 else 0


# --- nondeterminism probe ----------------------------------------------------------------------


@dataclass(frozen=True)
class AnswerDrift:
    """Spread of one answer across repeated queries of the same request."""

    cache_key: str
    question_id: str
    type: str
    n_samples: int
    max_abs_delta: float
    """Largest ``max - min`` of any outcome's probability across samples (Noul: P(yes))."""


@dataclass(frozen=True)
class NondeterminismReport:
    requests_total: int
    requests_sampled: int
    repeats: int
    answers: tuple[AnswerDrift, ...]
    cost_usd: float

    @property
    def max_abs_delta(self) -> float:
        return max((a.max_abs_delta for a in self.answers), default=0.0)


def _outcome_probabilities(answer: ChoiceAnswer | ScoreAnswer | NoulAnswer) -> dict[str, float]:
    if isinstance(answer, NoulAnswer):
        return {"true": answer.noul}
    return dict(answer.probabilities)


def _drift(key: str, responses: Sequence[SystemOneResponse]) -> list[AnswerDrift]:
    drifts = []
    for qid, first in responses[0].answers.items():
        samples = [_outcome_probabilities(r.answers[qid]) for r in responses if qid in r.answers]
        outcomes = {o for s in samples for o in s}
        delta = max(
            (max(s.get(o, 0.0) for s in samples) - min(s.get(o, 0.0) for s in samples))
            for o in outcomes
        )
        drifts.append(AnswerDrift(key, qid, first.type, len(samples), round(delta, 12)))
    return drifts


async def probe_nondeterminism(
    client: JevClient,
    requests: Sequence[SystemOneRequest],
    *,
    fraction: float = 0.05,
    repeats: int = 3,
    seed: int = 0,
) -> NondeterminismReport:
    """Re-query ``ceil(fraction * len(requests))`` requests (seeded sample) ``repeats`` times
    with the cache bypassed and report each answer's max |delta p| across the cached answer
    (fetched and cached first if missing) and the fresh ones."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    n_sample = math.ceil(fraction * len(requests))
    picked = sorted(random.Random(seed).sample(range(len(requests)), n_sample))

    async def one(request: SystemOneRequest) -> tuple[list[AnswerDrift], float]:
        calls = [client.system_one_with_meta(request)] + [
            client.system_one_with_meta(request, bypass_cache=True) for _ in range(repeats)
        ]
        results = await asyncio.gather(*calls)
        key = results[0][1].cache_key
        return _drift(key, [r for r, _ in results]), sum(rec.cost_usd for _, rec in results)

    outcomes = await asyncio.gather(*(one(requests[i]) for i in picked))
    return NondeterminismReport(
        requests_total=len(requests),
        requests_sampled=n_sample,
        repeats=repeats,
        answers=tuple(d for drifts, _ in outcomes for d in drifts),
        cost_usd=sum(cost for _, cost in outcomes),
    )
