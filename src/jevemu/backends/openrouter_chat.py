"""OpenRouter backend: answer logprobs from any OpenRouter model whose providers return them.

OpenRouter speaks the OpenAI Chat Completions dialect, so transport, cache, budget, retries and
accounting are :class:`~jevemu.backends.chat_api.ChatCompletionsBackend`'s. The model is a
parameter (default :data:`DEFAULT_MODEL`); ``docs/research/openrouter_probe_report.md`` has what
was measured.

- **Endpoints listing first.** The constructor takes the public listing
  ``GET /api/v1/models/{model}/endpoints`` (:func:`fetch_endpoints`, no key needed) and keeps the
  endpoints that list both ``logprobs`` and ``top_logprobs`` and, when ``providers`` is given,
  match one of them (``wafer`` matches the tags ``wafer`` and ``wafer/...``). None left: a
  ``ValueError`` before any chat request.
- **Price from the listing.** :func:`snapshot_price` is the highest rate over those endpoints and
  their time-of-day ``overrides`` (with one pinned provider without overrides, its exact price).
  It drives the budget guard, prices response-cache hits, and is recorded in the run manifest
  (``BackendInfo.price``), so no ``TOKEN_PRICES`` entry is needed. The spend of a network call is
  what OpenRouter billed (``usage.cost``), the snapshot price only when that is missing.
- **Request.** ``max_tokens`` (default :data:`DEFAULT_MAX_TOKENS`), ``temperature=0``,
  ``logprobs=true``, ``top_logprobs`` <= ``top_logprobs`` (default
  :data:`DEFAULT_TOP_LOGPROBS`), ``reasoning: {"enabled": false}`` and
  ``provider: {"require_parameters": true}`` (only providers that honor every parameter), plus
  ``order`` and ``allow_fallbacks: false`` when providers are pinned.
- **Structured read.** Needs a provider whose lists are its positions' own. Measured
  2026-09-27 on ``deepseek/deepseek-v4.1-flash`` (``docs/research/openrouter_probe_report.md``):
  Makora's are, and it answers greedily at ``temperature=0``; Wafer, for a token generated in
  the same decoding step as the one before it, returns a copy of an earlier position's list (a
  structured read there fails with :class:`~jevemu.errors.ChatAPIResponseError`) and sometimes
  generates a token other than its most likely one.
- **Identity.** A response must report the model or its dated canonical slug
  (``deepseek/deepseek-v4.1-flash-20260910``), bill no reasoning tokens, and, with pinned
  providers, come from one of them; anything else raises
  :class:`~jevemu.errors.ChatAPIResponseError`. Responses are counted per (model, provider).
- **Out of credit** is HTTP 402 (:class:`~jevemu.errors.ChatAPIQuotaExceeded`, not retried).

The API key comes from the argument or ``OPENROUTER_API_KEY``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from jevemu.backends.base import Capabilities, RenderedPrompt
from jevemu.backends.chat_api import (
    STUCK_REPLY_MAX_TOKENS,
    ChatCompletionsBackend,
    ResponseIdentity,
)
from jevemu.cache import ResponseCache
from jevemu.errors import ChatAPIConnectionError, ChatAPIHTTPError, ChatAPIResponseError
from jevemu.eval.costs import TokenPrice
from jevemu.jev_client import Clock, RetryPolicy, Sleep

__all__ = [
    "API_KEY_ENV",
    "BACKEND_NAME",
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_TOP_LOGPROBS",
    "LOGPROB_PARAMETERS",
    "OpenRouterChatBackend",
    "default_cache_path",
    "endpoints_path",
    "fetch_endpoints",
    "logprob_endpoints",
    "snapshot_price",
]

BACKEND_NAME = "openrouter_chat"
API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_BASE_URL = "https://openrouter.ai/api"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
DEFAULT_TOP_LOGPROBS = 20
"""The OpenAI-API convention; each provider's real cap is unmeasured (gpt-6-luna turned out to
allow 5). Lower it with ``top_logprobs`` if a provider refuses 20."""
DEFAULT_MAX_TOKENS = 16
"""Enough for a structured reply (``{ "answer": "Yes" }``: 7 to 13 completion tokens measured
on Makora); a free reply's first token is all that is read. OpenAI refused 1 for gpt-6-luna."""
LOGPROB_PARAMETERS = ("logprobs", "top_logprobs")
"""``supported_parameters`` an endpoint must list to be used."""

_CAPABILITIES = Capabilities(
    top_logprobs_max=DEFAULT_TOP_LOGPROBS,
    structured_choice=True,
    logprobs_mode="raw_logprobs",
    mask_reflected_in_logprobs=True,
    echo_prompt_logprobs=False,
    explicit_token_logprobs=False,
    assistant_prefill=False,
    prefix_caching=True,
    local_tokenizer=False,
    top_logprobs_exact=False,
)
_PER_MTOK = Decimal(1_000_000)
_CANONICAL_SEPARATOR = " | "
"""An endpoint's ``name`` is ``"<Provider> | <canonical slug>"``."""


def default_cache_path() -> Path:
    """``~/.cache/jevemu/openrouter_cache.sqlite``."""
    return Path.home() / ".cache" / "jevemu" / "openrouter_cache.sqlite"


def endpoints_path(model: str) -> str:
    """Path of ``model``'s endpoints listing, relative to the API base URL."""
    return f"/v1/models/{quote(model, safe='/')}/endpoints"


async def fetch_endpoints(
    model: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """``GET /api/v1/models/{model}/endpoints`` (public; no key)."""
    url = base_url.rstrip("/").removesuffix("/v1") + endpoints_path(model)
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        try:
            response = await client.get(url, headers={"Accept": "application/json"})
        except httpx.TransportError as exc:
            raise ChatAPIConnectionError(f"GET {url} failed: {type(exc).__name__}: {exc}") from exc
    if not response.is_success:
        raise ChatAPIHTTPError(response.status_code, response.text[:2000], service="OpenRouter")
    try:
        listing: dict[str, Any] = response.json()
    except ValueError as exc:
        raise ChatAPIResponseError("endpoints listing is not JSON", response.text[:2000]) from exc
    return listing


def _matches(tag: str, providers: Iterable[str]) -> bool:
    return any(tag == p or tag.startswith(p + "/") for p in providers)


def logprob_endpoints(
    listing: Mapping[str, Any], providers: Sequence[str] = ()
) -> list[Mapping[str, Any]]:
    """The endpoints of ``listing`` that list :data:`LOGPROB_PARAMETERS` and, if ``providers``
    is given, whose tag matches one of them."""
    data = listing.get("data")
    endpoints = data.get("endpoints") if isinstance(data, Mapping) else None
    if not isinstance(endpoints, list):
        raise ValueError("endpoints listing has no data.endpoints")
    return [
        endpoint
        for endpoint in endpoints
        if all(p in (endpoint.get("supported_parameters") or []) for p in LOGPROB_PARAMETERS)
        and (not providers or _matches(str(endpoint.get("tag", "")), providers))
    ]


def _rate(pricing: Mapping[str, Any], name: str) -> Decimal | None:
    value = pricing.get(name)
    if value is None:
        return None
    try:
        rate = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"pricing field {name!r} is not a number: {value!r}") from None
    if not rate.is_finite() or rate < 0:
        raise ValueError(f"pricing field {name!r} is not a price: {value!r}")
    return rate


def snapshot_price(
    price_id: str,
    endpoints: Sequence[Mapping[str, Any]],
    *,
    source: str,
    as_of: str,
) -> TokenPrice:
    """The highest per-token rates over ``endpoints`` and their ``overrides`` (a time window's
    prices; fields it omits keep the endpoint's), per 1M tokens.

    A tier that lists no cache-read (cache-write) rate bills those tokens as input, so the input
    rate stands in for it. Non-token fees (per request, per image) are not modelled; the billed
    ``usage.cost`` includes them.
    """
    if not endpoints:
        raise ValueError("no endpoints to price")
    inputs: list[Decimal] = []
    outputs: list[Decimal] = []
    reads: list[Decimal] = []
    writes: list[Decimal] = []
    for endpoint in endpoints:
        base = endpoint.get("pricing")
        if not isinstance(base, Mapping):
            raise ValueError(f"endpoint {endpoint.get('tag')!r} has no pricing")
        overrides = [{**base, **override} for override in base.get("overrides") or []]
        for tier in (base, *overrides):
            prompt = _rate(tier, "prompt")
            completion = _rate(tier, "completion")
            if prompt is None or completion is None:
                raise ValueError(f"endpoint {endpoint.get('tag')!r} lacks token prices")
            inputs.append(prompt)
            outputs.append(completion)
            read = _rate(tier, "input_cache_read")
            write = _rate(tier, "input_cache_write")
            reads.append(prompt if read is None else read)
            writes.append(prompt if write is None else write)
    return TokenPrice(
        price_id=price_id,
        input_usd_per_mtok=float(max(inputs) * _PER_MTOK),
        output_usd_per_mtok=float(max(outputs) * _PER_MTOK),
        cached_input_usd_per_mtok=float(max(reads) * _PER_MTOK),
        cache_write_usd_per_mtok=float(max(writes) * _PER_MTOK),
        source=source,
        as_of=as_of,
    )


class OpenRouterChatBackend(ChatCompletionsBackend):
    """``Backend`` over OpenRouter Chat Completions (see the module docstring).

    ``endpoints`` is ``model``'s endpoints listing (:func:`fetch_endpoints`). ``providers``
    pins provider slugs in order of preference, without fallbacks. ``cache=None`` uses
    :func:`default_cache_path`; pass ``ResponseCache.in_memory()`` to keep nothing on disk.
    """

    service = "OpenRouter"
    backend_name = BACKEND_NAME
    engine_version = "openrouter-api"
    api_key_env = API_KEY_ENV
    retryable_status = frozenset({408, 429, 500, 502, 503, 504})

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        endpoints: Mapping[str, Any],
        providers: Sequence[str] = (),
        top_logprobs: int = DEFAULT_TOP_LOGPROBS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        as_of: str | None = None,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        cache: ResponseCache | None = None,
        max_usd: float | None = None,
        max_rpm: int = 450,
        max_concurrency: int = 32,
        retry: RetryPolicy | None = None,
        timeout: float | httpx.Timeout = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if top_logprobs < 1:
            raise ValueError(f"top_logprobs must be >= 1, got {top_logprobs}")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        data = endpoints.get("data")
        listed = data.get("id") if isinstance(data, Mapping) else None
        if listed != model:
            raise ValueError(f"endpoints listing is for {listed!r}, not {model!r}")
        eligible = logprob_endpoints(endpoints, providers)
        if not eligible:
            wanted = f" among providers {list(providers)}" if providers else ""
            raise ValueError(
                f"no OpenRouter endpoint of {model!r}{wanted} lists "
                f"{' and '.join(LOGPROB_PARAMETERS)}"
            )
        self.providers = tuple(providers)
        self.endpoints = tuple(eligible)
        self.provider_names = frozenset(str(e.get("provider_name")) for e in eligible)
        self.top_logprobs = top_logprobs
        self.max_tokens = max_tokens
        base = base_url.rstrip("/").removesuffix("/v1")
        price = snapshot_price(
            f"openrouter:{model}@{'+'.join(self.providers) or 'any'}",
            eligible,
            source=base + endpoints_path(model),
            as_of=as_of or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        super().__init__(
            model,
            price=price,
            capabilities=dataclasses.replace(_CAPABILITIES, top_logprobs_max=top_logprobs),
            base_url=base_url,
            cache=cache if cache is not None else ResponseCache(default_cache_path()),
            owns_cache=cache is None,
            api_key=api_key,
            max_usd=max_usd,
            max_rpm=max_rpm,
            max_concurrency=max_concurrency,
            retry=retry,
            timeout=timeout,
            transport=transport,
            clock=clock,
            sleep=sleep,
        )

    def build_body(self, prompt: RenderedPrompt, top_logprobs: int) -> dict[str, Any]:
        routing: dict[str, Any] = {"require_parameters": True}
        if self.providers:
            routing["order"] = list(self.providers)
            routing["allow_fallbacks"] = False
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in prompt.messages],
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "reasoning": {"enabled": False},
            "provider": routing,
        }

    def max_output_tokens(self, body: Mapping[str, Any]) -> int:
        return int(body.get("max_tokens") or self.max_tokens)

    def with_max_output_tokens(self, body: Mapping[str, Any], cap: int) -> dict[str, Any]:
        return {**body, "max_tokens": cap}

    def health_flags(self) -> dict[str, str]:
        return {
            "reasoning": "disabled",
            "top_logprobs": str(self.top_logprobs),
            "max_tokens": str(self.max_tokens),
            "stuck_reply_max_tokens": str(STUCK_REPLY_MAX_TOKENS),
            "constraint": "json_schema",
            "providers": ",".join(self.providers) or "any",
        }

    def model_revision(self) -> str | None:
        """The canonical (dated) slug the eligible endpoints serve, if they agree."""
        slugs = {str(e.get("name", "")).partition(_CANONICAL_SEPARATOR)[2] for e in self.endpoints}
        return slugs.pop() if len(slugs) == 1 and "" not in slugs else None

    def price_snapshot(self) -> Mapping[str, Any]:
        return dataclasses.asdict(self.price)

    def identify(self, payload: Mapping[str, Any]) -> ResponseIdentity:
        model = str(payload.get("model", ""))
        if model != self.model and not model.startswith(self.model + "-"):
            raise ChatAPIResponseError(
                f"response reports model {model!r}, requested {self.model!r}", payload
            )
        provider = payload.get("provider")
        provider = None if provider is None else str(provider)
        if self.providers and provider is not None and provider not in self.provider_names:
            raise ChatAPIResponseError(
                f"response served by {provider!r}, pinned {sorted(self.provider_names)}", payload
            )
        fingerprint = payload.get("system_fingerprint")
        return ResponseIdentity(model, None if fingerprint is None else str(fingerprint), provider)

    def is_out_of_credit(self, status: int, body: Any) -> bool:
        return status == 402

    def call_cost(self, payload: Mapping[str, Any]) -> float:
        """What OpenRouter billed (``usage.cost``); the snapshot price if it is missing."""
        usage = payload.get("usage")
        cost = usage.get("cost") if isinstance(usage, Mapping) else None
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
            return float(cost)
        return super().call_cost(payload)
