"""OpenAI Chat Completions backend: answer logprobs from a hosted model (``gpt-6-luna``).

Transport, cache, budget, retries, the structured read and accounting are shared with the other
chat APIs (:class:`~jevemu.backends.chat_api.ChatCompletionsBackend`). What the OpenAI API allows
(``docs/research/openai_probe_report.md``):

- **No prefill, no echo, no tokenizer.** A final assistant message is not continued, the legacy
  Completions endpoint (``echo``, prompt logprobs) is not offered, and GPT-6's tokenizer is not
  public.
- **Logprobs need reasoning off.** Every request sends ``reasoning_effort="none"``,
  ``max_completion_tokens`` (:data:`MAX_COMPLETION_TOKENS`), ``temperature=0``, ``logprobs=true``
  and ``top_logprobs`` <= :data:`TOP_LOGPROBS_MAX`; a response billing reasoning tokens is
  rejected.
- **Structured read.** The answer is asked for as ``{"answer": "<label>"}`` (Structured
  Outputs accepts only object roots). The mask is reflected in the logprobs: the list at the
  value holds only labels.
- **Pricing.** Every call's tokens are priced with :data:`jevemu.eval.costs.TOKEN_PRICES`
  (``price_id``: the model, ``:flex`` for Flex processing).
- **Identity.** A response must report the pinned model or a dated snapshot of it
  (``gpt-6-luna-2026-...``); anything else raises :class:`~jevemu.errors.ChatAPIResponseError`.
  Responses are counted per (model, ``system_fingerprint``).
- **Out of credit** is 429 ``insufficient_quota`` (:class:`~jevemu.errors.ChatAPIQuotaExceeded`,
  not retried).

The API key comes from the argument or ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import httpx

from jevemu.backends.base import Capabilities, RenderedPrompt
from jevemu.backends.chat_api import (
    STUCK_REPLY_MAX_TOKENS,
    ChatCompletionsBackend,
    ResponseIdentity,
)
from jevemu.cache import ResponseCache
from jevemu.errors import ChatAPIResponseError
from jevemu.eval.costs import TOKEN_PRICES
from jevemu.jev_client import Clock, RetryPolicy, Sleep

__all__ = [
    "API_KEY_ENV",
    "BACKEND_NAME",
    "CAPABILITIES",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "MAX_COMPLETION_TOKENS",
    "TOP_LOGPROBS_MAX",
    "OpenAIChatBackend",
    "PromptCacheMode",
    "ServiceTier",
    "build_chat_request",
    "default_cache_path",
]

BACKEND_NAME = "openai_chat"
API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_BASE_URL = "https://api.openai.com"
DEFAULT_MODEL = "gpt-6-luna"
TOP_LOGPROBS_MAX = 5
"""Measured 2026-09-25: gpt-6-luna answers ``top_logprobs`` > 5 with HTTP 400 ("must be less
than or equal to 5"), although the Chat Completions reference documents 0..20. It may also
return fewer alternatives than requested (2 of 5 on a star-rating prompt, 1 at logprob 0.0 on
an easy one)."""

CAPABILITIES = Capabilities(
    top_logprobs_max=TOP_LOGPROBS_MAX,
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

ServiceTier = Literal["default", "flex"]
PromptCacheMode = Literal["implicit", "explicit"]
"""``prompt_cache_options.mode``. On GPT-5.6 and later, ``implicit`` writes the prompt up to the
last user message to the cache (billed at 1.25x input) once it reaches 1,024 tokens;
``explicit`` without breakpoints neither reads nor writes the cache."""

MAX_COMPLETION_TOKENS = 16
"""Measured 2026-09-25: ``max_completion_tokens=1`` fails with HTTP 400 ("Could not finish the
message because max_tokens or model output limit was reached"); with 16, a one-character answer
stops by itself and bills 4 completion tokens (0 reasoning tokens), a structured reply
``{"answer":"C"}`` 11 (2026-09-27)."""


def default_cache_path() -> Path:
    """``~/.cache/jevemu/openai_cache.sqlite``."""
    return Path.home() / ".cache" / "jevemu" / "openai_cache.sqlite"


def build_chat_request(
    model: str,
    prompt: RenderedPrompt,
    *,
    top_logprobs: int,
    reasoning_effort: str = "none",
    service_tier: ServiceTier = "default",
    prompt_cache_mode: PromptCacheMode | None = None,
) -> dict[str, Any]:
    """Chat-completions body for a new assistant turn (``prompt`` ends in the user turn: the API
    opens a new turn after an assistant prefill instead of continuing it)."""
    if not 0 <= top_logprobs <= TOP_LOGPROBS_MAX:
        raise ValueError(f"top_logprobs must be in 0..{TOP_LOGPROBS_MAX}, got {top_logprobs}")
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in prompt.messages],
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "reasoning_effort": reasoning_effort,
    }
    if service_tier != "default":
        body["service_tier"] = service_tier
    if prompt_cache_mode is not None:
        body["prompt_cache_options"] = {"mode": prompt_cache_mode}
    return body


class OpenAIChatBackend(ChatCompletionsBackend):
    """``Backend`` over OpenAI Chat Completions (see the module docstring).

    ``cache=None`` uses :func:`default_cache_path`; pass ``ResponseCache.in_memory()`` to keep
    nothing on disk. ``max_usd`` caps this backend's spend. ``service_tier="flex"`` requests Flex
    processing and prices it as ``<model>:flex``. ``prompt_cache_mode`` sets
    ``prompt_cache_options.mode`` (``None``: the API default).
    """

    service = "OpenAI"
    backend_name = BACKEND_NAME
    engine_version = "openai-api"
    api_key_env = API_KEY_ENV

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        cache: ResponseCache | None = None,
        max_usd: float | None = None,
        max_rpm: int = 450,
        max_concurrency: int = 32,
        reasoning_effort: str = "none",
        service_tier: ServiceTier = "default",
        prompt_cache_mode: PromptCacheMode | None = None,
        retry: RetryPolicy | None = None,
        timeout: float | httpx.Timeout = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        price_id = model if service_tier == "default" else f"{model}:{service_tier}"
        price = TOKEN_PRICES.get(price_id)
        if price is None:
            raise ValueError(f"no token price registered for {price_id!r}")
        super().__init__(
            model,
            price=price,
            capabilities=CAPABILITIES,
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
        self.reasoning_effort = reasoning_effort
        self.service_tier: ServiceTier = service_tier
        self.prompt_cache_mode = prompt_cache_mode

    def build_body(self, prompt: RenderedPrompt, top_logprobs: int) -> dict[str, Any]:
        return build_chat_request(
            self.model,
            prompt,
            top_logprobs=top_logprobs,
            reasoning_effort=self.reasoning_effort,
            service_tier=self.service_tier,
            prompt_cache_mode=self.prompt_cache_mode,
        )

    def max_output_tokens(self, body: Mapping[str, Any]) -> int:
        return int(body.get("max_completion_tokens") or MAX_COMPLETION_TOKENS)

    def with_max_output_tokens(self, body: Mapping[str, Any], cap: int) -> dict[str, Any]:
        return {**body, "max_completion_tokens": cap}

    def health_flags(self) -> dict[str, str]:
        flags = {
            "reasoning_effort": self.reasoning_effort,
            "top_logprobs": str(TOP_LOGPROBS_MAX),
            "max_completion_tokens": str(MAX_COMPLETION_TOKENS),
            "stuck_reply_max_tokens": str(STUCK_REPLY_MAX_TOKENS),
            "constraint": "json_schema",
            "service_tier": self.service_tier,
        }
        if self.prompt_cache_mode is not None:
            flags["prompt_cache_mode"] = self.prompt_cache_mode
        return flags

    def identify(self, payload: Mapping[str, Any]) -> ResponseIdentity:
        model = str(payload.get("model", ""))
        if model != self.model and not model.startswith(self.model + "-"):
            raise ChatAPIResponseError(
                f"response reports model {model!r}, pinned {self.model!r}", payload
            )
        fingerprint = payload.get("system_fingerprint")
        return ResponseIdentity(model, None if fingerprint is None else str(fingerprint), None)

    def is_out_of_credit(self, status: int, body: Any) -> bool:
        error = body.get("error") if isinstance(body, dict) else None
        kind = error.get("type") if isinstance(error, dict) else None
        return status == 429 and kind == "insufficient_quota"
