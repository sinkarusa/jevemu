"""vLLM OpenAI-compatible HTTP backend. Pure HTTP over ``httpx``; never imports ``vllm``.

Endpoints used (vLLM >= 0.12; recorded against 0.30.0):

- Prompt tokens: every request scores exact token ids, built in two parts. The conversation
  before the assistant prefill is rendered by ``POST /tokenize`` with the chat template's
  generation prompt (``add_generation_prompt=true``, server and request chat-template kwargs
  applied), exactly as the model's own reply would start. The prefill (``Answer:``, plus any
  continuation) is then tokenized as plain text and appended, its trailing whitespace as the
  whitespace's own tokens. The template never sees the prefill: rendering it as a continued
  assistant message (``continue_final_message=true``) drops whatever the template emits only
  on the generation prompt (Gemma 4 12B/26B-A4B with thinking off: the empty thought channel
  ``<|channel>thought\\n<channel|>``), and some templates trim the final assistant message
  (Qwen3.5/3.8 render ``Answer: `` as ``Answer:``). ``jevemu.backends.probe`` records whether
  the two renders differ for the served model.
- First token: those ids go to ``POST /v1/completions`` with ``max_tokens=1``, greedy,
  ``logprobs=k``. A label constraint is sent as the top-level
  ``structured_outputs={"choice": [...]}`` field, which the completions endpoint enforces as
  the chat endpoint does. ``guided_*`` fields were removed in vLLM 0.12 and are silently
  ignored by newer servers, so no builder here ever emits one.
- Echo scoring: the same ids plus the continuation's are scored by ``POST /v1/completions``
  with ``echo=true, logprobs=1, max_tokens=0``. vLLM accepts ``max_tokens=0`` with ``echo`` and
  returns only prompt logprobs. Requests that ask for prompt logprobs skip *reading* the
  prefix cache in vLLM's V1 engine, so echo calls recompute the shared prefix.
- ``POST /tokenize``, ``POST /detokenize``, ``GET /version``, ``GET /v1/models``.

Token ids: vLLM's logprobs carry either the token text or, with ``return_tokens_as_token_ids``,
a ``"token_id:N"`` placeholder, never both. The completions top-k is a map, so keyed by text it
keeps only one of several tokens that decode alike (Gemma's byte token ``<0x41>`` next to
``A``). First-token requests therefore ask for ids, and each id's text comes from
``/detokenize`` (cached); scoring adds same-text tokens.

vLLM clamps ``-inf`` logprobs to ``-9999.0`` in every response; parsers map values at or
below that floor back to ``-inf``.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any

import httpx

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
    "BACKEND_NAME",
    "COMPLETIONS_PATH",
    "DEFAULT_CAPABILITIES",
    "DEFAULT_CHAT_TEMPLATE_KWARGS_ENV",
    "DETOKENIZE_PATH",
    "IMAGE_ENV",
    "LOGPROB_FLOOR",
    "METRICS_PATH",
    "MODELS_PATH",
    "MODEL_REVISION_ENV",
    "TOKENIZE_PATH",
    "VERSION_PATH",
    "VLLMError",
    "VLLMHTTPBackend",
    "VLLMHTTPError",
    "build_chat_tokenize_request",
    "build_detokenize_request",
    "build_echo_request",
    "build_first_token_request",
    "build_tokenize_request",
    "first_token_ids",
    "parse_echo_response",
    "parse_first_token_response",
    "to_logprob",
]

BACKEND_NAME = "vllm_http"

LOGPROB_FLOOR = -9999.0
"""vLLM reports ``-inf`` logprobs as this value."""

IMAGE_ENV = "JEVEMU_VLLM_IMAGE"
"""Environment variable naming the server's container image (reference pinned by digest)."""

MODEL_REVISION_ENV = "JEVEMU_VLLM_MODEL_REVISION"
"""Environment variable naming the served model's HF revision (commit SHA)."""

DEFAULT_CHAT_TEMPLATE_KWARGS_ENV = "JEVEMU_VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS"
"""Environment variable holding the server's ``--default-chat-template-kwargs`` (a JSON object,
empty for none), as ``scripts/serve_vllm.sh`` launched it. vLLM applies it to every chat and
``/tokenize`` render but does not expose it over HTTP."""

DEFAULT_CAPABILITIES = Capabilities(
    top_logprobs_max=20,
    structured_choice=True,
    logprobs_mode="raw_logprobs",
    mask_reflected_in_logprobs=False,
    echo_prompt_logprobs=True,
    explicit_token_logprobs=False,
    assistant_prefill=True,
    prefix_caching=False,
)
"""Documented vLLM server defaults before probing (``--max-logprobs`` 20, raw logprobs).

The mask is never assumed to be reflected. Replace with ``jevemu.backends.probe``'s measured
capabilities before trusting any strategy choice.
"""

COMPLETIONS_PATH = "/v1/completions"
TOKENIZE_PATH = "/tokenize"
DETOKENIZE_PATH = "/detokenize"
VERSION_PATH = "/version"
MODELS_PATH = "/v1/models"
METRICS_PATH = "/metrics"

_RETRY_DELAYS = (0.2, 0.8)
"""Sleep before each retry; retries cover connection failures and 5xx only."""

_RETRYABLE_ERRORS = (httpx.NetworkError, httpx.ConnectTimeout, httpx.RemoteProtocolError)

_SNAPSHOT_RE = re.compile(r"/snapshots/([0-9a-f]{40})(?:/|$)")
_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


class VLLMError(RuntimeError):
    """The vLLM server was unreachable or returned something the adapter cannot use."""


class VLLMHTTPError(VLLMError):
    """The vLLM server answered with an HTTP error status."""

    def __init__(self, status_code: int, path: str, detail: str) -> None:
        super().__init__(f"{path} -> HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.path = path
        self.detail = detail


def to_logprob(value: float) -> float:
    """Map vLLM's ``-9999.0`` floor (and anything below it) to ``-inf``."""
    return -math.inf if value <= LOGPROB_FLOOR else float(value)


# --- request builders (pure) ---------------------------------------------------------------


def _split_trailing_whitespace(text: str) -> tuple[str, str]:
    """``text`` without its trailing whitespace, and that whitespace (all-whitespace text is
    kept whole)."""
    kept = text.rstrip()
    return (kept, text[len(kept) :]) if kept else (text, "")


def _conversation(prompt: RenderedPrompt) -> list[dict[str, str]]:
    """Chat messages before the assistant prefill (all of them for a prompt without one)."""
    messages = prompt.messages[:-1] if prompt.has_prefill else prompt.messages
    return [{"role": m.role, "content": m.content} for m in messages]


def _assistant_text(prompt: RenderedPrompt, continuation: str = "") -> str:
    """Prefill (empty for a prompt without one) + ``continuation``."""
    return (prompt.prefill if prompt.has_prefill else "") + continuation


def _choice(allowed: Sequence[str]) -> dict[str, list[str]]:
    choices = list(allowed)
    if not choices or not all(choices):
        raise ValueError("allowed must be a non-empty sequence of non-empty strings")
    return {"choice": choices}


def build_first_token_request(
    model: str,
    token_ids: Sequence[int],
    *,
    allowed: Sequence[str] | None,
    top_logprobs: int,
) -> dict[str, Any]:
    """Completions body for the greedy first token after exact prompt ``token_ids`` (from
    :meth:`VLLMHTTPBackend.tokenize_chat`). Logprobs are keyed by token id: the completions
    top-k is a map, and keyed by text it keeps one entry of several tokens decoding alike
    (Gemma's byte tokens ``<0x41>`` next to ``A``), the lowest."""
    if top_logprobs < 0:
        raise ValueError(f"top_logprobs must be >= 0, got {top_logprobs}")
    body: dict[str, Any] = {
        "model": model,
        "prompt": list(token_ids),
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": top_logprobs,
        "return_tokens_as_token_ids": True,
    }
    if allowed is not None:
        body["structured_outputs"] = _choice(allowed)
    return body


def build_chat_tokenize_request(
    model: str,
    prompt: RenderedPrompt,
    *,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """``/tokenize`` body rendering the conversation before ``prompt``'s assistant prefill,
    ending in the chat template's generation prompt; the prefill is not part of it."""
    body: dict[str, Any] = {
        "model": model,
        "messages": _conversation(prompt),
        "add_generation_prompt": True,
        "add_special_tokens": False,
    }
    if chat_template_kwargs:
        body["chat_template_kwargs"] = dict(chat_template_kwargs)
    return body


def build_tokenize_request(model: str, text: str) -> dict[str, Any]:
    """``/tokenize`` body for bare text (labels, continuations): no special tokens."""
    return {"model": model, "prompt": text, "add_special_tokens": False}


def build_detokenize_request(model: str, token_ids: Sequence[int]) -> dict[str, Any]:
    return {"model": model, "tokens": list(token_ids)}


def build_echo_request(model: str, token_ids: Sequence[int]) -> dict[str, Any]:
    """Completions body returning the logprob of every prompt token and generating nothing."""
    return {
        "model": model,
        "prompt": list(token_ids),
        "echo": True,
        "logprobs": 1,
        "max_tokens": 0,
        "temperature": 0.0,
    }


# --- response parsers (pure) ---------------------------------------------------------------


def _token_id(key: str) -> int:
    """Id in vLLM's ``"token_id:N"`` placeholder (``return_tokens_as_token_ids``)."""
    prefix, _, number = key.partition(":")
    if prefix != "token_id" or not number.lstrip("-").isdigit():
        raise VLLMError(f"expected a 'token_id:N' logprob key, got {key!r}")
    return int(number)


def first_token_ids(payload: Mapping[str, Any]) -> list[int]:
    """Token ids in a response built by :func:`build_first_token_request` (sampled token first,
    then the top-k), whose texts :func:`parse_first_token_response` needs."""
    choice = payload["choices"][0]
    logprobs = choice.get("logprobs") or {}
    tokens = logprobs.get("tokens") or []
    if not tokens:
        raise VLLMError(
            "completion returned no token logprobs "
            f"(finish_reason={choice.get('finish_reason')!r}, text={choice.get('text')!r})"
        )
    alternatives: Mapping[str, float] = (logprobs.get("top_logprobs") or [None])[0] or {}
    return [_token_id(tokens[0]), *(_token_id(key) for key in alternatives)]


def parse_first_token_response(
    payload: Mapping[str, Any],
    *,
    constrained: bool,
    top_logprobs: int,
    texts: Mapping[int, str],
) -> NextTokenDist:
    """Parse a completions response built by :func:`build_first_token_request`; ``texts`` maps
    every id of :func:`first_token_ids` to its text.

    Completions report the top-k as a ``{"token_id:N": logprob}`` map, which may add the
    sampled token to the ``top_logprobs`` requested; ``top`` keeps the best ``top_logprobs``
    entries.
    """
    ids = first_token_ids(payload)
    logprobs = payload["choices"][0]["logprobs"]
    alternatives: Mapping[str, float] = (logprobs.get("top_logprobs") or [None])[0] or {}
    top = sorted(
        (
            TokenLogprob(token=texts[i], token_id=i, logprob=to_logprob(v))
            for i, v in zip(ids[1:], alternatives.values(), strict=True)
        ),
        key=lambda t: t.logprob,
        reverse=True,
    )[:top_logprobs]
    sampled = TokenLogprob(
        token=texts[ids[0]], token_id=ids[0], logprob=to_logprob(logprobs["token_logprobs"][0])
    )
    return _next_token_dist(payload, tuple(top), sampled, constrained=constrained)


def _next_token_dist(
    payload: Mapping[str, Any],
    top: tuple[TokenLogprob, ...],
    sampled: TokenLogprob,
    *,
    constrained: bool,
) -> NextTokenDist:
    usage = payload["usage"]
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    return NextTokenDist(
        top=top,
        sampled=sampled,
        constrained=constrained,
        prompt_tokens=int(usage["prompt_tokens"]),
        cached_tokens=None if cached is None else int(cached),
    )


def parse_echo_response(
    payload: Mapping[str, Any], *, continuation: str, start: int, stop: int
) -> SeqScore:
    """Slice prompt tokens ``[start, stop)`` out of an echo response built by
    :func:`build_echo_request`.

    Position 0 carries vLLM's ``null`` logprob (nothing precedes it), so ``start`` must be at
    least 1; the continuation always follows a non-empty chat-template prefix.
    """
    if not 1 <= start <= stop:
        raise ValueError(f"need 1 <= start <= stop, got start={start}, stop={stop}")
    logprobs = payload["choices"][0]["logprobs"]
    tokens: list[str] = logprobs["tokens"]
    values: list[float | None] = logprobs["token_logprobs"]
    if len(tokens) < stop or len(values) < stop:
        raise VLLMError(f"echo returned {len(values)} prompt logprobs; expected at least {stop}")
    picked = values[start:stop]
    if any(v is None for v in picked):
        raise VLLMError("echo returned a null logprob inside the continuation")
    usage = payload.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    return SeqScore(
        continuation=continuation,
        tokens=tuple(tokens[start:stop]),
        token_logprobs=tuple(to_logprob(v) for v in picked if v is not None),
        prompt_tokens=None if prompt_tokens is None else int(prompt_tokens),
    )


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and "message" in error:
            return str(error["message"])
        if "message" in body:
            return str(body["message"])
    return response.text[:500]


def _json_object_env(name: str) -> dict[str, Any]:
    """The JSON object in environment variable ``name``; unset or blank means ``{}``."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {raw!r}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must hold a JSON object, got {raw!r}")
    return value


class VLLMHTTPBackend:
    """``Backend`` over a running ``vllm serve`` OpenAI-compatible server.

    ``capabilities`` defaults to :data:`DEFAULT_CAPABILITIES`; pass (or assign) the result of
    ``jevemu.backends.probe.probe_capabilities`` for measured values. ``image``,
    ``model_revision`` and ``server_chat_template_kwargs`` (defaults: ``JEVEMU_VLLM_IMAGE`` /
    ``JEVEMU_VLLM_MODEL_REVISION`` / ``JEVEMU_VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS``) describe the
    server launch and are reported by :meth:`health`; vLLM exposes none of them over HTTP (the
    commit SHA of a model served from a Hugging Face snapshot path wins over
    ``model_revision``). ``chat_template_kwargs`` is sent with every ``/tokenize`` chat render
    and overrides the server's defaults key by key.

    One ``httpx.AsyncClient`` and one semaphore of ``max_concurrency`` exist per event loop, so
    the backend survives repeated ``asyncio.run`` calls. Use ``async with`` or :meth:`aclose`.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        max_concurrency: int = 32,
        timeout: float | httpx.Timeout = 60.0,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        capabilities: Capabilities | None = None,
        image: str | None = None,
        model_revision: str | None = None,
        server_chat_template_kwargs: Mapping[str, Any] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.model = model
        self.capabilities = capabilities or DEFAULT_CAPABILITIES
        self.chat_template_kwargs = dict(chat_template_kwargs) if chat_template_kwargs else None
        self.image = image if image is not None else os.environ.get(IMAGE_ENV)
        self.model_revision = (
            model_revision if model_revision is not None else os.environ.get(MODEL_REVISION_ENV)
        )
        self.server_chat_template_kwargs = (
            dict(server_chat_template_kwargs)
            if server_chat_template_kwargs is not None
            else _json_object_env(DEFAULT_CHAT_TEMPLATE_KWARGS_ENV)
        )
        self._api_key = api_key
        self._max_concurrency = max_concurrency
        self._timeout = timeout
        self._transport = transport
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: httpx.AsyncClient | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._whitespace_ids: dict[str, list[int]] = {}
        """Token ids of trailing prefill whitespace (``" "``), cached: the tokenizer is fixed."""
        self._token_text: dict[int, str] = {}
        """Text of each token id seen in a first-token response, cached likewise."""

    @property
    def effective_chat_template_kwargs(self) -> dict[str, Any]:
        """Template kwargs every render sees: server defaults, then per-request overrides."""
        return {**self.server_chat_template_kwargs, **(self.chat_template_kwargs or {})}

    async def __aenter__(self) -> VLLMHTTPBackend:
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

    def _session(self) -> tuple[httpx.AsyncClient, asyncio.Semaphore]:
        loop = asyncio.get_running_loop()
        if self._client is None or self._semaphore is None or self._loop is not loop:
            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
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

    async def request_json(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> Any:
        """Send one request and return the decoded JSON body.

        Connection failures and 5xx responses are retried a few times with short sleeps;
        anything else raises :class:`VLLMHTTPError` / :class:`VLLMError` immediately.
        """
        return (await self._send(method, path, body)).json()

    async def _send(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> httpx.Response:
        client, semaphore = self._session()
        for delay in (*_RETRY_DELAYS, None):
            try:
                async with semaphore:
                    response = await client.request(method, path, json=body)
            except _RETRYABLE_ERRORS as exc:
                if delay is None:
                    raise VLLMError(f"{method} {self.base_url}{path} failed: {exc!r}") from exc
                await asyncio.sleep(delay)
                continue
            if response.status_code >= 500 and delay is not None:
                await asyncio.sleep(delay)
                continue
            if response.is_error:
                raise VLLMHTTPError(response.status_code, path, _error_detail(response))
            return response
        raise AssertionError("unreachable")  # pragma: no cover

    async def cache_config(self) -> dict[str, str]:
        """Labels of vLLM's ``vllm:cache_config_info`` metric (e.g. ``enable_prefix_caching``,
        ``block_size``); empty if ``/metrics`` is unavailable."""
        try:
            text = (await self._send("GET", METRICS_PATH)).text
        except VLLMHTTPError:
            return {}
        for line in text.splitlines():
            if line.startswith("vllm:cache_config_info{"):
                return dict(_LABEL_RE.findall(line))
        return {}

    async def next_token_logprobs(
        self, prompt: RenderedPrompt, *, allowed: Sequence[str] | None, top_k: int
    ) -> NextTokenDist:
        """Greedy first token after the prefill, scored from the exact ids of
        :meth:`tokenize_chat` on ``/v1/completions``."""
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        top_logprobs = min(top_k, self.capabilities.top_logprobs_max)
        ids = await self.tokenize_chat(prompt)
        body = build_first_token_request(
            self.model, ids, allowed=allowed, top_logprobs=top_logprobs
        )
        payload = await self.request_json("POST", COMPLETIONS_PATH, body)
        return parse_first_token_response(
            payload,
            constrained=allowed is not None,
            top_logprobs=top_logprobs,
            texts=await self.token_texts(first_token_ids(payload)),
        )

    async def token_texts(self, ids: Sequence[int]) -> dict[int, str]:
        """Text of each token id, each decoded alone by ``/detokenize`` once and cached."""
        unknown = list(dict.fromkeys(i for i in ids if i not in self._token_text))
        texts = await asyncio.gather(*(self.detokenize([i]) for i in unknown))
        self._token_text.update(zip(unknown, texts, strict=True))
        return {i: self._token_text[i] for i in ids}

    async def sequence_logprobs(
        self, prefix: RenderedPrompt, continuations: Sequence[str]
    ) -> list[SeqScore]:
        if not self.capabilities.echo_prompt_logprobs:
            raise CapabilityError("this vLLM server does not return echo prompt logprobs")
        prefill = _assistant_text(prefix)
        head, prefill_ids = await asyncio.gather(
            self._generation_prompt_ids(prefix), self._text_ids(prefill)
        )
        return list(
            await asyncio.gather(
                *(self._score_continuation(head, prefill, prefill_ids, c) for c in continuations)
            )
        )

    async def _score_continuation(
        self, head: Sequence[int], prefill: str, prefill_ids: Sequence[int], continuation: str
    ) -> SeqScore:
        ids = await self._continuation_ids(prefill, prefill_ids, continuation)
        if not ids:
            # Nothing to score and nothing sent.
            return SeqScore(
                continuation=continuation, tokens=(), token_logprobs=(), prompt_tokens=0
            )
        start = len(head) + len(prefill_ids)
        full = [*head, *prefill_ids, *ids]
        payload = await self.request_json(
            "POST", COMPLETIONS_PATH, build_echo_request(self.model, full)
        )
        return parse_echo_response(payload, continuation=continuation, start=start, stop=len(full))

    async def _continuation_ids(
        self, prefill: str, prefill_ids: Sequence[int], continuation: str
    ) -> list[int]:
        """Tokens of ``continuation`` as they follow the exact prefill tokens.

        The joint tokenization (prefill + continuation) is used when it extends
        ``prefill_ids``; if the continuation merges with the prefill's last token, the prefill is
        kept exact and the continuation's standalone tokens are appended, as generation would
        produce.
        """
        if not continuation:
            return []
        joint = await self._text_ids(prefill + continuation)
        n = len(prefill_ids)
        if len(joint) > n and joint[:n] == list(prefill_ids):
            return joint[n:]
        return await self.tokenize(continuation)

    async def tokenize_chat(self, prompt: RenderedPrompt, *, continuation: str = "") -> list[int]:
        """Token ids of ``prompt`` (prefill + ``continuation``) as the model reads them.

        The conversation before the prefill is rendered by the server's chat template with its
        generation prompt, as the model's own reply would start; prefill + ``continuation``
        follow as plain text (:meth:`_text_ids`). The template never renders the prefill as a
        continued assistant message, which drops what it emits only on the generation prompt
        (Gemma 4's empty thought channel) and may trim trailing whitespace (Qwen3.5/3.8).
        """
        head, text = await asyncio.gather(
            self._generation_prompt_ids(prompt),
            self._text_ids(_assistant_text(prompt, continuation)),
        )
        return [*head, *text]

    async def _generation_prompt_ids(self, prompt: RenderedPrompt) -> list[int]:
        body = build_chat_tokenize_request(
            self.model, prompt, chat_template_kwargs=self.chat_template_kwargs
        )
        payload = await self.request_json("POST", TOKENIZE_PATH, body)
        return [int(t) for t in payload["tokens"]]

    async def _text_ids(self, text: str) -> list[int]:
        """Token ids of assistant ``text``: the text without its trailing whitespace, then that
        whitespace's own tokens, so a label read after ``Answer: `` follows exactly the gap
        tokens ``jevemu.scoring.first_token.reading_plan`` measured."""
        kept, tail = _split_trailing_whitespace(text)
        ids = await self.tokenize(kept) if kept else []
        if tail:
            if tail not in self._whitespace_ids:
                self._whitespace_ids[tail] = await self.tokenize(tail)
            ids += self._whitespace_ids[tail]
        return ids

    async def tokenize(self, text: str) -> list[int]:
        payload = await self.request_json(
            "POST", TOKENIZE_PATH, build_tokenize_request(self.model, text)
        )
        return [int(t) for t in payload["tokens"]]

    async def detokenize(self, ids: Sequence[int]) -> str:
        payload = await self.request_json(
            "POST", DETOKENIZE_PATH, build_detokenize_request(self.model, ids)
        )
        return str(payload["prompt"])

    async def health(self) -> BackendInfo:
        version, models, cache = await asyncio.gather(
            self.request_json("GET", VERSION_PATH),
            self.request_json("GET", MODELS_PATH),
            self.cache_config(),
        )
        entries = models.get("data") or []
        entry = next((m for m in entries if m.get("id") == self.model), None)
        if entry is None:
            served = [m.get("id") for m in entries]
            raise VLLMError(f"model {self.model!r} is not served at {self.base_url}; got {served}")
        caps = self.capabilities
        # Declared by the capabilities (probed or default) unless the server reports them.
        flags = {
            "max-logprobs": str(caps.top_logprobs_max),
            "logprobs-mode": caps.logprobs_mode,
            "enable-prefix-caching": cache.get(
                "enable_prefix_caching", str(caps.prefix_caching)
            ).lower(),
        }
        if "block_size" in cache:
            flags["block-size"] = cache["block_size"]
        if "cache_dtype" in cache:
            flags["kv-cache-dtype"] = cache["cache_dtype"]
        if entry.get("max_model_len") is not None:
            flags["max-model-len"] = str(entry["max_model_len"])
        if self.image:
            flags["image"] = self.image
        if template_kwargs := self.effective_chat_template_kwargs:
            flags["chat-template-kwargs"] = json.dumps(
                template_kwargs, sort_keys=True, separators=(",", ":")
            )
        match = _SNAPSHOT_RE.search(str(entry.get("root") or ""))
        return BackendInfo(
            backend=BACKEND_NAME,
            model=str(entry["id"]),
            model_revision=match.group(1) if match else self.model_revision,
            engine_version=str(version["version"]),
            flags=flags,
        )
