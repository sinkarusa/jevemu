"""Typed exceptions raised by jevemu's API clients (``jevemu.jev_client`` and the Chat
Completions backends in ``jevemu.backends.chat_api``: OpenAI, OpenRouter).

HTTP errors map one class per status family: 401 auth, 400/422 validation, 429 rate limit,
529 overload; anything else is a plain :class:`JevHTTPError` (chat APIs: :class:`ChatAPIHTTPError`).
Messages and attributes never contain the API key: bodies are scrubbed by the client before an
exception is built. :class:`BudgetExceeded` is the common base of every client's budget guard.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BudgetExceeded",
    "ChatAPIAuthError",
    "ChatAPIBudgetExceeded",
    "ChatAPIConnectionError",
    "ChatAPIEmptyReply",
    "ChatAPIError",
    "ChatAPIHTTPError",
    "ChatAPIKeyMissing",
    "ChatAPIQuotaExceeded",
    "ChatAPIResponseError",
    "ChatAPIStuckReply",
    "JevAPIKeyMissing",
    "JevAuthError",
    "JevBudgetExceeded",
    "JevConnectionError",
    "JevError",
    "JevHTTPError",
    "JevOverloaded",
    "JevRateLimited",
    "JevResponseError",
    "JevRetryableHTTPError",
    "JevValidationError",
    "JevVersionDrift",
    "JevemuError",
]

_BODY_PREVIEW = 500


class JevemuError(Exception):
    """Base class for jevemu errors."""


class BudgetExceeded(JevemuError):
    """Sending the request could push spend past ``max_usd``; nothing was sent.

    ``estimated_usd`` is the pessimistic cost estimate of the refused request and
    ``committed_usd`` the spend already made plus estimates of requests still in flight.
    """

    service = "API"

    def __init__(self, *, max_usd: float, committed_usd: float, estimated_usd: float) -> None:
        self.max_usd = max_usd
        self.committed_usd = committed_usd
        self.estimated_usd = estimated_usd
        super().__init__(
            f"{self.service} budget ${max_usd:.6f} would be exceeded: ${committed_usd:.6f} "
            f"committed + ${estimated_usd:.6f} estimated for this request"
        )


class JevError(JevemuError):
    """Base class for errors talking to the real Jev API."""


class JevAPIKeyMissing(JevError):
    """A network call was needed but no API key was given or found in ``TYPESAFE_API_KEY``."""


class JevConnectionError(JevError):
    """The request never produced an HTTP response (DNS, connect, read timeout, ...)."""


class JevHTTPError(JevError):
    """Jev answered with an HTTP error status.

    ``body`` is the decoded JSON error body when it parses, else the (truncated) text.
    ``attempts`` counts HTTP attempts made for this request, including the failing one.
    ``request_id`` is Jev's ``x-typesafe-request-id`` header, when sent.
    """

    def __init__(
        self, status_code: int, body: Any, *, attempts: int = 1, request_id: str | None = None
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.attempts = attempts
        self.request_id = request_id
        where = f" (request {request_id})" if request_id else ""
        super().__init__(
            f"Jev HTTP {status_code}{where} after {attempts} attempt(s): {_preview(body)}"
        )


class JevAuthError(JevHTTPError):
    """401: missing or invalid API key."""


class JevValidationError(JevHTTPError):
    """The request was rejected as invalid; ``body["detail"]`` says why.

    Documented as 422 (schema errors: ``detail`` is a list of ``{type, loc, msg, input}``);
    Jev 1.13.0 also answers 400 with ``detail = {error_type: "api_usage_error", message}`` for
    semantic errors such as an unknown model or question type.
    """


class JevRetryableHTTPError(JevHTTPError):
    """A status Jev documents as retryable; raised only once retries are exhausted.

    ``retry_after`` is the last ``retry-after`` delay the server sent, in seconds.
    """

    def __init__(
        self,
        status_code: int,
        body: Any,
        *,
        attempts: int = 1,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(status_code, body, attempts=attempts, request_id=request_id)
        self.retry_after = retry_after


class JevRateLimited(JevRetryableHTTPError):
    """429: rate limit exceeded on every attempt."""


class JevOverloaded(JevRetryableHTTPError):
    """529: Jev overloaded on every attempt."""


class JevResponseError(JevError):
    """A 2xx response whose body is not a valid ``SystemOneResponse`` (or models list)."""

    def __init__(self, message: str, body: Any) -> None:
        self.body = body
        super().__init__(f"{message}: {_preview(body)}")


class JevVersionDrift(JevError):
    """Jev (or another server driven by ``JevClient``, named by ``service``) answered with a
    model version other than the pinned one."""

    def __init__(self, expected: str, actual: str, *, service: str = "Jev") -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"{service} answered with model {actual!r}, pinned {expected!r}")


class JevBudgetExceeded(JevError, BudgetExceeded):
    """The Jev client's budget guard refused a request (see :class:`BudgetExceeded`)."""

    service = "Jev"


class ChatAPIError(JevemuError):
    """Base class for errors talking to a Chat Completions API (OpenAI, OpenRouter)."""


class ChatAPIKeyMissing(ChatAPIError):
    """A network call was needed but no API key was given or found in the backend's variable
    (``OPENAI_API_KEY``, ``OPENROUTER_API_KEY``)."""


class ChatAPIConnectionError(ChatAPIError):
    """The request never produced an HTTP response, on every attempt."""


class ChatAPIHTTPError(ChatAPIError):
    """The API answered with an HTTP error status (after retries, for retryable ones).

    ``body`` is the decoded JSON error body when it parses, else the (truncated) text.
    ``request_id`` is the API's request-id header, when sent. ``service`` names the API.
    """

    def __init__(
        self,
        status_code: int,
        body: Any,
        *,
        attempts: int = 1,
        request_id: str | None = None,
        service: str = "chat API",
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.attempts = attempts
        self.request_id = request_id
        self.service = service
        where = f" (request {request_id})" if request_id else ""
        super().__init__(
            f"{service} HTTP {status_code}{where} after {attempts} attempt(s): {_preview(body)}"
        )


class ChatAPIAuthError(ChatAPIHTTPError):
    """401/403: missing, invalid or unauthorized API key."""


class ChatAPIQuotaExceeded(ChatAPIHTTPError):
    """The account is out of credit (OpenAI: 429 ``insufficient_quota``; OpenRouter: 402);
    retrying cannot help."""


class ChatAPIResponseError(ChatAPIError):
    """A 2xx response the backend cannot use (no logprobs, malformed body, wrong model).

    ``cost_usd`` and ``latency_ms`` are the billed cost and HTTP latency of the call that
    returned it, when the backend knows them (0 otherwise).
    """

    def __init__(self, message: str, body: Any) -> None:
        self.body = body
        self.cost_usd = 0.0
        self.latency_ms = 0.0
        super().__init__(f"{message}: {_preview(body)}")


class ChatAPIEmptyReply(ChatAPIResponseError):
    """A reply with no answer to read (billed anyway): the model ended its turn at once (no
    token logprobs), or a structured reply never reached its answer value. Sending it again can
    help."""


class ChatAPIStuckReply(ChatAPIEmptyReply):
    """A structured reply that never reached its answer value (it ran out of output tokens in
    the whitespace or JSON before it)."""


class ChatAPIBudgetExceeded(ChatAPIError, BudgetExceeded):
    """A chat backend's budget guard refused a request (see :class:`BudgetExceeded`)."""

    def __init__(
        self, *, service: str, max_usd: float, committed_usd: float, estimated_usd: float
    ) -> None:
        self.service = service
        super().__init__(max_usd=max_usd, committed_usd=committed_usd, estimated_usd=estimated_usd)


def _preview(body: Any) -> str:
    text = body if isinstance(body, str) else repr(body)
    return text if len(text) <= _BODY_PREVIEW else text[:_BODY_PREVIEW] + "..."
