"""A stand-in for the official TypeSafe Python SDK (``typesafe-sdk``), backed by the emulator.

Code written against the SDK switches to the emulator by changing one import::

    from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score          # Jev
    from jevemu.compat.typesafe import AsyncTypeSafeClient, Choice, Noul, Score  # jevemu

Mirrored from the SDK reference (docs.typesafe.ai/sdk/python, September 2026):

- Questions: ``Noul``, ``Choice`` and ``Score`` (keyword arguments; ``instructions`` optional),
  or question dictionaries with a ``type`` key, mixed freely.
- Clients: ``AsyncTypeSafeClient`` and ``TypeSafeClient`` with
  ``system_one(state, questions, *, model=None, timeout=None, response_model=None)``,
  ``models.list()``, context managers and ``aclose()`` / ``close()``.
- Responses: ``SystemOneResponse`` (``model``, ``usage``, ``answers``, ``nouls``, ``choices``,
  ``scores``, ``request_id``), ``NoulAnswer``, ``ChoiceAnswer``, ``ScoreAnswer`` (integer
  ``legend``/``probabilities`` keys, as in the SDK), ``Usage``, ``ListModelsResponse`` and
  ``ModelMetadata``. With ``response_model``, a ``SystemOneResponse`` subclass also gets its
  declared fields from the answers by question name; any other pydantic model validates the
  response body.
- Errors: ``TypeSafeError`` for invalid questions or state, ``TypeSafeAPITimeoutError`` (a
  ``TypeSafeAPIConnectionError``) when ``timeout`` expires.

Differences, because there is no HTTP API behind the client:

- Which emulator answers: ``emulator=`` if given; else, when ``base_url`` or ``backend_model``
  is given, a client-owned ``VLLMHTTPBackend`` at ``base_url`` (default ``JEVEMU_VLLM_URL``, then
  ``http://localhost:8000``) serving ``backend_model`` (default ``JEVEMU_VLLM_MODEL``, then
  ``Qwen/Qwen3-0.6B``), with ``api_key`` as the vLLM server's key; else the emulator set by
  :func:`set_default_emulator`; else a client-owned backend from those environment defaults.
  ``TYPESAFE_API_KEY`` and ``TYPESAFE_BASE_URL`` are never read. Client-owned emulators round
  like Jev (``round_to=0.01``); ``close()`` closes only a client-owned backend.
- ``model`` is sent as the request's ``model``, which the emulator ignores; responses report
  the emulator's model id (``"jevemu/<backend model>"``), as does ``models.list()``.
- ``request_id`` is a local id (``"jevemu-<32 hex>"``). Not provided: ``retry`` /
  ``RetryPolicy``, ``headers``, ``transport``, ``http_client``, ``extra_headers``,
  ``extra_body``, ``raw_http_response`` and the HTTP status error classes. Emulator and backend
  errors (e.g. :class:`~jevemu.emulator.QuestionError`) propagate unchanged.
- ``TypeSafeClient`` runs the async client on its own event loop, so it cannot be used from
  inside a running event loop.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Coroutine, Mapping, Sequence
from datetime import datetime, timezone
from functools import cached_property
from types import TracebackType
from typing import Annotated, Any, Literal, TypeVar, overload

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from jevemu.backends.base import BackendInfo
from jevemu.backends.vllm_http import VLLMHTTPBackend
from jevemu.emulator import Emulator, response_model_id
from jevemu.types import (
    DEFAULT_MODEL,
    ChoiceQuestion,
    Description,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)
from jevemu.types import SystemOneResponse as WireResponse

__all__ = [
    "Answer",
    "AsyncModels",
    "AsyncTypeSafeClient",
    "Choice",
    "ChoiceAnswer",
    "JSONContent",
    "ListModelsResponse",
    "ModelMetadata",
    "Models",
    "Noul",
    "NoulAnswer",
    "Question",
    "Questions",
    "Score",
    "ScoreAnswer",
    "SystemOneResponse",
    "TypeSafeAPIConnectionError",
    "TypeSafeAPITimeoutError",
    "TypeSafeClient",
    "TypeSafeError",
    "Usage",
    "set_default_emulator",
]

URL_ENV = "JEVEMU_VLLM_URL"
MODEL_ENV = "JEVEMU_VLLM_MODEL"
DEFAULT_URL = "http://localhost:8000"
DEFAULT_BACKEND_MODEL = "Qwen/Qwen3-0.6B"
JEV_PRECISION = 0.01
"""Jev reports probabilities, confidence and scores to two decimals."""

JSONContent = str | Mapping[str, Any] | Sequence[Any]
"""Text, a JSON object or an array (SDK ``typesafe_sdk.JSONContent``)."""

ResponseT = TypeVar("ResponseT", bound=BaseModel)
_T = TypeVar("_T")

_default_emulator: Emulator | None = None


def set_default_emulator(emulator: Emulator | None) -> None:
    """Emulator for clients constructed without ``emulator``, ``base_url`` or
    ``backend_model``, so SDK code that calls ``TypeSafeClient()`` runs unchanged against it.
    The caller keeps ownership (clients never close it); ``None`` restores the default."""
    global _default_emulator
    _default_emulator = emulator


# --- errors ------------------------------------------------------------------------------------


class TypeSafeError(Exception):
    """Base exception (SDK ``typesafe_sdk.TypeSafeError``); raised for invalid questions."""


class TypeSafeAPIConnectionError(TypeSafeError, ConnectionError):
    """A request produced no answer (SDK ``typesafe_sdk.TypeSafeAPIConnectionError``)."""


class TypeSafeAPITimeoutError(TypeSafeAPIConnectionError, TimeoutError):
    """A request exceeded its timeout, in seconds (``timeout``)."""

    def __init__(self, message: str, timeout: float) -> None:
        super().__init__(message)
        self.timeout = timeout


# --- questions ---------------------------------------------------------------------------------


class Noul(NoulQuestion):
    """A yes/no question with optional descriptions of either outcome."""

    instructions: Description = None


class Choice(ChoiceQuestion):
    """A question that selects between named alternatives."""

    instructions: Description = None


class Score(ScoreQuestion):
    """A question that assigns a score using an ordered rubric (level ``i`` scores ``i``)."""

    instructions: Description = None


Question = Noul | Choice | Score | Mapping[str, Any]
"""A question object or a question dictionary with a ``type`` key."""

Questions = Mapping[str, Question]


# --- responses ---------------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class NoulAnswer(_Model):
    """A yes/no answer: ``noul`` is the probability of yes."""

    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(_Model):
    """The selected option and each option's probability."""

    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(_Model):
    """The expected score, its rubric and each level's probability, keyed by integer level."""

    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[int, str | dict[str, Any] | list[Any] | None]
    probabilities: dict[int, float]


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(_Model):
    """Token counts for a request."""

    input_tokens: int | None = None
    output_tokens: int | None = None


class SystemOneResponse(_Model):
    """Answers keyed by question name, with model and usage metadata."""

    model: str
    usage: Usage
    answers: dict[str, Answer] = Field(default_factory=dict)
    _request_id: str = PrivateAttr(default="")

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {name: a for name, a in self.answers.items() if isinstance(a, NoulAnswer)}

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {name: a for name, a in self.answers.items() if isinstance(a, ChoiceAnswer)}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {name: a for name, a in self.answers.items() if isinstance(a, ScoreAnswer)}


class ModelMetadata(_Model):
    """One available model."""

    name: str
    description: str
    release_date: str


class ListModelsResponse(_Model):
    """The available models: the emulator's one model."""

    models: tuple[ModelMetadata, ...]
    _request_id: str = PrivateAttr(default="")

    @property
    def request_id(self) -> str:
        return self._request_id


# --- clients -----------------------------------------------------------------------------------


class AsyncModels:
    """The Models resource, reached through ``AsyncTypeSafeClient.models``."""

    def __init__(self, client: AsyncTypeSafeClient) -> None:
        self._client = client

    async def list(self) -> ListModelsResponse:
        """The emulator's model; its ``release_date`` is today's UTC date (the emulated model is
        assembled at request time)."""
        return _models_response(await self._client.emulator.warm_up())


class AsyncTypeSafeClient:
    """Asynchronous client (SDK ``typesafe_sdk.AsyncTypeSafeClient``) answered by an emulator.

    See the module docstring for which emulator answers and what differs from the SDK.
    ``timeout`` (seconds) bounds each ``system_one`` call; ``None`` waits indefinitely.
    """

    def __init__(
        self,
        *,
        emulator: Emulator | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        base_url: str | None = None,
        backend_model: str | None = None,
    ) -> None:
        own_backend = base_url is not None or backend_model is not None
        if emulator is not None and (own_backend or api_key is not None):
            raise ValueError("pass either emulator or base_url/backend_model/api_key, not both")
        self._backend: VLLMHTTPBackend | None = None
        if emulator is None and (own_backend or _default_emulator is None):
            self._backend = VLLMHTTPBackend(
                base_url or _env(URL_ENV) or DEFAULT_URL,
                backend_model or _env(MODEL_ENV) or DEFAULT_BACKEND_MODEL,
                api_key=api_key,
            )
            emulator = Emulator(self._backend, round_to=JEV_PRECISION)
        chosen = emulator or _default_emulator
        if chosen is None:  # pragma: no cover - one of the branches above always sets it
            raise AssertionError("no emulator")
        self.emulator: Emulator = chosen
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout

    @cached_property
    def models(self) -> AsyncModels:
        return AsyncModels(self)

    @overload
    async def system_one(
        self,
        state: JSONContent,
        questions: Questions,
        *,
        model: str | None = None,
        timeout: float | None = None,
        response_model: None = None,
    ) -> SystemOneResponse: ...

    @overload
    async def system_one(
        self,
        state: JSONContent,
        questions: Questions,
        *,
        model: str | None = None,
        timeout: float | None = None,
        response_model: type[ResponseT],
    ) -> ResponseT: ...

    async def system_one(
        self,
        state: JSONContent,
        questions: Questions,
        *,
        model: str | None = None,
        timeout: float | None = None,
        response_model: type[ResponseT] | None = None,
    ) -> SystemOneResponse | ResponseT:
        """Answer named questions about text or structured state.

        Raises :class:`TypeSafeError` when the questions or state are invalid and
        :class:`TypeSafeAPITimeoutError` when ``timeout`` (else the client's) expires.
        """
        request = _request(state, questions, model or self.model)
        limit = timeout if timeout is not None else self.timeout
        if limit is None:
            response = await self.emulator.system_one(request)
        else:
            try:
                response = await asyncio.wait_for(self.emulator.system_one(request), limit)
            except asyncio.TimeoutError as exc:
                raise TypeSafeAPITimeoutError(f"no answer within {limit} s", limit) from exc
        return _response(response, response_model)

    async def aclose(self) -> None:
        """Close the client-owned backend (a supplied or default emulator is left open)."""
        if self._backend is not None:
            await self._backend.aclose()

    async def __aenter__(self) -> AsyncTypeSafeClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class Models:
    """The Models resource, reached through ``TypeSafeClient.models``."""

    def __init__(self, client: TypeSafeClient) -> None:
        self._client = client

    def list(self) -> ListModelsResponse:
        """See :meth:`AsyncModels.list`."""
        return self._client._run(self._client._async.models.list())


class TypeSafeClient:
    """Synchronous client (SDK ``typesafe_sdk.TypeSafeClient``): :class:`AsyncTypeSafeClient`
    run on a private event loop. Same arguments."""

    def __init__(
        self,
        *,
        emulator: Emulator | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        base_url: str | None = None,
        backend_model: str | None = None,
    ) -> None:
        self._async = AsyncTypeSafeClient(
            emulator=emulator,
            api_key=api_key,
            model=model,
            timeout=timeout,
            base_url=base_url,
            backend_model=backend_model,
        )
        self._loop = asyncio.new_event_loop()

    @property
    def emulator(self) -> Emulator:
        return self._async.emulator

    @cached_property
    def models(self) -> Models:
        return Models(self)

    @overload
    def system_one(
        self,
        state: JSONContent,
        questions: Questions,
        *,
        model: str | None = None,
        timeout: float | None = None,
        response_model: None = None,
    ) -> SystemOneResponse: ...

    @overload
    def system_one(
        self,
        state: JSONContent,
        questions: Questions,
        *,
        model: str | None = None,
        timeout: float | None = None,
        response_model: type[ResponseT],
    ) -> ResponseT: ...

    def system_one(
        self,
        state: JSONContent,
        questions: Questions,
        *,
        model: str | None = None,
        timeout: float | None = None,
        response_model: type[ResponseT] | None = None,
    ) -> SystemOneResponse | ResponseT:
        """See :meth:`AsyncTypeSafeClient.system_one`."""
        if response_model is None:
            return self._run(self._async.system_one(state, questions, model=model, timeout=timeout))
        return self._run(
            self._async.system_one(
                state, questions, model=model, timeout=timeout, response_model=response_model
            )
        )

    def close(self) -> None:
        """Close the client-owned backend and the private event loop."""
        if not self._loop.is_closed():
            self._loop.run_until_complete(self._async.aclose())
            self._loop.close()

    def __enter__(self) -> TypeSafeClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        # SDK code often never closes its client (``TypeSafeClient().models.list()``); release
        # the private event loop when the client is collected.
        loop = getattr(self, "_loop", None)
        if loop is not None and not loop.is_closed() and not loop.is_running():
            loop.close()

    def _run(self, job: Coroutine[Any, Any, _T]) -> _T:
        if self._loop.is_closed():
            job.close()
            raise TypeSafeError("the client is closed")
        return self._loop.run_until_complete(job)


# --- conversions -------------------------------------------------------------------------------


def _env(name: str) -> str | None:
    """An environment value; empty or whitespace-only values count as unset (as in the SDK)."""
    return os.environ.get(name, "").strip() or None


def _request(state: JSONContent, questions: Questions, model: str) -> SystemOneRequest:
    if not questions:
        raise TypeSafeError("questions must not be empty")
    payload = {
        "state": state,
        "model": model,
        "questions": {
            name: q.model_dump() if isinstance(q, BaseModel) else dict(q)
            for name, q in questions.items()
        },
    }
    try:
        return SystemOneRequest.model_validate(payload)
    except ValidationError as exc:
        raise TypeSafeError(f"invalid System One request: {exc}") from exc


def _response(
    response: WireResponse, response_model: type[ResponseT] | None
) -> SystemOneResponse | ResponseT:
    """The emulator's wire response as the SDK type, or as ``response_model``."""
    body = response.model_dump(mode="json", exclude={"x_jevemu"})
    if response_model is None:
        result = SystemOneResponse.model_validate(body)
        result._request_id = _request_id()
        return result
    if issubclass(response_model, SystemOneResponse):
        # Declared answer fields are filled by question name; response fields win on clashes.
        typed = response_model.model_validate({**body["answers"], **body})
        typed._request_id = _request_id()
        return typed
    return response_model.model_validate(body)


def _request_id() -> str:
    return f"jevemu-{uuid.uuid4().hex}"


def _models_response(info: BackendInfo) -> ListModelsResponse:
    revision = info.model_revision or "unknown revision"
    result = ListModelsResponse(
        models=(
            ModelMetadata(
                name=response_model_id(info),
                description=(
                    f"jevemu emulation of Jev System One by {info.model} ({revision}) on "
                    f"{info.backend} {info.engine_version}"
                ),
                release_date=datetime.now(timezone.utc).date().isoformat(),
            ),
        )
    )
    result._request_id = _request_id()
    return result
