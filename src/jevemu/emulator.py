"""The emulator facade: Jev System One answers from a vLLM-served model.

:meth:`Emulator.system_one` answers a :class:`~jevemu.types.SystemOneRequest` the way Jev does:
every question sees the same state and is scored on its own (concurrently, at most
``max_concurrency`` questions in flight per emulator); one answer never becomes context for
another. Per question:

1. ``strategy.score`` (default ``auto``) turns (state, question) into log-probabilities over the
   question's keys (:class:`~jevemu.scoring.ScoreResult`, keys in question order).
2. With a ``debiaser`` (:mod:`jevemu.debias`), those log-probabilities are debiased; permuting
   and contextual debiasers score further variants of the question with the same strategy.
3. With a ``calibrators`` registry, the calibrator resolved for (backend, model,
   ``renderer.template_id``, question signature) maps that row of log-probabilities to
   probabilities. Without a registry the (debiased) probabilities are used as they are.
4. The probabilities become the Jev answer:

   - Choice: ``probabilities`` keyed by option key in question order; ``choice`` is the argmax
     (a tie goes to the option listed first).
   - Score: ``probabilities`` keyed ``"0"``..``"K-1"``; ``score = sum(i * p_i)``;
     ``legend = {str(i): criteria[i]}``.
   - Noul: ``noul = P(yes)``.
   - Choice and Score: ``confidence = confidence_fn(probabilities in key order, ordinal=...)``
     with ``ordinal`` true for Score (its levels are ordered).

**Rounding.** ``round_to`` (Jev reports to 0.01) rounds every reported number independently to
the nearest multiple of ``round_to`` (Python ``round``: ties to even), clamped to its valid
range: probabilities, confidence, ``noul`` and ``score``. Confidence and score are computed from
the unrounded probabilities, and ``choice`` is the argmax of the unrounded distribution, so the
rounded probabilities need not sum to exactly 1 (Jev's do not either) and may tie.

**Response fields.** ``model`` is ``"jevemu/<backend model>"`` (e.g. ``"jevemu/Qwen/Qwen3-0.6B"``):
it validates as a ``SystemOneResponse`` for both systems but never passes for a Jev version. The
request's ``model`` is ignored; the backend's model answers. ``usage.input_tokens`` sums the
prompt tokens of every backend call, the debiaser's included (each echo request counts its full
prompt, since echo recomputes the prefix). ``usage.output_tokens`` counts generated tokens: one
per first-token call on a local server (``max_tokens=1``), none for echo, and the billed output
tokens a paid-API backend reports (``NextTokenDist.api``). A paid-API backend's calls also add
the ``usage`` extra fields ``cached_input_tokens`` and ``cache_write_tokens``.

**Diagnostics.** With ``include_diagnostics=True``, ``x_jevemu`` maps each question id to an
:class:`~jevemu.types.EmulatorDiagnostics`: backend identity, strategy and label scheme, observed
mass, missing labels, the strategy's probabilities before debiasing and calibration, the
debiaser (its ``id``, the backend calls it added, the prior it divided out and every permuted
presentation it scored), the calibrator as ``"<name>:<resolution source>"`` (``None`` without a
registry), backend calls (debiaser included), latency (scoring, debiasing and answer assembly,
excluding time queued behind ``max_concurrency``), cached tokens, the strategy's and
debiaser's warnings (including every ``auto`` fallback) and, for a paid-API backend, ``api``
(:class:`~jevemu.types.ApiUsage`: calls, response-cache hits, HTTP latency, spend, reported
models and ``system_fingerprint`` values).

**Warm-up.** :meth:`Emulator.warm_up` runs once, before the first request unless called
explicitly. A :class:`~jevemu.backends.vllm_http.VLLMHTTPBackend` still holding the adapter's
documented default capabilities gets the measured ones (the capability probe, cached on disk
per server identity); backends with capabilities already set are trusted. It then reads the
backend's :class:`~jevemu.backends.base.BackendInfo` and checks which label surfaces (letters,
digits, ``Yes``/``No``, bare and space-prefixed) are single tokens, which the first-token
strategies need; the result is cached for scoring, and labels with no single-token surface are
logged and listed in :attr:`Emulator.unreadable_labels`. A backend without a local tokenizer
(``Capabilities.local_tokenizer`` false) skips the check.

**Prefill.** A renderer given by name is built with the ``Answer:`` prefill only if the backend
can continue one (``Capabilities.assistant_prefill``); otherwise prompts end in the user turn.

**Errors.** Jev answers every question or none: the first failure cancels the other questions
and raises :class:`QuestionError` naming the failed question (the one listed first if several
failed together), with the original exception as ``__cause__``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections.abc import Coroutine, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

import numpy as np

from jevemu.backends.base import (
    ApiCall,
    Backend,
    BackendInfo,
    Capabilities,
    NextTokenDist,
    RenderedPrompt,
    SeqScore,
)
from jevemu.backends.probe import ServerFlags, probe_capabilities
from jevemu.backends.vllm_http import DEFAULT_CAPABILITIES as VLLM_DEFAULT_CAPABILITIES
from jevemu.backends.vllm_http import VLLMHTTPBackend
from jevemu.calibrate import CalibrationKey, CalibratorRegistry
from jevemu.confidence import DEFAULT_CONFIDENCE, ConfidenceFn
from jevemu.debias import Debiaser, DebiasOutcome
from jevemu.errors import JevemuError
from jevemu.render import (
    CHOICE_LABELS,
    LAYOUTS,
    NOUL_LABELS,
    SCORE_LABELS,
    SURFACE_PREFIXES,
    PromptRenderer,
    answer_keys,
)
from jevemu.scoring import AutoStrategy, ScoreResult, ScoringStrategy
from jevemu.scoring.first_token import single_token_surfaces
from jevemu.types import (
    CACHE_WRITE_FIELD,
    CACHED_INPUT_FIELD,
    ApiUsage,
    ChoiceAnswer,
    ChoiceQuestion,
    EmulatorDiagnostics,
    JSONish,
    NoulAnswer,
    PermutationScore,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)

__all__ = [
    "MODEL_PREFIX",
    "Emulator",
    "QuestionError",
    "response_model_id",
    "server_flags_from_env",
]

logger = logging.getLogger(__name__)

MODEL_PREFIX = "jevemu/"
"""``SystemOneResponse.model`` is this prefix plus the backend's model id."""

MAX_LOGPROBS_ENV = "JEVEMU_VLLM_MAX_LOGPROBS"
LOGPROBS_MODE_ENV = "JEVEMU_VLLM_LOGPROBS_MODE"
DEFAULT_MAX_LOGPROBS = 576
"""``--max-logprobs`` of ``docker/vllm/compose.yaml``: every code label of a 255-option question in
one top-k (:mod:`jevemu.scoring.single_call`)."""

_ALL_LABELS: tuple[str, ...] = (*CHOICE_LABELS, *SCORE_LABELS, *NOUL_LABELS)

_T = TypeVar("_T")
_Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


class QuestionError(JevemuError):
    """A question could not be answered, so the request fails (Jev is all-or-nothing)."""

    def __init__(self, question_id: str, cause: BaseException) -> None:
        self.question_id = question_id
        super().__init__(f"question {question_id!r} failed: {type(cause).__name__}: {cause}")


def response_model_id(info: BackendInfo) -> str:
    """The ``model`` the emulator reports for a backend: ``"jevemu/<backend model>"``."""
    return MODEL_PREFIX + info.model


def server_flags_from_env() -> ServerFlags:
    """Declared vLLM launch flags for the capability probe, as ``scripts/serve_vllm.sh`` sets
    them: ``JEVEMU_VLLM_MAX_LOGPROBS`` (default 576) and ``JEVEMU_VLLM_LOGPROBS_MODE`` (default
    ``raw_logprobs``). The probe measures the effective top-logprobs cap at or below the
    declared one; the logprobs mode is not observable over HTTP."""
    return ServerFlags.model_validate(
        {
            "max_logprobs": int(os.environ.get(MAX_LOGPROBS_ENV) or DEFAULT_MAX_LOGPROBS),
            "logprobs_mode": os.environ.get(LOGPROBS_MODE_ENV) or "raw_logprobs",
        }
    )


# --- per-question accounting -------------------------------------------------------------------


@dataclass
class _Meter:
    generated_tokens: int = 0
    api_calls: list[ApiCall] = field(default_factory=list)
    api_cached_tokens: int = 0
    """Prompt-cache hits of the paid-API calls (the provider's cache, not the response cache)."""

    def api_usage(self) -> ApiUsage | None:
        calls = self.api_calls
        if not calls:
            return None
        return ApiUsage(
            calls=len(calls),
            response_cache_hits=sum(c.response_cached for c in calls),
            latency_ms=math.fsum(c.latency_ms for c in calls),
            cost_usd=math.fsum(c.cost_usd for c in calls),
            models=sorted({c.model for c in calls}),
            system_fingerprints=sorted(
                {c.system_fingerprint for c in calls if c.system_fingerprint is not None}
            ),
            providers=sorted({c.provider for c in calls if c.provider is not None}),
        )

    def usage_extra(self) -> dict[str, int]:
        """``Usage`` extra fields of the paid-API calls (none for local servers)."""
        if not self.api_calls:
            return {}
        return {
            CACHED_INPUT_FIELD: self.api_cached_tokens,
            CACHE_WRITE_FIELD: sum(c.cache_write_tokens or 0 for c in self.api_calls),
        }


_METER: ContextVar[_Meter | None] = ContextVar("jevemu_emulator_meter", default=None)
"""The meter of the question whose task is running; each question task sets its own."""


class _MeteredBackend:
    """Forwards to ``inner`` and counts generated tokens for the current question's meter.

    A first-token call on a local server generates exactly one token (``max_tokens=1``); echo
    generates none. A paid-API call reports its own billed output tokens and accounting
    (:class:`~jevemu.backends.base.ApiCall`). Tasks copy their context, so the concurrent calls
    a strategy makes all count toward the question that started them. One instance per emulator
    keeps the scoring tokenizer cache (keyed by backend object) warm across requests.
    """

    def __init__(self, inner: Backend) -> None:
        self.inner = inner

    @property
    def capabilities(self) -> Capabilities:
        return self.inner.capabilities

    @capabilities.setter
    def capabilities(self, value: Capabilities) -> None:
        self.inner.capabilities = value

    async def next_token_logprobs(
        self, prompt: RenderedPrompt, *, allowed: Sequence[str] | None, top_k: int
    ) -> NextTokenDist:
        dist = await self.inner.next_token_logprobs(prompt, allowed=allowed, top_k=top_k)
        meter = _METER.get()
        if meter is not None:
            if dist.api is None:
                meter.generated_tokens += 1
            else:
                meter.generated_tokens += dist.api.completion_tokens
                meter.api_calls.append(dist.api)
                meter.api_cached_tokens += dist.cached_tokens or 0
        return dist

    async def sequence_logprobs(
        self, prefix: RenderedPrompt, continuations: Sequence[str]
    ) -> list[SeqScore]:
        return await self.inner.sequence_logprobs(prefix, continuations)

    async def tokenize(self, text: str) -> list[int]:
        return await self.inner.tokenize(text)

    async def detokenize(self, ids: Sequence[int]) -> str:
        return await self.inner.detokenize(ids)

    async def health(self) -> BackendInfo:
        return await self.inner.health()


@dataclass(frozen=True)
class _Outcome:
    answer: _Answer
    diagnostics: EmulatorDiagnostics | None
    prompt_tokens: int
    meter: _Meter


@dataclass(frozen=True)
class _LoopLocal:
    """asyncio primitives bind to one event loop; ``system_one_sync`` runs a new loop per call."""

    loop: asyncio.AbstractEventLoop
    semaphore: asyncio.Semaphore
    lock: asyncio.Lock


# --- the emulator -----------------------------------------------------------------------------


class Emulator:
    """``SystemOneClient`` answering Jev requests with ``backend`` (see the module docstring).

    ``renderer`` is a :class:`~jevemu.render.PromptRenderer`, a layout name
    (``"question_first"``, ``"state_first"``) or ``"default"`` (the renderer's default layout,
    ``state_first``). ``strategy`` is ``"auto"`` or any :class:`~jevemu.scoring.ScoringStrategy`;
    for large models :class:`~jevemu.scoring.NoEchoAutoStrategy` is the measured best
    (``docs/research/selection.md``). ``debiaser`` is any :class:`~jevemu.debias.Debiaser`
    (``None``: no debiasing). ``confidence_fn`` defaults to
    :data:`~jevemu.confidence.DEFAULT_CONFIDENCE`
    (``mode_distance``, the fit of Jev's confidence). The emulator does not own ``backend``:
    close it yourself.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        renderer: PromptRenderer | str = "default",
        strategy: ScoringStrategy | Literal["auto"] = "auto",
        debiaser: Debiaser | None = None,
        calibrators: CalibratorRegistry | None = None,
        confidence_fn: ConfidenceFn = DEFAULT_CONFIDENCE,
        max_concurrency: int = 32,
        include_diagnostics: bool = False,
        round_to: float | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        if round_to is not None and not (math.isfinite(round_to) and 0.0 < round_to <= 1.0):
            raise ValueError(f"round_to must lie in (0, 1], got {round_to}")
        self.backend = backend
        self.renderer = _resolve_renderer(renderer, prefill=backend.capabilities.assistant_prefill)
        self.strategy = _resolve_strategy(strategy)
        self.debiaser = debiaser
        self.calibrators = calibrators
        self.confidence_fn = confidence_fn
        self.max_concurrency = max_concurrency
        self.include_diagnostics = include_diagnostics
        self.round_to = round_to
        self._metered = _MeteredBackend(backend)
        self._info: BackendInfo | None = None
        self._unreadable: tuple[str, ...] = ()
        self._local: _LoopLocal | None = None

    @property
    def info(self) -> BackendInfo | None:
        """The backend's identity, once :meth:`warm_up` has run."""
        return self._info

    @property
    def unreadable_labels(self) -> tuple[str, ...]:
        """Labels (letters, digits, ``Yes``/``No``) with no single-token surface, found by
        :meth:`warm_up`; first-token strategies report them as missing."""
        return self._unreadable

    async def warm_up(self, *, flags: ServerFlags | None = None) -> BackendInfo:
        """Probe capabilities, read backend info and verify label tokenization (once).

        ``flags`` are the vLLM server's declared launch flags for the capability probe
        (default: :func:`server_flags_from_env`); they only matter for a
        ``VLLMHTTPBackend`` that has not been given capabilities. Later calls return the
        cached info.
        """
        async with self._loop_local().lock:
            if self._info is None:
                backend = self.backend
                if (
                    isinstance(backend, VLLMHTTPBackend)
                    and backend.capabilities is VLLM_DEFAULT_CAPABILITIES
                ):
                    backend.capabilities = await probe_capabilities(
                        backend, flags or server_flags_from_env()
                    )
                info = await self._metered.health()
                if backend.capabilities.local_tokenizer:
                    single = await single_token_surfaces(self._metered, _ALL_LABELS)
                    self._unreadable = tuple(
                        label
                        for label in _ALL_LABELS
                        if not any(single[prefix + label] for prefix in SURFACE_PREFIXES)
                    )
                if self._unreadable:
                    logger.warning(
                        "%s: labels %s have no single-token surface; first-token strategies "
                        "will report them missing",
                        info.model,
                        list(self._unreadable),
                    )
                self._info = info
            return self._info

    async def system_one(self, request: SystemOneRequest) -> SystemOneResponse:
        info = await self.warm_up()
        semaphore = self._loop_local().semaphore
        outcomes = await _all_or_nothing(
            {
                question_id: self._answer(semaphore, info, request.state, question)
                for question_id, question in request.questions.items()
            }
        )
        meters = [o.meter for o in outcomes.values()]
        extra: dict[str, int] = {}
        for meter in meters:
            for name, count in meter.usage_extra().items():
                extra[name] = extra.get(name, 0) + count
        return SystemOneResponse(
            model=response_model_id(info),
            answers={question_id: o.answer for question_id, o in outcomes.items()},
            usage=Usage(
                input_tokens=sum(o.prompt_tokens for o in outcomes.values()),
                output_tokens=sum(m.generated_tokens for m in meters),
                **extra,
            ),
            x_jevemu={
                question_id: o.diagnostics
                for question_id, o in outcomes.items()
                if o.diagnostics is not None
            }
            if self.include_diagnostics
            else None,
        )

    def system_one_sync(self, request: SystemOneRequest) -> SystemOneResponse:
        """Blocking :meth:`system_one` on a fresh event loop (``asyncio.run``); cannot be called
        from a running event loop."""
        return asyncio.run(self.system_one(request))

    # -- internals --

    def _loop_local(self) -> _LoopLocal:
        loop = asyncio.get_running_loop()
        if self._local is None or self._local.loop is not loop:
            self._local = _LoopLocal(loop, asyncio.Semaphore(self.max_concurrency), asyncio.Lock())
        return self._local

    async def _answer(
        self, semaphore: asyncio.Semaphore, info: BackendInfo, state: JSONish, question: Question
    ) -> _Outcome:
        async with semaphore:
            meter = _Meter()
            _METER.set(meter)  # this task's own context: invisible to other questions
            started = time.perf_counter()
            result = await self._score(state, question)
            debiased = (
                None
                if self.debiaser is None
                else await self.debiaser.debias(state, question, result, self._score)
            )
            logprobs = result.logprobs if debiased is None else debiased.logprobs
            probs, calibrator = self._calibrate(info, question, logprobs)
            answer = self._assemble(question, result.keys, probs)
            latency_ms = (time.perf_counter() - started) * 1000.0
        diagnostics = (
            self._diagnostics(info, result, debiased, calibrator, latency_ms, meter.api_usage())
            if self.include_diagnostics
            else None
        )
        prompt_tokens = result.prompt_tokens + (debiased.extra_prompt_tokens if debiased else 0)
        return _Outcome(answer, diagnostics, prompt_tokens, meter)

    async def _score(self, state: JSONish, question: Question) -> ScoreResult:
        """``strategy.score`` on the metered backend, checking the keys it scored."""
        result = await self.strategy.score(self._metered, self.renderer, state, question)
        keys = answer_keys(question)
        if result.keys != keys:
            raise ValueError(
                f"strategy {result.strategy!r} scored keys {list(result.keys)}, "
                f"expected {list(keys)}"
            )
        return result

    def _calibrate(
        self, info: BackendInfo, question: Question, logprobs: Sequence[float]
    ) -> tuple[list[float], str | None]:
        """Calibrated probabilities in key order, and ``"<name>:<source>"`` (None: no registry)."""
        if self.calibrators is None:
            return [math.exp(lp) for lp in logprobs], None
        key = CalibrationKey.for_question(
            backend=info.backend,
            model=info.model,
            template_id=self.renderer.template_id,
            question=question,
        )
        resolved = self.calibrators.resolve(key)
        row = resolved.calibrator.transform(np.asarray([logprobs], dtype=np.float64))[0]
        return [_clip(float(p)) for p in row], f"{resolved.calibrator.name}:{resolved.source}"

    def _diagnostics(
        self,
        info: BackendInfo,
        result: ScoreResult,
        debiased: DebiasOutcome | None,
        calibrator: str | None,
        latency_ms: float,
        api: ApiUsage | None,
    ) -> EmulatorDiagnostics:
        keys = result.keys
        extra_calls = debiased.extra_calls if debiased else 0
        presentations = debiased.presentations if debiased else ()
        return EmulatorDiagnostics(
            backend=info.backend,
            backend_model=info.model,
            vllm_version=info.engine_version,
            strategy=result.strategy,
            label_scheme=result.label_scheme,
            permutations=max(1, len(presentations)),
            observed_mass=_clip(result.observed_mass),
            missing_labels=list(result.missing),
            raw_probabilities={key: _clip(p) for key, p in result.probabilities.items()},
            debiaser=None if self.debiaser is None else self.debiaser.id,
            debias_calls=extra_calls,
            debias_prior=None if debiased is None else debiased.prior,
            permutation_scores=[
                PermutationScore(
                    order=[keys[j] for j in p.order],
                    probabilities={
                        keys[j]: _clip(math.exp(lp)) for j, lp in enumerate(p.by_option())
                    },
                )
                for p in presentations
            ],
            calibrator=calibrator,
            n_backend_calls=result.n_backend_calls + extra_calls,
            latency_ms=latency_ms,
            cached_tokens=result.cached_tokens,
            warnings=list(result.warnings) + list(debiased.warnings if debiased else ()),
            api=api,
        )

    def _assemble(self, question: Question, keys: Sequence[str], probs: Sequence[float]) -> _Answer:
        rnd = self._round
        if not isinstance(question, (ChoiceQuestion, ScoreQuestion)):
            return NoulAnswer(noul=rnd(probs[keys.index("true")]))
        probabilities = {key: rnd(p) for key, p in zip(keys, probs, strict=True)}
        confidence = rnd(self.confidence_fn(probs, ordinal=isinstance(question, ScoreQuestion)))
        if isinstance(question, ChoiceQuestion):
            best = max(range(len(keys)), key=probs.__getitem__)  # first maximum wins ties
            return ChoiceAnswer(
                choice=keys[best], probabilities=probabilities, confidence=confidence
            )
        top = float(len(keys) - 1)
        expected = math.fsum(i * p for i, p in enumerate(probs))
        return ScoreAnswer(
            score=rnd(_clip(expected, top), top),
            legend={str(i): level for i, level in enumerate(question.criteria)},
            probabilities=probabilities,
            confidence=confidence,
        )

    def _round(self, value: float, upper: float = 1.0) -> float:
        """``value`` to the nearest multiple of ``round_to`` (unchanged without), in [0, upper]."""
        if self.round_to is None:
            return value
        # round(..., 12) drops the float noise of the multiplication (0.88000000000000001).
        return _clip(round(round(value / self.round_to) * self.round_to, 12), upper)


def _resolve_renderer(renderer: PromptRenderer | str, *, prefill: bool) -> PromptRenderer:
    """``renderer`` as given, or the named layout rendered with the prefill iff the backend can
    continue one."""
    if isinstance(renderer, PromptRenderer):
        return renderer
    if renderer == "default":
        return PromptRenderer(prefill=prefill)
    for layout in LAYOUTS:
        if renderer == layout:
            return PromptRenderer(layout, prefill=prefill)
    raise ValueError(f"unknown renderer {renderer!r}; expected 'default' or one of {LAYOUTS}")


def _resolve_strategy(strategy: ScoringStrategy | Literal["auto"]) -> ScoringStrategy:
    if not isinstance(strategy, str):
        return strategy
    if strategy != "auto":
        raise ValueError(f"unknown strategy {strategy!r}; pass 'auto' or a ScoringStrategy")
    return AutoStrategy()


def _clip(value: float, upper: float = 1.0) -> float:
    return min(upper, max(0.0, value))


async def _all_or_nothing(jobs: Mapping[str, Coroutine[Any, Any, _T]]) -> dict[str, _T]:
    """Run ``jobs`` concurrently; on the first failure cancel the rest and raise
    :class:`QuestionError` for the failed job listed first. Results keep ``jobs`` order."""
    tasks = {name: asyncio.ensure_future(job) for name, job in jobs.items()}
    try:
        await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_EXCEPTION)
    finally:
        pending = [task for task in tasks.values() if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    for name, task in tasks.items():
        error = None if task.cancelled() else task.exception()
        if error is not None:
            raise QuestionError(name, error) from error
    return {name: task.result() for name, task in tasks.items()}
