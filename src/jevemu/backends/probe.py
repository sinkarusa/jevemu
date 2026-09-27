"""Capability probe: measure what a live vLLM server actually does.

The probe sends a handful of fixed requests and records raw evidence for each question the
scoring code depends on. First-token requests take the adapter's own path (exact prompt ids
from ``/tokenize`` on ``/v1/completions``), so every capability is measured where scoring uses it:

a. **Constraint enforced.** ``structured_outputs.choice=["Q", "Z"]`` on a question whose answer
   is never Q or Z must still produce Q or Z; otherwise :class:`ConstraintNotEnforced`.
b. **Mask reflected in logprobs.** The same two-option prompt with and without the constraint.
   If the constrained top-k holds only allowed tokens with finite logprobs and their mass is ~1,
   the reported logprobs are post-mask.
c. **``--max-logprobs`` cap.** ``logprobs`` at and above the declared cap: error, silent
   truncation, or accepted.
d. **Label tokenization.** Single-token status of ``A..Z``, ``a..z``, ``0..9``, ``Yes``/``No``,
   bare and with a leading space.
e. **Echo.** ``max_tokens=0`` echo scoring, the null first prompt logprob, prefix-cache reads.
f. **Prefill.** The adapter's prompt (generation prompt, then the prefill as text) ends with the
   prefill and reaches the model whole. It is also compared with the template's own rendering
   of the prefill as a continued assistant message (``continue_final_message``); a mismatch is
   recorded, not an error: the template emits something only on the generation prompt (Gemma 4
   12B/26B-A4B with thinking off: the empty thought channel), which the adapter keeps.

Results are cached as JSON per (vLLM version, model, revision, image, effective chat-template
kwargs, declared flags) under ``~/.cache/jevemu/probes/``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import re
import string
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from jevemu.backends.base import Capabilities, ChatMessage, NextTokenDist, RenderedPrompt
from jevemu.backends.vllm_http import (
    COMPLETIONS_PATH,
    LOGPROB_FLOOR,
    TOKENIZE_PATH,
    VLLMHTTPBackend,
    VLLMHTTPError,
    build_echo_request,
    build_first_token_request,
    first_token_ids,
    parse_first_token_response,
)

__all__ = [
    "LABELS",
    "MCQ_PROMPT",
    "PREFILL_PROMPT",
    "PROBE_SCHEMA_VERSION",
    "ConstraintNotEnforced",
    "ProbeReport",
    "ServerFlags",
    "default_cache_dir",
    "format_report",
    "load_or_run_probe",
    "probe_capabilities",
    "run_probe",
]

PROBE_SCHEMA_VERSION = 2
"""Bump when probe semantics or the report layout change; invalidates cached reports."""

LABELS: tuple[str, ...] = (
    *string.ascii_uppercase,
    *string.ascii_lowercase,
    *string.digits,
    "Yes",
    "No",
)

MCQ_PROMPT = RenderedPrompt(
    messages=(
        ChatMessage(
            "system", "Answer the multiple-choice question with the letter of the correct option."
        ),
        ChatMessage("user", "Which of these is a fruit?\nA) Hammer\nB) Apple"),
        ChatMessage("assistant", "Answer:"),
    ),
    template_id="probe/mcq",
)
"""Letter-answer multiple-choice prompt ending in an ``Answer:`` prefill; ``B`` is correct."""
_MCQ_LABELS = ("A", "B")
_FOREIGN_LABELS = ("Q", "Z")

PREFILL_PROMPT = RenderedPrompt(
    messages=(
        ChatMessage("user", "What is the capital of France?"),
        ChatMessage("assistant", "The capital of France is"),
    ),
    template_id="probe/prefill",
)
"""Prefill prompt whose best continuation is `` Paris``."""
_ECHO_TEXT = "The capital of France is Paris."
_ECHO_CONTINUATIONS = (" Paris", " London", " Paris, the city of light.")

_MASS_TOLERANCE = 1e-3
_EVIDENCE_TOP = 8
_DRIFT_FLOOR = -30.0
_BATCH = 16


class ConstraintNotEnforced(RuntimeError):
    """The server ignored ``structured_outputs.choice``: constrained scoring is unsafe."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ServerFlags(_Model):
    """Launch flags the probe cannot read from the server; they key the probe cache."""

    max_logprobs: int = 20
    """Declared ``--max-logprobs`` (verified by probe c)."""
    logprobs_mode: Literal["raw_logprobs", "processed_logprobs"] = "raw_logprobs"
    """Declared ``--logprobs-mode``; vLLM does not expose it over HTTP."""


class TokenEvidence(_Model):
    token: str
    logprob: float
    """As reported by vLLM (``-9999.0`` floor kept)."""


class ConstraintProbe(_Model):
    allowed: list[str]
    unconstrained_token: str
    constrained_token: str
    enforced: bool


class MaskProbe(_Model):
    allowed: list[str]
    top_logprobs: int
    unconstrained_top: list[TokenEvidence]
    constrained_top: list[TokenEvidence]
    constrained_returned: int
    """Entries in the constrained top-k (evidence lists are truncated)."""
    invalid_finite: list[str]
    """Tokens outside ``allowed`` with a finite logprob in the constrained response."""
    floored: int
    """Constrained top-k entries at vLLM's -9999 floor (masked tokens)."""
    allowed_mass: float
    """Summed probability of the allowed tokens in the constrained response."""
    reflected: bool


class CapAttempt(_Model):
    requested: int
    status: int
    returned: int | None = None
    error: str | None = None


class MaxLogprobsProbe(_Model):
    declared: int
    at_cap: CapAttempt
    above_cap: CapAttempt
    above_cap_behavior: Literal["error", "truncated", "accepted"]
    effective_cap: int


class LabelTokens(_Model):
    label: str
    bare: list[int]
    spaced: list[int]


class LabelProbe(_Model):
    labels: list[LabelTokens]
    multi_token_bare: list[str]
    multi_token_spaced: list[str]


class EchoProbe(_Model):
    available: bool
    error: str | None = None
    max_tokens_zero: bool
    """``max_tokens=0`` accepted and no generated token appended."""
    first_token_null: bool
    prompt_tokens: list[str]
    token_logprobs: list[float | None]
    repeat_cached_tokens: int | None
    """``cached_tokens`` on an identical second echo request (0: prefix cache not read)."""
    continuation_totals: dict[str, float]
    """Adapter-level ``sequence_logprobs`` totals after the prefill prompt."""


class PrefillProbe(_Model):
    prefill: str
    rendered_tail: str
    """Detokenized tail of the adapter's prompt: generation prompt, then the prefill as text."""
    rendered_ends_with_prefill: bool
    tokenize_count: int
    prompt_tokens: int
    """``usage.prompt_tokens`` of the first-token call on the adapter's ids."""
    sampled_token: str
    continued_tail: str
    """Detokenized tail of the template's ``continue_final_message`` rendering."""
    continued_matches: bool
    """The template renders the prefill as a continued assistant message to the adapter's ids.
    ``False`` for templates that emit something only on the generation prompt."""
    honored: bool


class PrefixCacheProbe(_Model):
    metrics_enable_prefix_caching: bool | None
    repeat_cached_tokens: int | None
    """``cached_tokens`` on the second of two chat requests sharing a prompt."""
    enabled: bool


class DeterminismProbe(_Model):
    """How stable reported logprobs are across prefix-cache hits and batch composition."""

    echo_vs_uncached_gap: float | None
    """|echo - first-token| logprob of " Paris" after a fresh (never cached) prompt."""
    cached_tokens: int | None
    """``cached_tokens`` of the repeated first-token request."""
    cached_drift: float
    """Max |delta logprob| over shared top-k tokens (logprob > -30), uncached vs cached call."""
    batch_size: int
    batch_spread: float
    """max - min logprob of the top token across ``batch_size`` concurrent identical requests."""


class ExplicitTokenProbe(_Model):
    requested: dict[str, int]
    returned: list[TokenEvidence]
    error: str | None = None
    server_supports: bool
    """``logprob_token_ids`` works on this server. The adapter does not use it yet,
    so ``Capabilities.explicit_token_logprobs`` stays False."""


class ProbeReport(_Model):
    schema_version: int = PROBE_SCHEMA_VERSION
    created_at: str
    base_url: str
    vllm_version: str
    model: str
    model_revision: str | None
    image: str | None
    flags: ServerFlags
    backend_flags: dict[str, str]
    constraint: ConstraintProbe
    mask: MaskProbe
    max_logprobs: MaxLogprobsProbe
    labels: LabelProbe
    echo: EchoProbe
    prefill: PrefillProbe
    prefix_cache: PrefixCacheProbe
    determinism: DeterminismProbe
    explicit_token_ids: ExplicitTokenProbe
    capabilities: Capabilities


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "jevemu" / "probes"


def _evidence(entries: Sequence[Mapping[str, Any]]) -> list[TokenEvidence]:
    return [TokenEvidence(token=e["token"], logprob=e["logprob"]) for e in entries]


async def _top_entries(
    backend: VLLMHTTPBackend, payload: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """The first token's top-k (a ``{"token_id:N": logprob}`` map in completions) with each
    token's text, best first."""
    alternatives: Mapping[str, float] = payload["choices"][0]["logprobs"]["top_logprobs"][0] or {}
    ids = first_token_ids(payload)[1:]
    texts = await backend.token_texts(ids)
    entries = [
        {"token": texts[i], "logprob": v} for i, v in zip(ids, alternatives.values(), strict=True)
    ]
    return sorted(entries, key=lambda e: -e["logprob"])


async def _parse(
    backend: VLLMHTTPBackend, payload: Mapping[str, Any], *, constrained: bool, top_logprobs: int
) -> NextTokenDist:
    texts = await backend.token_texts(first_token_ids(payload))
    return parse_first_token_response(
        payload, constrained=constrained, top_logprobs=top_logprobs, texts=texts
    )


async def _first_token(
    backend: VLLMHTTPBackend,
    prompt: RenderedPrompt,
    *,
    allowed: Sequence[str] | None,
    top_logprobs: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Raw completions payload of the adapter's first-token request (plus ``extra`` fields)."""
    ids = await backend.tokenize_chat(prompt)
    body = build_first_token_request(backend.model, ids, allowed=allowed, top_logprobs=top_logprobs)
    payload: dict[str, Any] = await backend.request_json(
        "POST", COMPLETIONS_PATH, {**body, **(extra or {})}
    )
    return payload


async def _probe_constraint(backend: VLLMHTTPBackend) -> ConstraintProbe:
    free, forced = await asyncio.gather(
        _first_token(backend, MCQ_PROMPT, allowed=None, top_logprobs=1),
        _first_token(backend, MCQ_PROMPT, allowed=_FOREIGN_LABELS, top_logprobs=1),
    )
    free_token = (await _parse(backend, free, constrained=False, top_logprobs=1)).sampled.token
    forced_token = (await _parse(backend, forced, constrained=True, top_logprobs=1)).sampled.token
    return ConstraintProbe(
        allowed=list(_FOREIGN_LABELS),
        unconstrained_token=free_token,
        constrained_token=forced_token,
        enforced=forced_token in _FOREIGN_LABELS,
    )


async def _probe_mask(backend: VLLMHTTPBackend, top_logprobs: int) -> tuple[MaskProbe, int | None]:
    """Returns the probe and the constrained call's ``cached_tokens`` (same prompt, second)."""
    free = await _first_token(backend, MCQ_PROMPT, allowed=None, top_logprobs=top_logprobs)
    forced = await _first_token(backend, MCQ_PROMPT, allowed=_MCQ_LABELS, top_logprobs=top_logprobs)
    forced_top, free_top = await asyncio.gather(
        _top_entries(backend, forced), _top_entries(backend, free)
    )
    finite = [e for e in forced_top if e["logprob"] > LOGPROB_FLOOR]
    invalid = [e["token"] for e in finite if e["token"] not in _MCQ_LABELS]
    mass = math.fsum(math.exp(e["logprob"]) for e in finite if e["token"] in _MCQ_LABELS)
    probe = MaskProbe(
        allowed=list(_MCQ_LABELS),
        top_logprobs=top_logprobs,
        unconstrained_top=_evidence(free_top[:_EVIDENCE_TOP]),
        constrained_top=_evidence(forced_top[:_EVIDENCE_TOP]),
        constrained_returned=len(forced_top),
        invalid_finite=invalid,
        floored=len(forced_top) - len(finite),
        allowed_mass=mass,
        reflected=not invalid and abs(mass - 1.0) < _MASS_TOLERANCE,
    )
    cached = (
        await _parse(backend, forced, constrained=True, top_logprobs=top_logprobs)
    ).cached_tokens
    return probe, cached


async def _cap_attempt(backend: VLLMHTTPBackend, n: int) -> CapAttempt:
    try:
        payload = await _first_token(backend, MCQ_PROMPT, allowed=None, top_logprobs=n)
    except VLLMHTTPError as exc:
        return CapAttempt(requested=n, status=exc.status_code, error=exc.detail)
    # Keyed by token id, so distinct tokens with the same text are all counted.
    returned = len(first_token_ids(payload)) - 1
    return CapAttempt(requested=n, status=200, returned=min(n, returned))


async def _probe_max_logprobs(backend: VLLMHTTPBackend, declared: int) -> MaxLogprobsProbe:
    at_cap = await _cap_attempt(backend, declared)
    above = await _cap_attempt(backend, declared + 1)
    if at_cap.status == 200 and at_cap.returned is not None:
        effective = min(declared, at_cap.returned)
    else:
        lo, hi = 0, declared - 1  # largest accepted value in [0, declared)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if (await _cap_attempt(backend, mid)).status == 200:
                lo = mid
            else:
                hi = mid - 1
        effective = lo
    behavior: Literal["error", "truncated", "accepted"]
    if above.status != 200:
        behavior = "error"
    elif (above.returned or 0) < above.requested:
        behavior = "truncated"
    else:
        behavior = "accepted"
    return MaxLogprobsProbe(
        declared=declared,
        at_cap=at_cap,
        above_cap=above,
        above_cap_behavior=behavior,
        effective_cap=effective,
    )


async def _probe_labels(backend: VLLMHTTPBackend) -> LabelProbe:
    bare = await asyncio.gather(*(backend.tokenize(label) for label in LABELS))
    spaced = await asyncio.gather(*(backend.tokenize(" " + label) for label in LABELS))
    labels = [
        LabelTokens(label=label, bare=b, spaced=s)
        for label, b, s in zip(LABELS, bare, spaced, strict=True)
    ]
    return LabelProbe(
        labels=labels,
        multi_token_bare=[t.label for t in labels if len(t.bare) != 1],
        multi_token_spaced=[t.label for t in labels if len(t.spaced) != 1],
    )


async def _probe_echo(backend: VLLMHTTPBackend) -> EchoProbe:
    ids = await backend.tokenize(_ECHO_TEXT)
    body = build_echo_request(backend.model, ids)
    try:
        first = await backend.request_json("POST", COMPLETIONS_PATH, body)
        repeat = await backend.request_json("POST", COMPLETIONS_PATH, body)
    except VLLMHTTPError as exc:
        return EchoProbe(
            available=False,
            error=exc.detail,
            max_tokens_zero=False,
            first_token_null=False,
            prompt_tokens=[],
            token_logprobs=[],
            repeat_cached_tokens=None,
            continuation_totals={},
        )
    logprobs = first["choices"][0]["logprobs"]
    values: list[float | None] = logprobs["token_logprobs"]
    details = repeat.get("usage", {}).get("prompt_tokens_details") or {}
    max_tokens_zero = len(values) == len(ids)

    totals: dict[str, float] = {}
    if max_tokens_zero and backend.capabilities.echo_prompt_logprobs:
        scores = await backend.sequence_logprobs(PREFILL_PROMPT, _ECHO_CONTINUATIONS)
        totals = {s.continuation: s.total for s in scores}
    return EchoProbe(
        available=max_tokens_zero,
        max_tokens_zero=max_tokens_zero,
        first_token_null=bool(values) and values[0] is None,
        prompt_tokens=list(logprobs["tokens"]),
        token_logprobs=values,
        repeat_cached_tokens=details.get("cached_tokens"),
        continuation_totals=totals,
    )


def _logprob_of(dist: NextTokenDist, token: str) -> float | None:
    return next((t.logprob for t in dist.top if t.token == token), None)


async def _probe_determinism(backend: VLLMHTTPBackend) -> DeterminismProbe:
    nonce = ChatMessage("system", f"Probe {uuid.uuid4().hex}.")  # defeats the prefix cache
    prompt = RenderedPrompt((nonce, *PREFILL_PROMPT.messages), template_id="probe/determinism")
    fresh = await backend.next_token_logprobs(prompt, allowed=None, top_k=20)
    gap = None
    if backend.capabilities.echo_prompt_logprobs:
        (score,) = await backend.sequence_logprobs(prompt, [" Paris"])
        paris = _logprob_of(fresh, " Paris")
        if paris is not None and len(score.tokens) == 1:
            gap = abs(score.total - paris)
    cached = await backend.next_token_logprobs(prompt, allowed=None, top_k=20)
    shared = [
        (t.logprob, other)
        for t in fresh.top
        if t.logprob > _DRIFT_FLOOR and (other := _logprob_of(cached, t.token)) is not None
    ]
    batch = await asyncio.gather(
        *(backend.next_token_logprobs(prompt, allowed=None, top_k=20) for _ in range(_BATCH))
    )
    top_values = [v for d in batch if (v := _logprob_of(d, fresh.top[0].token)) is not None]
    return DeterminismProbe(
        echo_vs_uncached_gap=gap,
        cached_tokens=cached.cached_tokens,
        cached_drift=max((abs(a - b) for a, b in shared), default=0.0),
        batch_size=_BATCH,
        batch_spread=max(top_values) - min(top_values) if top_values else 0.0,
    )


async def _probe_prefill(backend: VLLMHTTPBackend) -> PrefillProbe:
    continued_body: dict[str, Any] = {
        "model": backend.model,
        "messages": [{"role": m.role, "content": m.content} for m in PREFILL_PROMPT.messages],
        "continue_final_message": True,
        "add_generation_prompt": False,
        "add_special_tokens": False,
    }
    if backend.chat_template_kwargs:
        continued_body["chat_template_kwargs"] = dict(backend.chat_template_kwargs)
    ids, continued = await asyncio.gather(
        backend.tokenize_chat(PREFILL_PROMPT),
        backend.request_json("POST", TOKENIZE_PATH, continued_body),
    )
    continued_ids = [int(t) for t in continued["tokens"]]
    rendered, continued_text, dist = await asyncio.gather(
        backend.detokenize(ids),
        backend.detokenize(continued_ids),
        backend.next_token_logprobs(PREFILL_PROMPT, allowed=None, top_k=1),
    )
    ends = rendered.endswith(PREFILL_PROMPT.prefill)
    return PrefillProbe(
        prefill=PREFILL_PROMPT.prefill,
        rendered_tail=rendered[-80:],
        rendered_ends_with_prefill=ends,
        tokenize_count=len(ids),
        prompt_tokens=dist.prompt_tokens,
        sampled_token=dist.sampled.token,
        continued_tail=continued_text[-80:],
        continued_matches=continued_ids == ids,
        honored=ends and dist.prompt_tokens == len(ids),
    )


async def _probe_explicit_ids(backend: VLLMHTTPBackend) -> ExplicitTokenProbe:
    ids = await asyncio.gather(*(backend.tokenize(label) for label in _MCQ_LABELS))
    requested = {label: toks[0] for label, toks in zip(_MCQ_LABELS, ids, strict=True)}
    try:
        payload = await _first_token(
            backend,
            MCQ_PROMPT,
            allowed=None,
            top_logprobs=0,
            extra={"logprob_token_ids": list(requested.values())},
        )
    except VLLMHTTPError as exc:
        return ExplicitTokenProbe(
            requested=requested, returned=[], error=exc.detail, server_supports=False
        )
    returned = await _top_entries(backend, payload)
    return ExplicitTokenProbe(
        requested=requested,
        returned=_evidence(returned),
        server_supports=set(_MCQ_LABELS) <= {e["token"] for e in returned},
    )


async def run_probe(backend: VLLMHTTPBackend, flags: ServerFlags) -> ProbeReport:
    """Run every probe against the live server (no cache)."""
    info = await backend.health()
    cache_config = await backend.cache_config()
    max_logprobs = await _probe_max_logprobs(backend, flags.max_logprobs)
    constraint = await _probe_constraint(backend)
    mask, repeat_cached = await _probe_mask(backend, max_logprobs.effective_cap)
    labels = await _probe_labels(backend)
    echo = await _probe_echo(backend)
    prefill = await _probe_prefill(backend)
    explicit = await _probe_explicit_ids(backend)
    determinism = await _probe_determinism(backend)

    metrics_flag = cache_config.get("enable_prefix_caching")
    metrics_prefix = None if metrics_flag is None else metrics_flag.lower() == "true"
    prefix_cache = PrefixCacheProbe(
        metrics_enable_prefix_caching=metrics_prefix,
        repeat_cached_tokens=repeat_cached,
        enabled=metrics_prefix if metrics_prefix is not None else bool(repeat_cached),
    )
    capabilities = Capabilities(
        top_logprobs_max=max_logprobs.effective_cap,
        structured_choice=constraint.enforced,
        logprobs_mode=flags.logprobs_mode,
        mask_reflected_in_logprobs=constraint.enforced and mask.reflected,
        echo_prompt_logprobs=echo.available,
        explicit_token_logprobs=False,
        assistant_prefill=prefill.honored,
        prefix_caching=prefix_cache.enabled,
    )
    return ProbeReport(
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        base_url=backend.base_url,
        vllm_version=info.engine_version,
        model=info.model,
        model_revision=info.model_revision,
        image=backend.image,
        flags=flags,
        backend_flags={
            **info.flags,
            "max-logprobs": str(capabilities.top_logprobs_max),
            "logprobs-mode": capabilities.logprobs_mode,
        },
        constraint=constraint,
        mask=mask,
        max_logprobs=max_logprobs,
        labels=labels,
        echo=echo,
        prefill=prefill,
        prefix_cache=prefix_cache,
        determinism=determinism,
        explicit_token_ids=explicit,
        capabilities=capabilities,
    )


def _cache_path(
    cache_dir: Path,
    *,
    vllm_version: str,
    model: str,
    model_revision: str | None,
    image: str | None,
    chat_template_kwargs: Mapping[str, Any],
    flags: ServerFlags,
) -> Path:
    key = json.dumps(
        {
            "schema_version": PROBE_SCHEMA_VERSION,
            "vllm_version": vllm_version,
            "model": model,
            "model_revision": model_revision,
            "image": image,
            "chat_template_kwargs": dict(chat_template_kwargs),
            "flags": flags.model_dump(mode="json"),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    return cache_dir / f"{slug}-vllm{vllm_version}-{digest}.json"


async def load_or_run_probe(
    backend: VLLMHTTPBackend,
    flags: ServerFlags,
    *,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> ProbeReport:
    """Cached :func:`run_probe`; the key is (vLLM version, model, revision, image, effective chat
    template kwargs, flags)."""
    info = await backend.health()
    path = _cache_path(
        cache_dir or default_cache_dir(),
        vllm_version=info.engine_version,
        model=info.model,
        model_revision=info.model_revision,
        image=backend.image,
        chat_template_kwargs=backend.effective_chat_template_kwargs,
        flags=flags,
    )
    if not refresh and path.is_file():
        try:
            return ProbeReport.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            pass  # stale layout: re-probe and overwrite
    report = await run_probe(backend, flags)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)
    return report


async def probe_capabilities(
    backend: VLLMHTTPBackend,
    flags: ServerFlags,
    *,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> Capabilities:
    """Measured capabilities for ``backend``'s server (cached).

    Raises :class:`ConstraintNotEnforced` when the server ignores ``structured_outputs``.
    Assign the result to ``backend.capabilities``.
    """
    report = await load_or_run_probe(backend, flags, cache_dir=cache_dir, refresh=refresh)
    if not report.constraint.enforced:
        raise ConstraintNotEnforced(
            f"vLLM {report.vllm_version} ignored structured_outputs.choice="
            f"{report.constraint.allowed}: generated {report.constraint.constrained_token!r}"
        )
    return report.capabilities


def _tokens(evidence: Sequence[TokenEvidence]) -> str:
    return ", ".join(f"`{e.token!r}` {e.logprob:.4g}" for e in evidence)


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def format_report(report: ProbeReport) -> str:
    """Markdown summary of a probe report."""
    caps = report.capabilities
    mask, cap, echo, labels = report.mask, report.max_logprobs, report.echo, report.labels
    rows = [
        ("vLLM", report.vllm_version),
        ("model", f"{report.model} (revision {report.model_revision or 'unknown'})"),
        ("image", report.image or "unknown"),
        ("declared flags", report.flags.model_dump_json()),
        ("chat-template kwargs", report.backend_flags.get("chat-template-kwargs", "none")),
        (
            "a. constraint enforced",
            f"{report.constraint.enforced}: choice {report.constraint.allowed} -> "
            f"`{report.constraint.constrained_token!r}` "
            f"(unconstrained `{report.constraint.unconstrained_token!r}`)",
        ),
        (
            "b. mask reflected in logprobs",
            f"{mask.reflected}: allowed mass {mask.allowed_mass:.6f}, "
            f"{len(mask.invalid_finite)} invalid finite, {mask.floored}/"
            f"{mask.constrained_returned} floored at -9999",
        ),
        (
            "c. max-logprobs",
            f"declared {cap.declared}, at cap -> HTTP {cap.at_cap.status} "
            f"({cap.at_cap.returned} returned); {cap.above_cap.requested} -> "
            f"{cap.above_cap_behavior} (HTTP {cap.above_cap.status}); "
            f"effective {cap.effective_cap}",
        ),
        (
            "d. labels",
            f"{len(labels.labels)} labels; multi-token bare: {labels.multi_token_bare or 'none'}; "
            f"multi-token spaced: {labels.multi_token_spaced or 'none'}",
        ),
        (
            "e. echo",
            f"available {echo.available}, max_tokens=0 {echo.max_tokens_zero}, "
            f"first token null {echo.first_token_null}, repeat cached_tokens "
            f"{echo.repeat_cached_tokens}, totals "
            + ", ".join(f"{c!r} {v:.3f}" for c, v in echo.continuation_totals.items()),
        ),
        (
            "f. prefill",
            f"{report.prefill.honored}: tail `{report.prefill.rendered_tail[-40:]!r}`, "
            f"tokenize {report.prefill.tokenize_count} vs completion "
            f"{report.prefill.prompt_tokens} tokens, next `{report.prefill.sampled_token!r}`; "
            + (
                "continue_final_message renders the same ids"
                if report.prefill.continued_matches
                else "continue_final_message renders differently: tail "
                f"`{report.prefill.continued_tail[-40:]!r}`"
            ),
        ),
        (
            "prefix cache",
            f"{report.prefix_cache.enabled} (metrics "
            f"{report.prefix_cache.metrics_enable_prefix_caching}, repeat cached_tokens "
            f"{report.prefix_cache.repeat_cached_tokens})",
        ),
        (
            "determinism",
            f"echo vs uncached first token {_num(report.determinism.echo_vs_uncached_gap)}; "
            f"cache hit ({report.determinism.cached_tokens} tokens) drift "
            f"{report.determinism.cached_drift:.4f}; spread over "
            f"{report.determinism.batch_size} concurrent {report.determinism.batch_spread:.4f}",
        ),
        (
            "logprob_token_ids",
            f"server supports {report.explicit_token_ids.server_supports}: "
            f"{_tokens(report.explicit_token_ids.returned)}",
        ),
    ]
    lines = ["| Probe | Result |", "| --- | --- |"]
    lines += ["| " + k + " | " + v.replace("|", "\\|") + " |" for k, v in rows]
    lines += [
        "",
        f"Unconstrained top: {_tokens(mask.unconstrained_top)}",
        f"Constrained top: {_tokens(mask.constrained_top)}",
        "",
        "Capabilities: " + ", ".join(f"{k}={v}" for k, v in dataclasses.asdict(caps).items()),
    ]
    return "\n".join(lines)
