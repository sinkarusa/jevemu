"""S1 first-token strategy: one call, logprobs of the label surfaces at the answer position.

The prompt ends with the ``Answer:`` prefill. Where the answer is read, and which surfaces count,
depends on how the backend's tokenizer splits the labels (checked once per backend with
:func:`~jevemu.render.labels.verify_single_token`; see :mod:`jevemu.scoring.constrained` for the
live evidence behind these rules):

- **Direct.** When every label's space-prefixed form (``" A"``, ``" Yes"``) is one token, read
  right after ``Answer:`` and merge every single-token surface of a label (``"A"`` and ``" A"``).
- **After the gap.** When no label's space-prefixed form is one token but each splits into a
  space token plus the bare label (Qwen3 digits: ``" 7"`` = ``" "`` + ``"7"``), the space is the
  shared first step of every label's natural path. The prefill is extended to ``Answer: `` and
  the bare labels are read there.
- Otherwise read directly; labels without a single-token surface go to the missing-label policy.
- **No local tokenizer** (``Capabilities.local_tokenizer`` false, hosted APIs): nothing can be
  checked, so every surface (``"A"``, ``" A"``) is read directly as returned.
- **Structured read** (a constrained read without ``assistant_prefill``: a hosted chat API's
  Structured Outputs). The reply is a JSON value from an ``enum`` of the bare labels; the
  backend reads the distribution over the labels where that value starts
  (:func:`~jevemu.backends.chat_api.parse_structured_response`).

Top-k: an unconstrained read requests the full ``top_logprobs_max`` because non-label tokens
compete for the slots. A label must fit the top-k with each of its competing surfaces
(:func:`surface_variants`); a label that still misses it gets the upper bound
(:mod:`jevemu.scoring.missing_mass`). A constrained read whose mask is reflected in the
logprobs requests only the grammar's possible first tokens, except a structured read: JSON
tokens (``"A``, ``"``) precede or share the value's first token, so it requests the full top-k.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from jevemu.backends.base import Backend, Capabilities, CapabilityError, ChatMessage, RenderedPrompt
from jevemu.render.labels import (
    SURFACE_PREFIXES,
    LabelCapacityError,
    LabelScheme,
    scheme_for,
    verify_single_token,
)
from jevemu.render.renderer import PromptRenderer
from jevemu.scoring.aggregate import (
    low_mass_warning,
    merge_variants,
    observed_mass,
    observed_surfaces,
    renormalize,
)
from jevemu.scoring.base import ScoreResult
from jevemu.scoring.missing_mass import fill_missing
from jevemu.types import JSONish, Question

__all__ = [
    "GAP",
    "FirstTokenStrategy",
    "ReadingPlan",
    "extend_prefill",
    "fits_top_k",
    "grammar_first_tokens",
    "reading_plan",
    "score_first_token",
    "single_token_surfaces",
    "surface_variants",
    "token_ids",
    "token_text",
]

GAP = " "
"""The space a label is generated with after ``Answer:`` (the model writes ``" A"``)."""


@dataclass
class _TokenizerCache:
    single_token: dict[str, bool] = field(default_factory=dict)
    """Surface -> is one token, from ``verify_single_token``."""
    baseline: tuple[int, ...] | None = None
    """Ids the tokenizer adds to every input (measured on ``""``)."""
    ids: dict[str, tuple[int, ...]] = field(default_factory=dict)
    texts: dict[int, str] = field(default_factory=dict)
    """Token id -> its text decoded alone."""


_CACHES: weakref.WeakKeyDictionary[object, _TokenizerCache] = weakref.WeakKeyDictionary()


def _cache_for(backend: Backend) -> _TokenizerCache:
    """Per-backend cache (a backend serves one model); fresh if it cannot be weakly keyed."""
    try:
        return _CACHES.setdefault(backend, _TokenizerCache())
    except TypeError:
        return _TokenizerCache()


async def single_token_surfaces(backend: Backend, labels: Sequence[str]) -> dict[str, bool]:
    """:func:`verify_single_token` for every surface of ``labels``, cached per backend."""
    cache = _cache_for(backend).single_token
    surfaces = [prefix + label for label in labels for prefix in SURFACE_PREFIXES]
    todo = [
        label
        for label in dict.fromkeys(labels)
        if any(prefix + label not in cache for prefix in SURFACE_PREFIXES)
    ]
    if todo:
        cache.update(await verify_single_token(backend, todo))
    return {surface: cache[surface] for surface in surfaces}


async def token_ids(backend: Backend, text: str) -> tuple[int, ...]:
    """Token ids of ``text`` without the special tokens the tokenizer adds to every input.

    Cached per backend. Special tokens are measured on the empty string, as
    :func:`verify_single_token` does.
    """
    cache = _cache_for(backend)
    if cache.baseline is None:
        cache.baseline = tuple(await backend.tokenize(""))
    if text not in cache.ids:
        cache.ids[text] = _strip(tuple(await backend.tokenize(text)), cache.baseline)
    return cache.ids[text]


async def token_text(backend: Backend, token_id: int) -> str:
    """Text of one token decoded alone, cached per backend."""
    cache = _cache_for(backend).texts
    if token_id not in cache:
        cache[token_id] = await backend.detokenize([token_id])
    return cache[token_id]


def _strip(ids: tuple[int, ...], baseline: tuple[int, ...]) -> tuple[int, ...]:
    """``ids`` without the baseline's leading/trailing special tokens (BOS ... EOS)."""
    for split in range(len(baseline) + 1):
        head, tail = baseline[:split], baseline[split:]
        end = len(ids) - len(tail)
        if end >= split and ids[:split] == head and ids[end:] == tail:
            return ids[split:end]
    return ids


@dataclass(frozen=True)
class ReadingPlan:
    """Where to read a scheme's labels and which token texts count for each label."""

    gap: str
    """Text appended to the prefill before reading (``""`` or ``" "``)."""
    surfaces: Mapping[str, str]
    """Readable single-token surface -> label, label-major."""

    def surfaces_of(self, label: str) -> tuple[str, ...]:
        return tuple(s for s, owner in self.surfaces.items() if owner == label)


async def reading_plan(
    backend: Backend, scheme: LabelScheme, *, constrained: bool = False
) -> ReadingPlan:
    """How ``backend``'s tokenizer lets a first-token call read ``scheme`` (see module doc);
    ``constrained`` for a read under a constraint over the plan's surfaces."""
    labels = scheme.labels
    caps = backend.capabilities
    if constrained and not caps.assistant_prefill:
        return ReadingPlan("", {label: label for label in labels})
    if not caps.local_tokenizer:
        return ReadingPlan("", {s: label for label in labels for s in scheme.variants(label)})
    single = await single_token_surfaces(backend, scheme.labels)
    direct = {s: label for label in labels for s in scheme.variants(label) if single[s]}
    if all(single[GAP + label] for label in labels):
        return ReadingPlan("", direct)
    if not any(single[GAP + label] for label in labels) and all(single[label] for label in labels):
        gap_ids, spaced, bare = await asyncio.gather(
            token_ids(backend, GAP),
            asyncio.gather(*(token_ids(backend, GAP + label) for label in labels)),
            asyncio.gather(*(token_ids(backend, label) for label in labels)),
        )
        if len(gap_ids) == 1 and all(s == gap_ids + b for s, b in zip(spaced, bare, strict=True)):
            return ReadingPlan(GAP, {label: label for label in labels})
    return ReadingPlan("", direct)


def extend_prefill(prompt: RenderedPrompt, text: str) -> RenderedPrompt:
    """``prompt`` with ``text`` appended to its assistant prefill."""
    if not text:
        return prompt
    messages = (*prompt.messages[:-1], ChatMessage("assistant", prompt.prefill + text))
    return RenderedPrompt(messages=messages, template_id=prompt.template_id)


def grammar_first_tokens(allowed: Sequence[str]) -> int:
    """Upper bound on distinct first tokens a ``choice`` constraint over ``allowed`` permits.

    The grammar accepts any token that is a prefix of an allowed string (``" "`` for
    ``" A"``, ``"Ye"`` for ``"Yes"``). With the mask reflected, all of them rank above the
    masked tokens, so this many top-k slots show every allowed label.
    """
    return len({text[:end] for text in allowed for end in range(1, len(text) + 1)})


def surface_variants(capabilities: Capabilities) -> int:
    """Surfaces per label that compete for top-k slots.

    After the ``Answer:`` prefill both ``"A"`` and ``" A"`` are likely. At the start of a new
    assistant turn (no prefill) the bare surface carries the label, so one slot per label is
    counted; a space-prefixed surface is still merged when it shows up.
    """
    return len(SURFACE_PREFIXES) if capabilities.assistant_prefill else 1


def fits_top_k(capabilities: Capabilities, question: Question) -> bool:
    """Every competing surface of every letter/digit/Yes-No label fits in one top-k list."""
    try:
        scheme = scheme_for(question)
    except LabelCapacityError:
        return False
    return len(scheme) * surface_variants(capabilities) <= capabilities.top_logprobs_max


async def score_first_token(
    backend: Backend,
    renderer: PromptRenderer,
    state: JSONish,
    question: Question,
    *,
    constrained: bool,
    name: str,
    scheme: LabelScheme | None = None,
) -> ScoreResult:
    """Shared S1/S2 scoring: one first-token call, variants merged, missing labels filled.

    ``scheme`` defaults to :func:`~jevemu.render.labels.scheme_for` of ``question``.
    """
    if scheme is None:
        scheme = scheme_for(question)
    plan = await reading_plan(backend, scheme, constrained=constrained)
    prompt = extend_prefill(renderer.render(state, question, scheme), plan.gap)
    caps = backend.capabilities
    allowed: list[str] | None = None
    top_k = caps.top_logprobs_max
    if constrained:
        if not plan.surfaces:
            raise ValueError(
                f"no {scheme.name} label is a single token for this backend; "
                "a first-token constraint cannot express any label"
            )
        allowed = list(plan.surfaces)
        if caps.mask_reflected_in_logprobs and caps.assistant_prefill:
            top_k = min(top_k, grammar_first_tokens(allowed))
    dist = await backend.next_token_logprobs(prompt, allowed=allowed, top_k=top_k)
    seen = observed_surfaces(dist.top, plan.surfaces)
    merged = merge_variants(seen, plan.surfaces)
    missing = {
        label: plan.surfaces_of(label) or scheme.variants(label)
        for label in scheme.labels
        if label not in merged
    }
    fill = await fill_missing(backend, prompt, dist, missing, observed=seen, allowed=allowed or ())
    raw = tuple(
        merged[label] if label in merged else fill.logprobs[label] for label in scheme.labels
    )
    mass = observed_mass(merged.values())
    warnings = list(fill.warnings)
    low = low_mass_warning(mass, caps.logprobs_mode)
    if low is not None:
        warnings.append(low)
    return ScoreResult(
        keys=scheme.keys,
        logprobs=renormalize(raw),
        raw_logprobs=raw,
        observed_mass=mass,
        missing=tuple(scheme.key_of(label) for label in scheme.labels if label in missing),
        truncated=fill.truncated,
        strategy=name,
        label_scheme=scheme.name,
        n_backend_calls=1 + fill.n_backend_calls,
        prompt_tokens=dist.prompt_tokens + fill.prompt_tokens,
        cached_tokens=dist.cached_tokens,
        warnings=tuple(warnings),
    )


class FirstTokenStrategy:
    """S1: unconstrained first-token logprobs, renormalized over the labels.

    On a backend without ``assistant_prefill`` (a hosted chat API) a question whose labels do not
    fit the top-k raises ``CapabilityError``: a read that sees at most ``top_logprobs_max`` of its
    labels is not reported as an answer (gpt-6-luna: 5 top logprobs, so MMLU-Pro's 10 options,
    banking77 and CLINC150 are out).
    """

    name = "first_token"

    def supports(self, capabilities: Capabilities, question: Question) -> bool:
        return fits_top_k(capabilities, question)

    async def score(
        self, backend: Backend, renderer: PromptRenderer, state: JSONish, question: Question
    ) -> ScoreResult:
        caps = backend.capabilities
        if not caps.assistant_prefill and not fits_top_k(caps, question):
            raise CapabilityError(
                f"the question's labels do not fit the backend's {caps.top_logprobs_max} top "
                "logprobs, and without assistant_prefill a first-token read cannot see them all"
            )
        return await score_first_token(
            backend, renderer, state, question, constrained=False, name=self.name
        )
