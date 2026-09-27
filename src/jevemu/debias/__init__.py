"""Debiasers: remove the emulator's answer-position bias before calibration.

- ``permutation``: mean over reordered presentations; ``n - 1`` extra scoring passes per
  Choice question (``F - 1`` for all cyclic shifts).
- ``pride``: divide by a prior estimated from permuted questions; ``F - 1`` extra passes for
  an ``alpha`` share of questions (about ``x(1 + alpha (F - 1))``), none with fitted priors.
- ``contextual``: divide by the distribution for a content-free state; ``len(content_free)``
  extra passes once per distinct question.
- ``batch``: divide by the mean distribution of a recorded batch; no extra passes (priors are
  fitted offline).

``F`` is the number of free options (``jevemu.debias.base``). Only Choice questions permute;
``contextual`` and ``batch`` also apply to Score and Noul questions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jevemu.debias.base import (
    DEFAULT_FIXED_TEXTS,
    Debiaser,
    DebiasOutcome,
    LabelPriors,
    Presentation,
    Scorer,
    apply_prior,
    average_presentations,
    cyclic_orders,
    free_positions,
    full_cycle,
    item_prior,
    permuted_question,
    prior_key,
)
from jevemu.debias.batch import BatchDebiaser, fit_batch_priors
from jevemu.debias.contextual import DEFAULT_CONTENT_FREE, ContextualDebiaser
from jevemu.debias.permutation import DEFAULT_MAX_OPTIONS, PermutationDebiaser
from jevemu.debias.pride import DEFAULT_ALPHA, PriDeDebiaser, fit_pride_priors

__all__ = [
    "DEBIASERS",
    "DEFAULT_ALPHA",
    "DEFAULT_CONTENT_FREE",
    "DEFAULT_FIXED_TEXTS",
    "DEFAULT_MAX_OPTIONS",
    "BatchDebiaser",
    "ContextualDebiaser",
    "DebiasOutcome",
    "Debiaser",
    "LabelPriors",
    "PermutationDebiaser",
    "Presentation",
    "PriDeDebiaser",
    "Scorer",
    "apply_prior",
    "average_presentations",
    "cyclic_orders",
    "debiaser_from_spec",
    "fit_batch_priors",
    "fit_pride_priors",
    "free_positions",
    "full_cycle",
    "item_prior",
    "permuted_question",
    "prior_key",
]

DEBIASERS: dict[str, type[Debiaser]] = {
    cls.name: cls for cls in (PermutationDebiaser, PriDeDebiaser, ContextualDebiaser, BatchDebiaser)
}
"""Debiaser classes by ``name``."""


def _optional_int(value: str) -> int | None:
    return None if value in ("none", "all") else int(value)


def _texts(value: str) -> tuple[str, ...]:
    return tuple(value.split("|")) if value else ()


_PARAMS: dict[str, dict[str, tuple[str, Callable[[str], Any]]]] = {
    "permutation": {
        "kind": ("kind", str),
        "n": ("n_permutations", _optional_int),
        "seed": ("seed", int),
        "max_options": ("max_options", _optional_int),
        "fixed": ("fixed_texts", _texts),
    },
    "pride": {
        "alpha": ("alpha", float),
        "seed": ("seed", int),
        "max_options": ("max_options", _optional_int),
        "priors": ("priors", LabelPriors.load),
        "fixed": ("fixed_texts", _texts),
    },
    "contextual": {
        "content_free": ("content_free", _texts),
        "fixed": ("fixed_texts", _texts),
    },
    "batch": {
        "priors": ("priors", LabelPriors.load),
        "fixed": ("fixed_texts", _texts),
    },
}
"""Spec parameter -> (constructor argument, parser) per debiaser."""


def debiaser_from_spec(spec: str) -> Debiaser:
    """Build a debiaser from ``"name"`` or ``"name:key=value,key=value"``.

    Parameters: ``permutation``: ``kind`` (cyclic|random), ``n`` (int|all), ``seed``,
    ``max_options`` (int|none), ``fixed``; ``pride``: ``alpha``, ``seed``, ``max_options``,
    ``priors`` (a :class:`LabelPriors` JSON file), ``fixed``; ``contextual``:
    ``content_free``, ``fixed``; ``batch``: ``priors`` (required), ``fixed``. List values
    (``content_free``, ``fixed``) separate items with ``|``, e.g.
    ``"contextual:content_free=N/A|[MASK]"``.
    """
    name, _, rest = spec.partition(":")
    if name not in _PARAMS:
        raise ValueError(f"unknown debiaser {name!r}; known: {sorted(_PARAMS)}")
    allowed = _PARAMS[name]
    kwargs: dict[str, Any] = {}
    for part in filter(None, rest.split(",")):
        key, sep, value = part.partition("=")
        if not sep or key not in allowed:
            raise ValueError(f"{name}: bad parameter {part!r}; known: {sorted(allowed)}")
        argument, parse = allowed[key]
        kwargs[argument] = parse(value)
    if name == "batch" and "priors" not in kwargs:
        raise ValueError("batch needs priors=<LabelPriors JSON>")
    return DEBIASERS[name](**kwargs)
