"""Prompt rendering: answer label schemes and byte-stable chat prompts ending in ``Answer:``."""

from __future__ import annotations

from jevemu.render.labels import (
    CHOICE_LABELS,
    CODE_LABELS,
    NOUL_KEYS,
    NOUL_LABELS,
    SCORE_LABELS,
    SURFACE_PREFIXES,
    LabelCapacityError,
    LabelScheme,
    SchemeName,
    answer_keys,
    choice_scheme,
    code_scheme,
    key_scheme,
    noul_scheme,
    scheme_for,
    score_scheme,
    sequence_scheme_for,
    verify_single_token,
)
from jevemu.render.renderer import (
    DEFAULT_LAYOUT,
    LAYOUTS,
    PREFILL,
    Layout,
    PromptRenderer,
    render_value,
)

__all__ = [
    "CHOICE_LABELS",
    "CODE_LABELS",
    "DEFAULT_LAYOUT",
    "LAYOUTS",
    "NOUL_KEYS",
    "NOUL_LABELS",
    "PREFILL",
    "SCORE_LABELS",
    "SURFACE_PREFIXES",
    "LabelCapacityError",
    "LabelScheme",
    "Layout",
    "PromptRenderer",
    "SchemeName",
    "answer_keys",
    "choice_scheme",
    "code_scheme",
    "key_scheme",
    "noul_scheme",
    "render_value",
    "scheme_for",
    "score_scheme",
    "sequence_scheme_for",
    "verify_single_token",
]
