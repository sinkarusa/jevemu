"""Shared builders for the scoring unit tests (imported by module name; not a test file)."""

from __future__ import annotations

import dataclasses
import math
import string
from typing import Any

from jevemu.backends.base import Capabilities
from jevemu.backends.fake import DEFAULT_CAPABILITIES
from jevemu.render import CHOICE_LABELS, PromptRenderer
from jevemu.types import ChoiceQuestion, NoulQuestion, ScoreQuestion

RENDERER = PromptRenderer("question_first")
STATE = "I bought a banana at the market."

LETTER_VOCAB: list[str] = [
    *CHOICE_LABELS,
    *(" " + label for label in CHOICE_LABELS),
    "Yes",
    " Yes",
    "No",
    " No",
    *string.digits,
    " ",
    " **",
    "\n",
]
"""Letters and Yes/No are one token bare and spaced; digits only bare (``" 7"`` is two)."""

VLLM_LIKE = dataclasses.replace(DEFAULT_CAPABILITIES, mask_reflected_in_logprobs=True)
"""Measured vLLM 0.30.0 behavior: the grammar mask is reflected in the logprobs."""


def caps(base: Capabilities = VLLM_LIKE, **changes: Any) -> Capabilities:
    return dataclasses.replace(base, **changes)


def lp(p: float) -> float:
    return math.log(p) if p > 0 else -math.inf


def choice(n: int) -> ChoiceQuestion:
    return ChoiceQuestion(instructions="Pick one.", criteria={f"opt-{i}": None for i in range(n)})


def score(n: int) -> ScoreQuestion:
    return ScoreQuestion(instructions="Rate it.", criteria=[f"level {i}" for i in range(n)])


NOUL = NoulQuestion(instructions="Is it fruit?")
