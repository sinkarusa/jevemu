"""Answer label schemes: the label the model writes for each option, and its token surfaces.

Choice options get letters (``A``..``Z`` then ``a``..``z``), Score levels get digits (``0``..``9``)
and Noul gets ``Yes``/``No``. Each label maps back to the key the Jev answer is reported under:
the option key for Choice, the level index as a string (``"0"``..``"K-1"``) for Score, and
``"true"``/``"false"`` for Noul. A label can be generated bare (``"A"``) or space-prefixed
(``" A"``); scoring merges both surfaces into the one label.

Strategies that score whole label texts (echo, trie) label Choice options by their own keys
(the ``"keys"`` scheme), which has no capacity limit below Jev's 255 options.

``auto_single`` labels Choice questions with more options than it reads as letters by two-capital
codes (``AA``, ``AB``, ...; the ``"codes"`` scheme), keeping the codes whose bare and
space-prefixed surfaces are each one token for the served model
(:mod:`jevemu.scoring.single_call`).
"""

from __future__ import annotations

import asyncio
import string
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from jevemu.backends.base import Backend
from jevemu.types import ChoiceQuestion, Question, ScoreQuestion

__all__ = [
    "CHOICE_LABELS",
    "CODE_LABELS",
    "NOUL_KEYS",
    "NOUL_LABELS",
    "SCORE_LABELS",
    "SURFACE_PREFIXES",
    "LabelCapacityError",
    "LabelScheme",
    "SchemeName",
    "answer_keys",
    "choice_scheme",
    "code_scheme",
    "key_scheme",
    "noul_scheme",
    "scheme_for",
    "score_scheme",
    "sequence_scheme_for",
    "verify_single_token",
]

SchemeName = Literal["letters", "digits", "yes_no", "keys", "codes"]

CHOICE_LABELS: tuple[str, ...] = tuple(string.ascii_uppercase + string.ascii_lowercase)
CODE_LABELS: tuple[str, ...] = tuple(
    first + second for first in string.ascii_uppercase for second in string.ascii_uppercase
)
"""Candidate two-capital codes in lexicographic order (``AA``, ``AB``, ..., ``ZZ``: 676)."""
SCORE_LABELS: tuple[str, ...] = tuple(string.digits)
NOUL_LABELS: tuple[str, ...] = ("Yes", "No")
NOUL_KEYS: tuple[str, ...] = ("true", "false")
"""Answer keys for ``NOUL_LABELS``, matching Noul ``criteria`` keys; P(yes) is P(``"true"``)."""


def answer_keys(question: Question) -> tuple[str, ...]:
    """The keys a question's answer is reported under, in question order: Choice option keys,
    Score levels ``"0"``..``"K-1"``, Noul :data:`NOUL_KEYS`."""
    if isinstance(question, ChoiceQuestion):
        return tuple(question.criteria)
    if isinstance(question, ScoreQuestion):
        return tuple(str(i) for i in range(len(question.criteria)))
    return NOUL_KEYS


SURFACE_PREFIXES: tuple[str, ...] = ("", " ")
"""Prefixes a label may be generated with after the ``Answer:`` prefill: ``"A"`` and ``" A"``."""


class LabelCapacityError(ValueError):
    """The question has more options than the label scheme has labels.

    Raised for Choice above 52 options so the strategy selector can fall back to a strategy
    that scores option texts instead of single-character labels.
    """

    def __init__(self, scheme: SchemeName, needed: int, capacity: int) -> None:
        super().__init__(
            f"{scheme!r} label scheme has {capacity} labels; the question needs {needed}"
        )
        self.scheme: SchemeName = scheme
        self.needed = needed
        self.capacity = capacity


@dataclass(frozen=True)
class LabelScheme:
    """Labels in display order, each paired with the answer key it stands for.

    ``labels[i]`` is rendered for, and scored as, ``keys[i]``. A permuted presentation (for
    debiasing) is a scheme whose ``keys`` are in the permuted order.
    """

    name: SchemeName
    labels: tuple[str, ...]
    keys: tuple[str, ...]
    _key_by_label: dict[str, str] = field(init=False, repr=False, compare=False)
    _label_by_key: dict[str, str] = field(init=False, repr=False, compare=False)
    _label_by_surface: dict[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.keys):
            raise ValueError(f"{len(self.labels)} labels for {len(self.keys)} keys")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("label scheme keys must be unique")
        by_surface = {prefix + label: label for label in self.labels for prefix in SURFACE_PREFIXES}
        if len(by_surface) != len(self.labels) * len(SURFACE_PREFIXES):
            raise ValueError("label surfaces must be unique across labels")
        object.__setattr__(self, "_key_by_label", dict(zip(self.labels, self.keys, strict=True)))
        object.__setattr__(self, "_label_by_key", dict(zip(self.keys, self.labels, strict=True)))
        object.__setattr__(self, "_label_by_surface", by_surface)

    def __len__(self) -> int:
        return len(self.labels)

    def key_of(self, label: str) -> str:
        """Answer key for ``label``; ``KeyError`` if it is not a label of this scheme."""
        return self._key_by_label[label]

    def label_of(self, key: str) -> str:
        """Label shown for answer ``key``; ``KeyError`` if the scheme has no such key."""
        return self._label_by_key[key]

    def variants(self, label: str) -> tuple[str, ...]:
        """Token surfaces that count as ``label``: bare first, then space-prefixed."""
        if label not in self._key_by_label:
            raise KeyError(label)
        return tuple(prefix + label for prefix in SURFACE_PREFIXES)

    @property
    def surfaces(self) -> tuple[str, ...]:
        """Every surface of every label, label-major: ``("A", " A", "B", " B", ...)``."""
        return tuple(self._label_by_surface)

    def label_of_surface(self, surface: str) -> str | None:
        """Label a generated token's text counts as (``" A"`` -> ``"A"``), or ``None``.

        Case-sensitive: with more than 26 options ``"a"`` is a label of its own.
        """
        return self._label_by_surface.get(surface)


def choice_scheme(options: Sequence[str]) -> LabelScheme:
    """Letters for Choice options, in the given order; raises ``LabelCapacityError`` above 52."""
    if len(options) > len(CHOICE_LABELS):
        raise LabelCapacityError("letters", len(options), len(CHOICE_LABELS))
    return LabelScheme("letters", CHOICE_LABELS[: len(options)], tuple(options))


def code_scheme(options: Sequence[str], codes: Sequence[str]) -> LabelScheme:
    """Choice options labelled by the first ``len(options)`` of ``codes``, in the given order.

    ``codes`` are the backend's usable codes in display order (the subset of
    :data:`CODE_LABELS` that is single-token for the model); raises ``LabelCapacityError`` when
    there are fewer codes than options.
    """
    if len(options) > len(codes):
        raise LabelCapacityError("codes", len(options), len(codes))
    return LabelScheme("codes", tuple(codes[: len(options)]), tuple(options))


def score_scheme(n_levels: int) -> LabelScheme:
    """Digits ``0``..``n_levels-1`` for Score levels; raises ``LabelCapacityError`` above 10."""
    if n_levels > len(SCORE_LABELS):
        raise LabelCapacityError("digits", n_levels, len(SCORE_LABELS))
    return LabelScheme(
        "digits", SCORE_LABELS[:n_levels], tuple(str(level) for level in range(n_levels))
    )


def noul_scheme() -> LabelScheme:
    """``Yes`` for key ``"true"``, ``No`` for key ``"false"``."""
    return LabelScheme("yes_no", NOUL_LABELS, NOUL_KEYS)


def key_scheme(options: Sequence[str]) -> LabelScheme:
    """Choice options labelled by their own keys, in the given order.

    For strategies that score whole label texts (echo, trie): the prompt lists the option keys
    without letters and the model is scored on writing a key. Any option count Jev accepts
    (up to 255) fits.
    """
    return LabelScheme("keys", tuple(options), tuple(options))


def scheme_for(question: Question) -> LabelScheme:
    """The default label scheme for ``question``, in the question's option order."""
    if isinstance(question, ChoiceQuestion):
        return choice_scheme(tuple(question.criteria))
    if isinstance(question, ScoreQuestion):
        return score_scheme(len(question.criteria))
    return noul_scheme()


def sequence_scheme_for(question: Question) -> LabelScheme:
    """The label scheme for strategies that score label texts: option keys for Choice, else
    the same digits / ``Yes``-``No`` scheme as :func:`scheme_for`."""
    if isinstance(question, ChoiceQuestion):
        return key_scheme(tuple(question.criteria))
    return scheme_for(question)


async def verify_single_token(
    backend: Backend, labels: LabelScheme | Sequence[str]
) -> dict[str, bool]:
    """Whether each surface of each label (``"A"``, ``" A"``, ...) is one token for ``backend``.

    Run at warm-up: first-token strategies can only read labels whose surfaces are single
    tokens. Tokens the tokenizer adds to every input (BOS/EOS) are measured on the empty
    string and discounted, so the answer does not depend on whether the backend's
    ``tokenize`` adds special tokens.
    """
    if isinstance(labels, str):
        raise TypeError("labels must be a sequence of labels, not a single string")
    names = labels.labels if isinstance(labels, LabelScheme) else tuple(labels)
    surfaces = [prefix + label for label in names for prefix in SURFACE_PREFIXES]
    baseline, *encoded = await asyncio.gather(
        backend.tokenize(""), *(backend.tokenize(surface) for surface in surfaces)
    )
    return {
        surface: len(ids) - len(baseline) == 1
        for surface, ids in zip(surfaces, encoded, strict=True)
    }
