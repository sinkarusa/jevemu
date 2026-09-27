"""Debiasing contract and the position maths every debiaser shares.

A debiaser runs after scoring and before calibration, on the emulator only (Jev is a black
box). It removes the model's preference for answer *positions* (the first option, the letter
``A``, the digit ``0``, ``Yes``) that does not depend on the content being judged.

**Positions.** A question's positions are its answer keys in question order
(:func:`~jevemu.render.answer_keys`): Choice option keys, Score levels, Noul ``true``/``false``.
A permuted *presentation* of a Choice question shows the same options in another order;
``order[i]`` is the index of the original option shown at position ``i``. For letter-keyed
questions (keys ``A``, ``B``, ... in order) the keys stay put and the option descriptions move,
so position ``i`` always carries label ``keys[i]``; for any other keys the options are reordered
and keep their names. Either way the model's distribution over positions maps back to the
original options.

**Fixed options.** Options whose description is one of ``fixed_texts`` (default: the question
banks' ``"I don't know"``) never move and are never divided by a prior: their probability
reflects content (abstaining), not position. Every other position is *free*.

**Label priors.** Every debiaser but ``permutation`` divides the scored distribution by a
prior over the free positions and renormalizes: ``log p' = log p - log prior`` on free
positions, with the log prior centered (geometric mean 1) so fixed options keep their weight
relative to the free block. A prior applies to every question with the same :func:`prior_key`
(question signature plus fixed positions). :class:`LabelPriors` stores fitted priors as JSON,
so a prior estimated once (offline, from a recorded run) debiases later requests at no cost.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from jevemu.calibrate import question_signature
from jevemu.eval.bank import IDK_TEXT
from jevemu.render import CHOICE_LABELS, answer_keys
from jevemu.scoring import ScoreResult
from jevemu.types import ChoiceQuestion, Description, JSONish, Question

__all__ = [
    "DEFAULT_FIXED_TEXTS",
    "PRIORS_FORMAT",
    "PRIOR_FLOOR",
    "DebiasOutcome",
    "Debiaser",
    "LabelPriors",
    "Presentation",
    "Scorer",
    "apply_prior",
    "average_presentations",
    "cyclic_orders",
    "free_positions",
    "full_cycle",
    "item_prior",
    "log_normalize",
    "mean_prior",
    "passthrough",
    "permuted_question",
    "prior_by_key",
    "prior_key",
    "score_orders",
]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

DEFAULT_FIXED_TEXTS: tuple[str, ...] = (IDK_TEXT,)
"""Option descriptions that stay in place and are never debiased."""

PRIOR_FLOOR = 1e-6
"""Smallest prior probability divided out, so a position with no mass cannot blow up."""

PRIORS_FORMAT = "jevemu.label_priors"
PRIORS_VERSION = 1

Scorer = Callable[[JSONish, Question], Awaitable[ScoreResult]]
"""Scores (state, question) with the emulator's backend, renderer and strategy."""


@dataclass(frozen=True)
class Presentation:
    """One scored presentation of a question."""

    order: tuple[int, ...]
    """``order[i]``: index of the original option shown at position ``i``."""
    logprobs: tuple[float, ...]
    """Log-probabilities by position (the presented question's key order), normalized."""

    def by_option(self) -> FloatArray:
        """Log-probabilities by original option index."""
        out = np.empty(len(self.order), dtype=np.float64)
        out[list(self.order)] = self.logprobs
        return out


@dataclass(frozen=True)
class DebiasOutcome:
    """A debiaser's result for one question."""

    logprobs: tuple[float, ...]
    """Debiased log-probabilities in the question's key order, normalized."""
    extra_calls: int = 0
    """Backend calls beyond the base scoring call(s)."""
    extra_prompt_tokens: int = 0
    presentations: tuple[Presentation, ...] = ()
    """Every presentation scored, the question's own order first; empty if only that one."""
    prior: Mapping[str, float] | None = None
    """The prior divided out, by key of the free positions; ``None`` if none was applied."""
    warnings: tuple[str, ...] = ()


class Debiaser(Protocol):
    name: str
    """Registry name (``permutation``, ``pride``, ``contextual``, ``batch``)."""

    @property
    def id(self) -> str:
        """Name plus every setting that changes answers, e.g. ``"pride(alpha=0.05,seed=0)"``."""
        ...

    async def debias(
        self, state: JSONish, question: Question, base: ScoreResult, score: Scorer
    ) -> DebiasOutcome:
        """Debias ``base`` (``question`` scored in its own order); ``score`` scores variants."""
        ...


# --- positions ---------------------------------------------------------------------------------


def free_positions(
    question: Question, fixed_texts: Sequence[str] = DEFAULT_FIXED_TEXTS
) -> tuple[bool, ...]:
    """Per position: whether it may move and be debiased (Choice options not in
    ``fixed_texts``; every Score level and Noul key)."""
    if not isinstance(question, ChoiceQuestion):
        return tuple(True for _ in answer_keys(question))
    return tuple(not _is_fixed(text, fixed_texts) for text in question.criteria.values())


def _is_fixed(text: Description, fixed_texts: Sequence[str]) -> bool:
    return isinstance(text, str) and text in fixed_texts


def prior_key(question: Question, free: Sequence[bool]) -> str:
    """The questions one prior applies to: the question signature, plus the fixed positions
    (``"|fixed=4"``) when there are any."""
    fixed = [str(i) for i, is_free in enumerate(free) if not is_free]
    signature = question_signature(question)
    return f"{signature}|fixed={','.join(fixed)}" if fixed else signature


def _letter_keyed(question: ChoiceQuestion) -> bool:
    keys = tuple(question.criteria)
    return keys == CHOICE_LABELS[: len(keys)]


def permuted_question(question: ChoiceQuestion, order: Sequence[int]) -> ChoiceQuestion:
    """``question`` with original option ``order[i]`` shown at position ``i`` (module
    docstring: letter keys keep their place and descriptions move; other keys move)."""
    keys = list(question.criteria)
    if sorted(order) != list(range(len(keys))):
        raise ValueError(f"order {list(order)} is not a permutation of {len(keys)} options")
    texts = list(question.criteria.values())
    if _letter_keyed(question):
        criteria = {key: texts[order[i]] for i, key in enumerate(keys)}
    else:
        criteria = {keys[j]: texts[j] for j in order}
    return question.model_copy(update={"criteria": criteria})


def cyclic_orders(free: Sequence[bool], n_shifts: int | None = None) -> list[tuple[int, ...]]:
    """Cyclic shifts of the free positions (fixed ones stay), the identity first.

    All ``F`` shifts by default, so every free option visits every free position once;
    ``n_shifts`` evenly spaced shifts (rounded) otherwise.
    """
    slots = [i for i, is_free in enumerate(free) if is_free]
    count = len(slots)
    if count < 2:
        return [tuple(range(len(free)))]
    wanted = count if n_shifts is None else max(1, min(n_shifts, count))
    shifts = sorted({round(j * count / wanted) % count for j in range(wanted)})
    orders = []
    for shift in shifts:
        order = list(range(len(free)))
        for j, slot in enumerate(slots):
            order[slot] = slots[(j + shift) % count]
        orders.append(tuple(order))
    return orders


# --- log-space maths ---------------------------------------------------------------------------


def log_normalize(z: FloatArray) -> FloatArray:
    """``z - logsumexp(z)``; ``-inf`` entries stay ``-inf``."""
    peak = float(np.max(z))
    if not math.isfinite(peak):
        raise ValueError("cannot normalize a row without a finite entry")
    out: FloatArray = z - (peak + math.log(float(np.exp(z - peak).sum())))
    return out


def apply_prior(logp: ArrayLike, prior: Sequence[float], free: Sequence[bool]) -> FloatArray:
    """Divide the free positions of ``logp`` by ``prior`` (one entry per free position) and
    renormalize; the log prior is centered over the free positions (module docstring)."""
    mask = np.asarray(free, dtype=np.bool_)
    values = np.asarray(prior, dtype=np.float64)
    if values.shape != (int(mask.sum()),):
        raise ValueError(f"prior has {values.shape[0]} entries for {int(mask.sum())} positions")
    log_prior = np.log(np.maximum(values, PRIOR_FLOOR))
    shift = np.zeros(mask.shape[0], dtype=np.float64)
    shift[mask] = log_prior - log_prior.mean()
    return log_normalize(np.asarray(logp, dtype=np.float64) - shift)


def average_presentations(presentations: Sequence[Presentation]) -> FloatArray:
    """Log of the mean, over presentations, of the probabilities of each original option."""
    probs = np.mean([np.exp(p.by_option()) for p in presentations], axis=0)
    with np.errstate(divide="ignore"):
        return log_normalize(np.log(probs))


def item_prior(presentations: Sequence[Presentation], free: Sequence[bool]) -> FloatArray:
    """PriDe's per-question prior over the free positions from all its cyclic shifts.

    With ``P(position i | shift s) ∝ prior_i * P(content at i under s)``, averaging
    ``log P(position i | shift s)`` over shifts that move every free option through every free
    position leaves ``log prior_i`` plus a constant: the prior is the softmax of that mean.
    Positions of ``-inf`` are floored at :data:`PRIOR_FLOOR` first. Raises unless
    :func:`full_cycle` holds.
    """
    if not full_cycle(presentations, free):
        raise ValueError("item_prior needs every cyclic shift of the free options")
    mask = np.asarray(free, dtype=np.bool_)
    rows = np.array([p.logprobs for p in presentations], dtype=np.float64)
    mean_log = np.maximum(rows[:, mask], math.log(PRIOR_FLOOR)).mean(axis=0)
    return np.exp(log_normalize(mean_log))


def full_cycle(presentations: Sequence[Presentation], free: Sequence[bool]) -> bool:
    """Whether ``presentations`` show every free option at every free position exactly once
    (all ``F`` cyclic shifts, as :func:`item_prior` needs); fixed positions never move."""
    slots = [i for i, is_free in enumerate(free) if is_free]
    if len(slots) < 2 or len(presentations) != len(slots):
        return False
    for position in range(len(free)):
        shown = sorted(p.order[position] for p in presentations)
        expected = slots if position in slots else [position] * len(slots)
        if shown != expected:
            return False
    return True


def mean_prior(priors: Iterable[Sequence[float]]) -> FloatArray:
    """Renormalized mean of priors (or of probability vectors restricted to free positions)."""
    stacked = np.array(list(priors), dtype=np.float64)
    if stacked.ndim != 2 or stacked.shape[0] == 0:
        raise ValueError("mean_prior needs at least one prior")
    mean = stacked.mean(axis=0)
    total = float(mean.sum())
    if total <= 0.0:
        raise ValueError("priors have no mass")
    out: FloatArray = mean / total
    return out


# --- stored priors -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelPriors:
    """Fitted label priors by :func:`prior_key`, with the number of questions behind each."""

    method: str
    """How they were estimated (``"pride"`` or ``"batch"``)."""
    priors: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    """Prior key -> prior over the free positions (sums to 1)."""
    counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.priors) != set(self.counts):
            raise ValueError("priors and counts must have the same keys")
        for key, prior in self.priors.items():
            values = np.asarray(prior, dtype=np.float64)
            if values.ndim != 1 or values.shape[0] < 1 or not np.isfinite(values).all():
                raise ValueError(f"{key}: a prior must be a non-empty list of finite numbers")
            if (values < 0.0).any() or not math.isclose(float(values.sum()), 1.0, abs_tol=1e-6):
                raise ValueError(f"{key}: a prior must be non-negative and sum to 1")

    def to_json(self) -> dict[str, Any]:
        return {
            "format": PRIORS_FORMAT,
            "version": PRIORS_VERSION,
            "method": self.method,
            "priors": [
                {"key": key, "count": self.counts[key], "prior": list(self.priors[key])}
                for key in sorted(self.priors)
            ],
        }

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> LabelPriors:
        if d.get("format") != PRIORS_FORMAT or d.get("version") != PRIORS_VERSION:
            raise ValueError(
                f"not a {PRIORS_FORMAT} v{PRIORS_VERSION} document: "
                f"format={d.get('format')!r}, version={d.get('version')!r}"
            )
        entries = d["priors"]
        return cls(
            method=str(d["method"]),
            priors={e["key"]: tuple(float(v) for v in e["prior"]) for e in entries},
            counts={e["key"]: int(e["count"]) for e in entries},
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        text = json.dumps(self.to_json(), indent=2, allow_nan=False)
        Path(path).write_text(text + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> LabelPriors:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def digest(self) -> str:
        """First 12 hex characters of the SHA-256 of the canonical JSON (for ids)."""
        text = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def prior_by_key(
    question: Question, free: Sequence[bool], prior: Sequence[float]
) -> dict[str, float]:
    """``prior`` (free positions) keyed by the question's keys, for diagnostics."""
    keys = [key for key, is_free in zip(answer_keys(question), free, strict=True) if is_free]
    return {key: float(value) for key, value in zip(keys, prior, strict=True)}


def passthrough(base: ScoreResult, *warnings: str) -> DebiasOutcome:
    """``base`` unchanged (the debiaser does not apply to this question)."""
    return DebiasOutcome(logprobs=base.logprobs, warnings=warnings)


async def score_orders(
    state: JSONish,
    question: ChoiceQuestion,
    orders: Sequence[tuple[int, ...]],
    base: ScoreResult,
    score: Scorer,
) -> tuple[list[Presentation], int, int]:
    """Score every presentation in ``orders`` (the first must be the identity, which reuses
    ``base``): the presentations, and the extra backend calls and prompt tokens spent."""
    identity = tuple(range(len(question.criteria)))
    if not orders or orders[0] != identity:
        raise ValueError("the first order must be the identity")
    results = await asyncio.gather(
        *(score(state, permuted_question(question, order)) for order in orders[1:])
    )
    presentations = [Presentation(identity, base.logprobs)]
    presentations += [
        Presentation(order, r.logprobs) for order, r in zip(orders[1:], results, strict=True)
    ]
    calls = sum(r.n_backend_calls for r in results)
    tokens = sum(r.prompt_tokens for r in results)
    return presentations, calls, tokens
