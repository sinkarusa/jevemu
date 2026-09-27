"""Pure log-space helpers shared by the scoring strategies.

A first-token call returns token logprobs; these helpers turn them into label logprobs:

- **Surface matching.** A top-k entry counts for a label when its decoded text is one of the
  label's surfaces (``"A"`` or ``" A"``). Entries at ``-inf`` are not observations: vLLM
  floors masked tokens to -9999 (mapped to ``-inf``) and still lists them in the top-k.
- **Variant merging.** A label's logprob is the logsumexp over its observed surfaces.
- **Renormalization.** ``p(label) = exp(l - logsumexp(l over valid labels))``.
- **Observed mass.** The summed valid-label probability before renormalization. Below
  :data:`LOW_MASS_THRESHOLD` under raw logprobs the model wanted to write something else.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from jevemu.backends.base import SeqScore, TokenLogprob

__all__ = [
    "LOW_MASS_THRESHOLD",
    "logsumexp",
    "low_mass_warning",
    "merge_variants",
    "observed_mass",
    "observed_surfaces",
    "renormalize",
    "sequence_logprob",
]

LOW_MASS_THRESHOLD = 0.5
"""Observed valid-label mass below this (under raw logprobs) triggers a warning."""


def logsumexp(values: Iterable[float]) -> float:
    """``log(sum(exp(v)))`` computed stably; ``-inf`` for an empty or all ``-inf`` input.

    Raises ``ValueError`` on NaN or ``+inf``: neither is a logprob.
    """
    items = list(values)
    for value in items:
        if math.isnan(value) or value == math.inf:
            raise ValueError(f"not a logprob: {value!r}")
    peak = max(items, default=-math.inf)
    if peak == -math.inf:
        return -math.inf
    return peak + math.log(math.fsum(math.exp(v - peak) for v in items))


def observed_surfaces(
    top: Sequence[TokenLogprob], surfaces: Mapping[str, str] | Iterable[str]
) -> dict[str, float]:
    """Logprob of each surface present in ``top`` with a finite logprob.

    ``surfaces`` holds the surface texts to look for (a mapping's keys are used). Duplicate
    entries for one text (distinct token ids decoding alike) are distinct tokens, so their
    probabilities add.
    """
    wanted = set(surfaces)
    found: dict[str, list[float]] = {}
    for entry in top:
        if entry.token in wanted and entry.logprob > -math.inf:
            found.setdefault(entry.token, []).append(entry.logprob)
    return {surface: logsumexp(lps) for surface, lps in found.items()}


def merge_variants(
    surface_logprobs: Mapping[str, float], name_of_surface: Mapping[str, str]
) -> dict[str, float]:
    """Label logprob = logsumexp over its observed surfaces (``"A"`` and ``" A"`` -> ``"A"``).

    Labels with no observed surface are absent from the result, never zero.
    """
    grouped: dict[str, list[float]] = {}
    for surface, logprob in surface_logprobs.items():
        grouped.setdefault(name_of_surface[surface], []).append(logprob)
    return {name: logsumexp(lps) for name, lps in grouped.items()}


def renormalize(logprobs: Sequence[float]) -> tuple[float, ...]:
    """Log-probabilities over the valid labels: ``l - logsumexp(l)``; ``-inf`` stays ``-inf``.

    Scores at ``+inf`` (an unbounded PMI) take all the mass, shared equally. Raises
    ``ValueError`` when no label has any probability or on NaN.
    """
    if any(math.isnan(lp) for lp in logprobs):
        raise ValueError("cannot renormalize NaN scores")
    infinite = [lp == math.inf for lp in logprobs]
    if any(infinite):
        share = -math.log(sum(infinite))
        return tuple(share if inf else -math.inf for inf in infinite)
    total = logsumexp(logprobs)
    if total == -math.inf:
        raise ValueError("no valid label has any probability; cannot renormalize")
    return tuple(lp - total for lp in logprobs)


def observed_mass(logprobs: Iterable[float]) -> float:
    """Summed probability of the given (observed, valid-label) logprobs, capped at 1.

    The cap only matters for echo scores of options that are prefixes of each other, whose
    sequence probabilities overlap.
    """
    return min(1.0, math.exp(logsumexp(logprobs)))


def low_mass_warning(
    mass: float, logprobs_mode: Literal["raw_logprobs", "processed_logprobs"]
) -> str | None:
    """The design's warning when valid labels got little mass under raw logprobs, else None.

    Processed logprobs are already reshaped by sampling parameters, so their mass is not a
    signal.
    """
    if logprobs_mode != "raw_logprobs" or mass >= LOW_MASS_THRESHOLD:
        return None
    return (
        f"observed valid-label mass {mass:.3f} < {LOW_MASS_THRESHOLD}: "
        "the model wanted to write something else"
    )


def sequence_logprob(score: SeqScore) -> float:
    """Total logprob of an echo-scored continuation; ``-inf`` if any token is floored.

    Rejects a score with no tokens (its empty sum would read as probability 1) and NaN or
    positive-infinite token logprobs.
    """
    if not score.token_logprobs:
        raise ValueError(
            f"echo returned no tokens for continuation {score.continuation!r}; "
            "an empty score is not a probability"
        )
    for value in score.token_logprobs:
        if math.isnan(value) or value == math.inf:
            raise ValueError(f"echo returned logprob {value!r} for {score.continuation!r}")
    return score.total
