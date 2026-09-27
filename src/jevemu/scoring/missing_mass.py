"""Missing-label policy: a valid label absent from a first-token top-k never becomes a silent zero.

In the design's order of preference:

1. **Explicit-token logprobs.** Needs ``Capabilities.explicit_token_logprobs``. The ``Backend``
   protocol has no explicit-token method yet, so this step is always skipped; a
   backend advertising the capability gets a warning saying so.
2. **Echo.** With ``Capabilities.echo_prompt_logprobs``, echo-score only the missing labels'
   surfaces as continuations of the same prompt and merge them per label. Echo logprobs are
   unconstrained. When the first-token call was constrained and the mask is reflected in its
   logprobs, the observed values are renormalized over the allowed tokens, so the likeliest
   observed surface is echo-scored too and the offset between its two values converts the
   echo scores into the constrained scale (no observed surface, or an anchor echo of ``-inf``,
   leaves no scale and falls through to step 3).
3. **Upper bound.** Otherwise each missing label gets an upper bound and the result is marked
   ``truncated``: the smallest returned logprob (a label absent from a sorted top-k list can be
   no likelier than its last entry). The bound is per surface: a label with two unseen surfaces
   could hold up to twice that much. A backend whose top-k is cut by a probability threshold
   rather than a count (``Capabilities.top_logprobs_exact`` false: gpt-6-luna returned 2 of 5
   requested entries holding 0.985 of the mass) makes that bound loose; there it is capped by
   the mass the list leaves over (``1 - sum of its probabilities``), which also bounds any unseen
   token. A constrained read whose mask is reflected in its logprobs gives all the mass to
   allowed tokens, so when every missing surface was allowed the leftover is exactly what they
   share, and it alone is the bound: a structured read
   (:func:`~jevemu.backends.chat_api.parse_structured_response`) lists one entry per label that
   may sum several token paths, so its smallest entry bounds nothing.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from jevemu.backends.base import Backend, NextTokenDist, RenderedPrompt
from jevemu.scoring.aggregate import logsumexp, sequence_logprob

__all__ = ["MissingFill", "MissingPolicy", "fill_missing", "upper_bound"]

MissingPolicy = Literal["echo", "upper_bound"]


@dataclass(frozen=True)
class MissingFill:
    """Estimated logprobs for labels a first-token call did not return."""

    logprobs: dict[str, float]
    """Label -> logprob, on the same scale as the first-token call's logprobs."""
    policy: MissingPolicy | None
    """How the values were obtained; ``None`` when nothing was missing."""
    truncated: bool
    """Values are upper bounds rather than observations."""
    n_backend_calls: int
    prompt_tokens: int
    """Prompt tokens the echo requests processed (prefix + continuation each)."""
    warnings: tuple[str, ...] = ()


LEFTOVER_FLOOR = 2.0**-23
"""Smallest leftover mass the bound uses: float32 epsilon, the resolution of float32 logprobs
(OpenAI's). A top-k summing to 1 within rounding (one entry at logprob 0.0) must not bound a
missing label at probability 0."""


def upper_bound(dist: NextTokenDist, *, leftover: bool = False, masked: bool = False) -> float:
    """Smallest finite logprob the call returned (the sampled token counts too); with
    ``leftover``, at most the log of the mass the top-k leaves over (floored at
    :data:`LEFTOVER_FLOOR`); with ``masked`` (the read's mask is reflected in its logprobs),
    that log alone."""
    finite = [t.logprob for t in (*dist.top, dist.sampled) if math.isfinite(t.logprob)]
    if not finite:
        raise ValueError("the first-token call returned no finite logprob to bound missing labels")
    if not (leftover or masked):
        return min(finite)
    listed = math.fsum(math.exp(t.logprob) for t in dist.top if math.isfinite(t.logprob))
    left = math.log(max(1.0 - listed, LEFTOVER_FLOOR))
    return left if masked else min(min(finite), left)


async def fill_missing(
    backend: Backend,
    prompt: RenderedPrompt,
    dist: NextTokenDist,
    missing: Mapping[str, Sequence[str]],
    *,
    observed: Mapping[str, float],
    allowed: Collection[str] = (),
) -> MissingFill:
    """Logprobs for the ``missing`` labels of ``dist`` (read after ``prompt``).

    ``missing`` maps each missing label to the surfaces that count for it (the surfaces the
    first-token read looked for); ``observed`` maps each surface found in ``dist`` to its
    logprob and anchors echo scores when ``dist`` is on a constrained scale; ``allowed`` holds
    the surfaces a constrained ``dist`` allowed.
    """
    if not missing:
        return MissingFill({}, None, False, 0, 0)
    empty = sorted(name for name, surfaces in missing.items() if not surfaces)
    if empty:
        raise ValueError(f"missing labels {empty} have no surfaces to score")
    caps = backend.capabilities
    notes: list[str] = []
    if caps.explicit_token_logprobs:
        notes.append(
            "explicit-token logprobs advertised, but the backend protocol has no "
            "explicit-token method yet; skipped"
        )
    head = f"{sorted(missing)} missing from the top-{len(dist.top)}"
    rescale = dist.constrained and caps.mask_reflected_in_logprobs
    if not caps.echo_prompt_logprobs:
        notes.append(f"{head}: echo unavailable")
    elif rescale and not observed:
        notes.append(f"{head}: no observed label surface to rescale echo into the constraint")
    else:
        anchor = max(observed.items(), key=lambda item: item[1]) if rescale else None
        filled = await _echo(backend, prompt, dist, missing, anchor)
        if filled is not None:
            logprobs, n_calls, prompt_tokens = filled
            notes.append(f"{head}: echo-scored")
            return MissingFill(logprobs, "echo", False, n_calls, prompt_tokens, tuple(notes))
        notes.append(f"{head}: echo scored the anchor surface at -inf, so echo cannot be rescaled")
    # only a hosted structured read (no prefill) lists one entry per label; vLLM keeps the old bound
    masked = (
        rescale
        and not caps.assistant_prefill
        and all(s in allowed for surfaces in missing.values() for s in surfaces)
    )
    bound = upper_bound(dist, leftover=not caps.top_logprobs_exact, masked=masked)
    notes.append(f"{head}: assigned the upper bound {bound:.4g}")
    return MissingFill({name: bound for name in missing}, "upper_bound", True, 0, 0, tuple(notes))


async def _echo(
    backend: Backend,
    prompt: RenderedPrompt,
    dist: NextTokenDist,
    missing: Mapping[str, Sequence[str]],
    anchor: tuple[str, float] | None,
) -> tuple[dict[str, float], int, int] | None:
    """Echo-scored missing labels, or None when the anchor makes rescaling impossible."""
    surfaces = [s for name in missing for s in missing[name]]
    continuations = surfaces if anchor is None else [*surfaces, anchor[0]]
    scores = await backend.sequence_logprobs(prompt, continuations)
    totals = [sequence_logprob(score) for score in scores]
    # Each echo request recomputes the prompt (no prefix cache for prompt logprobs).
    prompt_tokens = sum(dist.prompt_tokens + len(score.tokens) for score in scores)
    offset = 0.0
    if anchor is not None:
        anchor_echo = totals.pop()
        if anchor_echo == -math.inf:
            return None
        offset = anchor_echo - anchor[1]
    by_surface = dict(zip(surfaces, totals, strict=True))
    logprobs = {
        name: logsumexp(by_surface[s] for s in surfaces_of) - offset
        for name, surfaces_of in missing.items()
    }
    return logprobs, len(continuations), prompt_tokens
