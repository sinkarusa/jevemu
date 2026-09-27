from __future__ import annotations

import math

import pytest
from scoring_helpers import caps, lp

from jevemu.backends.base import ChatMessage, NextTokenDist, RenderedPrompt, TokenLogprob
from jevemu.backends.fake import FakeBackend
from jevemu.scoring.missing_mass import fill_missing

PROMPT = RenderedPrompt(
    messages=(ChatMessage("user", "Pick."), ChatMessage("assistant", "Answer:")),
    template_id="t",
)


def dist(*entries: tuple[str, float], constrained: bool = True) -> NextTokenDist:
    top = tuple(TokenLogprob(token, None, value) for token, value in entries)
    return NextTokenDist(top=top, sampled=top[0], constrained=constrained, prompt_tokens=10)


async def test_echo_that_zeroes_the_anchor_falls_back_to_the_smallest_entry_bound() -> None:
    # a vLLM-shaped constrained read (prefill, mask reflected) keeps the smallest-entry bound
    backend = FakeBackend(
        capabilities=caps(),
        sequences={" C": [(" C", lp(0.1))], " A": [(" A", -math.inf)]},
    )
    first = dist((" A", lp(0.7)), (" B", lp(0.3)), (" x", -math.inf))
    fill = await fill_missing(
        backend,
        PROMPT,
        first,
        {"C": [" C"]},
        observed={" A": lp(0.7), " B": lp(0.3)},
        allowed=(" A", " B", " C"),
    )
    assert fill.policy == "upper_bound"
    assert fill.logprobs == {"C": pytest.approx(lp(0.3))}
    assert fill.truncated
    assert any("cannot be rescaled" in w for w in fill.warnings)


async def test_echo_prompt_tokens_count_every_recomputed_prefix() -> None:
    backend = FakeBackend(
        capabilities=caps(mask_reflected_in_logprobs=False),
        sequences={" C": [(" ", lp(0.5)), ("C", lp(0.2))], "C": [("C", lp(0.01))]},
    )
    first = dist((" A", lp(0.7)), constrained=False)
    fill = await fill_missing(backend, PROMPT, first, {"C": ["C", " C"]}, observed={})
    assert fill.logprobs["C"] == pytest.approx(lp(0.11))
    assert (fill.n_backend_calls, fill.prompt_tokens) == (2, 10 + 1 + 10 + 2)


async def test_a_missing_label_needs_something_to_score() -> None:
    with pytest.raises(ValueError, match="no surfaces"):
        await fill_missing(FakeBackend(), PROMPT, dist((" A", 0.0)), {"C": []}, observed={})
