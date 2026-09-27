"""S4 echo sequence scoring on scripted continuations."""

from __future__ import annotations

import math

import pytest
from scoring_helpers import RENDERER, STATE, VLLM_LIKE, lp

from jevemu.backends.base import RenderedPrompt
from jevemu.backends.fake import FakeBackend
from jevemu.scoring import EchoStrategy
from jevemu.types import ChoiceQuestion

ITEMS = ChoiceQuestion(
    instructions="What kind of item?", criteria={"fruit": None, "tool": None, "vehicle": None}
)
VOCAB = [chr(c) for c in range(32, 127)] + ["\n"]
"""Covers any ASCII prompt text; echo tokenizes the prompt to count its tokens."""

CONDITIONAL = {
    " fruit": [(" fruit", lp(0.6))],
    " tool": [(" to", lp(0.5)), ("ol", lp(0.4))],
    " vehicle": [(" ve", lp(0.2)), ("hic", lp(0.5)), ("le", lp(0.5))],
}


def backend_for(sequences: object) -> FakeBackend:
    return FakeBackend(capabilities=VLLM_LIKE, sequences=sequences, vocab=VOCAB)  # type: ignore[arg-type]


def probs(values: tuple[float, ...]) -> list[float]:
    return [math.exp(v) for v in values]


async def test_sum_scores_whole_option_texts_after_the_gap() -> None:
    result = await EchoStrategy().score(backend_for(CONDITIONAL), RENDERER, STATE, ITEMS)
    assert result.keys == ("fruit", "tool", "vehicle")
    assert probs(result.logprobs) == pytest.approx([0.6 / 0.85, 0.2 / 0.85, 0.05 / 0.85])
    assert result.observed_mass == pytest.approx(0.85)
    assert (result.strategy, result.label_scheme) == ("echo_sum", "keys")
    assert result.n_backend_calls == 3


async def test_mean_normalizes_by_token_count() -> None:
    result = await EchoStrategy("mean").score(backend_for(CONDITIONAL), RENDERER, STATE, ITEMS)
    means = [lp(0.6), (lp(0.5) + lp(0.4)) / 2, (lp(0.2) + 2 * lp(0.5)) / 3]
    assert result.raw_logprobs == pytest.approx(tuple(means))
    total = math.fsum(math.exp(m) for m in means)
    assert probs(result.logprobs) == pytest.approx([math.exp(m) / total for m in means])


def _with_prior(prompt: RenderedPrompt, continuation: str) -> list[tuple[str, float]]:
    """The content-free state prefers ``" tool"``; the real state prefers ``" fruit"``."""
    if "N/A" in prompt.messages[-2].content:
        return [
            (continuation, {" fruit": lp(0.2), " tool": lp(0.6), " vehicle": lp(0.2)}[continuation])
        ]
    return CONDITIONAL[continuation]


async def test_pmi_subtracts_the_content_free_prior() -> None:
    result = await EchoStrategy("pmi").score(backend_for(_with_prior), RENDERER, STATE, ITEMS)
    pmi = [lp(0.6) - lp(0.2), lp(0.2) - lp(0.6), lp(0.05) - lp(0.2)]
    assert result.raw_logprobs == pytest.approx(tuple(pmi))
    total = math.fsum(math.exp(v) for v in pmi)
    assert probs(result.logprobs) == pytest.approx([math.exp(v) / total for v in pmi])
    assert result.n_backend_calls == 6
    assert result.observed_mass == pytest.approx(0.85)  # of the real state, not the prior


async def test_floored_token_zeroes_only_its_option_in_every_mode() -> None:
    floored = {**CONDITIONAL, " tool": [(" to", lp(0.5)), ("ol", -math.inf)]}
    for mode in ("sum", "mean", "pmi"):
        result = await EchoStrategy(mode).score(backend_for(floored), RENDERER, STATE, ITEMS)  # type: ignore[arg-type]
        p = probs(result.logprobs)
        assert p[1] == 0.0
        assert math.fsum(p) == pytest.approx(1.0)
        assert not any(math.isnan(v) for v in result.logprobs)


async def test_every_option_floored_is_an_error_not_a_uniform_guess() -> None:
    floored = {k: [(k, -math.inf)] for k in CONDITIONAL}
    with pytest.raises(ValueError, match="no valid label"):
        await EchoStrategy().score(backend_for(floored), RENDERER, STATE, ITEMS)


async def test_a_continuation_scored_with_no_tokens_is_rejected() -> None:
    # The adapter drops vLLM's null first prompt token; an empty score must never read as
    # probability 1.
    empty = {**CONDITIONAL, " tool": []}
    with pytest.raises(ValueError, match="no tokens"):
        await EchoStrategy().score(backend_for(empty), RENDERER, STATE, ITEMS)


async def test_pmi_option_impossible_without_the_state_takes_the_mass() -> None:
    def prior_zero(prompt: RenderedPrompt, continuation: str) -> list[tuple[str, float]]:
        if "N/A" in prompt.messages[-2].content and continuation == " vehicle":
            return [(" vehicle", -math.inf)]
        return CONDITIONAL[continuation]

    result = await EchoStrategy("pmi").score(backend_for(prior_zero), RENDERER, STATE, ITEMS)
    assert probs(result.logprobs) == pytest.approx([0.0, 0.0, 1.0])
    assert any("unbounded" in w for w in result.warnings)


async def test_255_options_are_scored_one_request_each() -> None:
    keys = [f"item {i}" for i in range(255)]
    question = ChoiceQuestion(instructions="Pick.", criteria=dict.fromkeys(keys))
    weights = {f" {k}": float(i + 1) for i, k in enumerate(keys)}
    total = math.fsum(weights.values())

    def table(prompt: RenderedPrompt, continuation: str) -> list[tuple[str, float]]:
        return [(continuation, math.log(weights[continuation] / total))]

    result = await EchoStrategy().score(backend_for(table), RENDERER, STATE, question)
    assert result.keys == tuple(keys)
    assert probs(result.logprobs) == pytest.approx([(i + 1) / total for i in range(255)])
    assert result.n_backend_calls == 255


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown echo mode"):
        EchoStrategy("max")  # type: ignore[arg-type]
