"""Strategy selection (``auto``) from the design's table."""

from __future__ import annotations

import dataclasses
import math

import pytest
from scoring_helpers import LETTER_VOCAB, NOUL, RENDERER, STATE, VLLM_LIKE, caps, choice, lp, score

from jevemu.backends.base import Capabilities, CapabilityError, RenderedPrompt
from jevemu.backends.fake import FakeBackend
from jevemu.scoring import AutoStrategy, select, select_strategy
from jevemu.types import Question

NO_ECHO = caps(echo_prompt_logprobs=False)


@pytest.mark.parametrize(
    ("capabilities", "question", "strategy"),
    [
        (VLLM_LIKE, choice(32), "constrained"),  # 32 x 2 = 64 <= cap
        (VLLM_LIKE, choice(33), "echo_sum"),  # 66 > 64
        (VLLM_LIKE, choice(60), "echo_sum"),  # no letters beyond 52
        (VLLM_LIKE, choice(255), "echo_sum"),
        (NO_ECHO, choice(33), "trie"),
        (caps(structured_choice=False), choice(4), "first_token"),
        (VLLM_LIKE, NOUL, "constrained"),
        (VLLM_LIKE, score(10), "constrained"),
        (caps(top_logprobs_max=20), score(10), "constrained"),  # vLLM's unprobed default cap
        (caps(top_logprobs_max=19), score(10), "echo_sum"),
    ],
)
def test_selection_table(capabilities: Capabilities, question: Question, strategy: str) -> None:
    assert select_strategy(capabilities, question).name == strategy


def test_default_path_records_no_fallback() -> None:
    assert select(VLLM_LIKE, choice(4)).fallbacks == ()


def test_overflow_fallbacks_are_logged_in_table_order() -> None:
    fallbacks = select(NO_ECHO, choice(40)).fallbacks
    assert "40 labels x 2 variants = 80 > top_logprobs_max 64" in fallbacks[0]
    assert "S5 skipped" in fallbacks[1]
    assert "echo unavailable" in fallbacks[2]


def test_advertised_explicit_tokens_are_skipped_until_implemented() -> None:
    selection = select(caps(explicit_token_logprobs=True), choice(60))
    assert selection.strategy.name == "echo_sum"
    assert any("S5" in f and "not implemented" in f for f in selection.fallbacks)


def test_structured_choice_fallback_is_logged() -> None:
    (fallback,) = select(caps(structured_choice=False), NOUL).fallbacks
    assert "S1" in fallback


HOSTED_API = caps(
    top_logprobs_max=20,
    structured_choice=True,
    mask_reflected_in_logprobs=True,
    echo_prompt_logprobs=False,
    assistant_prefill=False,
    local_tokenizer=False,
)
"""A chat API without prefill, echo or tokenizer, with Structured Outputs (OpenRouter, top 20)."""


@pytest.mark.parametrize(
    "question", [choice(2), choice(20), score(10), NOUL], ids=["2", "20", "score10", "noul"]
)
@pytest.mark.parametrize(
    ("structured", "strategy"), [(True, "constrained"), (False, "first_token")]
)
def test_no_prefill_reads_a_new_turn_structured_when_the_api_can(
    question: Question, structured: bool, strategy: str
) -> None:
    capabilities = dataclasses.replace(HOSTED_API, structured_choice=structured)
    assert select(capabilities, question).strategy.name == strategy
    assert AutoStrategy().supports(capabilities, question)


@pytest.mark.parametrize("n", [21, 60])
def test_no_prefill_without_room_in_the_top_k_has_no_strategy(n: int) -> None:
    with pytest.raises(CapabilityError, match="assistant_prefill"):
        select(HOSTED_API, choice(n))
    assert not AutoStrategy().supports(HOSTED_API, choice(n))


async def test_auto_scores_with_the_selected_strategy_and_logs_why() -> None:
    keys = [f"opt-{i}" for i in range(60)]

    def table(prompt: RenderedPrompt, continuation: str) -> list[tuple[str, float]]:
        return [(continuation, lp(0.9) if continuation == " opt-7" else lp(0.001))]

    backend = FakeBackend(
        capabilities=VLLM_LIKE,
        sequences=table,
        vocab=[chr(c) for c in range(32, 127)] + ["\n"],
    )
    result = await AutoStrategy().score(backend, RENDERER, STATE, choice(60))
    assert result.strategy == "echo_sum"
    assert result.keys == tuple(keys)
    assert max(result.probabilities, key=result.probabilities.__getitem__) == "opt-7"
    assert [w for w in result.warnings if w.startswith("auto: ")] == [
        "auto: 60 options exceed the 52 letter labels: first-token scoring cannot see every label",
        "auto: explicit-token logprobs unavailable: S5 skipped",
        "auto: using S4 echo",
    ]


async def test_auto_default_path_is_exact_constrained_scoring() -> None:
    backend = FakeBackend(
        capabilities=VLLM_LIKE,
        next_token={" A": lp(0.7), " B": lp(0.2), " **": lp(0.1)},
        vocab=LETTER_VOCAB,
    )
    result = await AutoStrategy().score(backend, RENDERER, STATE, choice(2))
    assert result.strategy == "constrained"
    assert [math.exp(v) for v in result.logprobs] == pytest.approx([7 / 9, 2 / 9])
    assert result.warnings == ()
