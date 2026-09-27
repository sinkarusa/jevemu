from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytest

from jevemu.backends.base import (
    Backend,
    Capabilities,
    CapabilityError,
    ChatMessage,
    RenderedPrompt,
)
from jevemu.backends.fake import DEFAULT_CAPABILITIES, FakeBackend

PROMPT = RenderedPrompt(
    messages=(ChatMessage("user", "Pick one."), ChatMessage("assistant", "Answer:")),
    template_id="t@1",
)
TABLE = {" A": math.log(0.1), " B": math.log(0.6), " C": math.log(0.2), " Z": math.log(0.1)}


def caps(**changes: Any) -> Capabilities:
    return dataclasses.replace(DEFAULT_CAPABILITIES, **changes)


async def test_top_sorted_and_truncated_to_top_k() -> None:
    backend: Backend = FakeBackend(next_token=TABLE)
    dist = await backend.next_token_logprobs(PROMPT, allowed=None, top_k=2)
    assert [t.token for t in dist.top] == [" B", " C"]
    assert dist.sampled.token == " B"
    assert not dist.constrained


async def test_top_k_clamped_to_capability_cap() -> None:
    backend = FakeBackend(capabilities=caps(top_logprobs_max=3), next_token=TABLE)
    dist = await backend.next_token_logprobs(PROMPT, allowed=None, top_k=10)
    assert [t.token for t in dist.top] == [" B", " C", " A"]


async def test_allowed_with_mask_reflected_renormalizes_and_floors_masked_tokens() -> None:
    backend = FakeBackend(
        capabilities=caps(mask_reflected_in_logprobs=True),
        next_token=TABLE,
    )
    dist = await backend.next_token_logprobs(PROMPT, allowed=[" A", " C"], top_k=5)
    assert dist.constrained
    # As vLLM 0.30.0: allowed tokens renormalized first, masked tokens fill the rest at -inf.
    assert [t.token for t in dist.top] == [" C", " A", " B", " Z"]
    finite = [t for t in dist.top if math.isfinite(t.logprob)]
    assert [t.token for t in finite] == [" C", " A"]
    assert math.fsum(math.exp(t.logprob) for t in finite) == pytest.approx(1.0)
    assert [math.exp(t.logprob) for t in finite] == pytest.approx([2 / 3, 1 / 3])
    assert [t.logprob for t in dist.top[2:]] == [-math.inf, -math.inf]
    assert dist.sampled.token == " C"
    assert math.exp(dist.sampled.logprob) == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("top_k", "cap", "expected"),
    [(3, 64, [" C", " A", " B"]), (5, 1, [" C"]), (0, 64, [])],
)
async def test_masked_top_truncated_to_top_k_and_cap(
    top_k: int, cap: int, expected: list[str]
) -> None:
    backend = FakeBackend(
        capabilities=caps(mask_reflected_in_logprobs=True, top_logprobs_max=cap),
        next_token=TABLE,
    )
    dist = await backend.next_token_logprobs(PROMPT, allowed=[" A", " C"], top_k=top_k)
    assert [t.token for t in dist.top] == expected
    assert dist.sampled.token == " C"


async def test_allowed_without_mask_reflected_reports_raw_but_samples_allowed() -> None:
    backend = FakeBackend(next_token=TABLE)
    dist = await backend.next_token_logprobs(PROMPT, allowed=[" A", " C"], top_k=5)
    assert dist.constrained
    assert [t.token for t in dist.top] == [" B", " C", " A", " Z"]
    assert dist.top[0].logprob == pytest.approx(math.log(0.6))
    assert dist.sampled.token == " C"
    assert dist.sampled.logprob == pytest.approx(math.log(0.2))


async def test_allowed_with_no_scripted_mass_raises() -> None:
    backend = FakeBackend(next_token={" A": 0.0, " B": -math.inf})
    with pytest.raises(ValueError, match="allowed"):
        await backend.next_token_logprobs(PROMPT, allowed=[" B", " Q"], top_k=2)


async def test_callable_script_receives_prompt_and_allowed() -> None:
    def script(prompt: RenderedPrompt, allowed: object) -> dict[str, float]:
        return {prompt.prefill: 0.0} if allowed is None else {"x": 0.0}

    backend = FakeBackend(next_token=script)
    assert (await backend.next_token_logprobs(PROMPT, allowed=None, top_k=1)).sampled.token == (
        "Answer:"
    )
    assert (await backend.next_token_logprobs(PROMPT, allowed=["x"], top_k=1)).sampled.token == "x"


async def test_allowed_without_structured_choice_is_capability_error() -> None:
    backend = FakeBackend(capabilities=caps(structured_choice=False), next_token=TABLE)
    with pytest.raises(CapabilityError, match="structured_choice"):
        await backend.next_token_logprobs(PROMPT, allowed=[" A"], top_k=1)


async def test_sequence_logprobs_in_input_order() -> None:
    backend = FakeBackend(
        sequences={" yes": [(" yes", -0.5)], " no": [(" n", -1.0), ("o", -math.inf)]}
    )
    scores = await backend.sequence_logprobs(PROMPT, [" no", " yes"])
    assert [s.continuation for s in scores] == [" no", " yes"]
    assert scores[0].tokens == (" n", "o")
    assert scores[0].total == -math.inf
    assert scores[1].total == -0.5


async def test_sequence_logprobs_without_echo_is_capability_error() -> None:
    backend = FakeBackend(
        capabilities=caps(echo_prompt_logprobs=False),
        sequences={" yes": [(" yes", -0.5)]},
    )
    with pytest.raises(CapabilityError, match="echo_prompt_logprobs"):
        await backend.sequence_logprobs(PROMPT, [" yes"])


async def test_unscripted_continuation_raises() -> None:
    backend = FakeBackend(sequences={" yes": [(" yes", -0.5)]})
    with pytest.raises(ValueError, match="maybe"):
        await backend.sequence_logprobs(PROMPT, [" maybe"])


async def test_tokenizer_greedy_longest_match_round_trips() -> None:
    backend = FakeBackend(vocab=["A", "An", "Ans", "swer", "s", "w", "e", "r", ":", " "])
    ids = await backend.tokenize("Answer: A")
    assert ids == [2, 5, 6, 7, 8, 9, 0]
    assert await backend.detokenize(ids) == "Answer: A"


async def test_tokenizer_unknown_char_raises() -> None:
    backend = FakeBackend(vocab=["a"])
    with pytest.raises(ValueError, match="'b' at offset 1"):
        await backend.tokenize("ab")
    with pytest.raises(ValueError, match="outside vocab"):
        await backend.detokenize([1])


async def test_token_ids_come_from_vocab() -> None:
    backend = FakeBackend(next_token={" B": -0.1, "?": -3.0}, vocab=["x", " B"])
    dist = await backend.next_token_logprobs(PROMPT, allowed=None, top_k=2)
    assert [(t.token, t.token_id) for t in dist.top] == [(" B", 1), ("?", None)]


async def test_calls_are_logged() -> None:
    backend = FakeBackend(next_token=TABLE)
    await backend.next_token_logprobs(PROMPT, allowed=[" A"], top_k=3)
    await backend.health()
    assert backend.calls == [
        ("next_token_logprobs", {"prompt": PROMPT, "allowed": (" A",), "top_k": 3}),
        ("health", {}),
    ]
