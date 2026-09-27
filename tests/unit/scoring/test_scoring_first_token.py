"""S1/S2 first-token strategies and the missing-label policy on FakeBackend."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scoring_helpers import (
    LETTER_VOCAB,
    NOUL,
    RENDERER,
    STATE,
    VLLM_LIKE,
    caps,
    choice,
    lp,
    score,
)

from jevemu.backends.base import CapabilityError, RenderedPrompt
from jevemu.backends.fake import FakeBackend
from jevemu.render import LabelCapacityError, PromptRenderer
from jevemu.scoring import ConstrainedStrategy, FirstTokenStrategy, ScoreResult
from jevemu.types import ChoiceQuestion, Question

S1 = FirstTokenStrategy()
S2 = ConstrainedStrategy()


async def run(
    strategy: FirstTokenStrategy | ConstrainedStrategy,
    backend: FakeBackend,
    question: Question,
) -> ScoreResult:
    return await strategy.score(backend, RENDERER, STATE, question)


def probs(result: ScoreResult) -> list[float]:
    return [math.exp(v) for v in result.logprobs]


async def test_constrained_is_exact_when_the_mask_is_reflected() -> None:
    table = {
        " A": lp(0.5),
        "A": lp(0.1),
        " B": lp(0.2),
        "B": lp(0.05),
        " C": lp(0.05),
        " **": lp(0.1),
    }
    backend = FakeBackend(capabilities=VLLM_LIKE, next_token=table, vocab=LETTER_VOCAB)
    result = await run(S2, backend, choice(3))
    assert result.keys == ("opt-0", "opt-1", "opt-2")
    assert probs(result) == pytest.approx([0.6 / 0.9, 0.25 / 0.9, 0.05 / 0.9], rel=1e-12)
    assert result.observed_mass == pytest.approx(1.0)
    assert (result.missing, result.truncated, result.warnings) == ((), False, ())
    assert result.n_backend_calls == 1


async def test_unconstrained_merges_variants_and_ignores_other_tokens() -> None:
    table = {" A": lp(0.3), "A": lp(0.1), " B": lp(0.2), " **": lp(0.4)}
    backend = FakeBackend(next_token=table, vocab=LETTER_VOCAB)
    result = await run(S1, backend, choice(2))
    assert probs(result) == pytest.approx([0.4 / 0.6, 0.2 / 0.6])
    assert result.raw_logprobs == pytest.approx((lp(0.4), lp(0.2)))
    assert result.observed_mass == pytest.approx(0.6)


async def test_keys_follow_question_order_not_rank() -> None:
    question = ChoiceQuestion(instructions="Pick.", criteria={"zeta": None, "alpha": None})
    backend = FakeBackend(next_token={" B": lp(0.9), " A": lp(0.1)}, vocab=LETTER_VOCAB)
    result = await run(S1, backend, question)
    assert result.keys == ("zeta", "alpha")
    assert result.probabilities == pytest.approx({"zeta": 0.1, "alpha": 0.9})


async def test_noul_reports_true_and_false() -> None:
    backend = FakeBackend(
        capabilities=VLLM_LIKE, next_token={" Yes": lp(0.8), " No": lp(0.2)}, vocab=LETTER_VOCAB
    )
    noul = await run(S2, backend, NOUL)
    assert noul.keys == ("true", "false")
    assert noul.probabilities == pytest.approx({"true": 0.8, "false": 0.2})


def _after_gap(prompt: RenderedPrompt, allowed: Sequence[str] | None) -> dict[str, float]:
    """The model writes a space first; the digit distribution lives after it."""
    if prompt.prefill == "Answer:":
        return {" ": lp(0.98), "3": lp(0.019), "1": lp(0.001)}
    assert prompt.prefill == "Answer: "
    return {"0": lp(0.001), "1": lp(0.2), "2": lp(0.5), "3": lp(0.299)}


@pytest.mark.parametrize("strategy", [S1, S2], ids=["S1", "S2"])
async def test_digits_are_read_after_the_space_they_follow(
    strategy: FirstTokenStrategy | ConstrainedStrategy,
) -> None:
    # " 7" is two tokens (" " + "7"), as on Qwen3: reading bare digits right after "Answer:"
    # would score a path the model does not take.
    backend = FakeBackend(capabilities=VLLM_LIKE, next_token=_after_gap, vocab=LETTER_VOCAB)
    result = await run(strategy, backend, score(4))
    assert result.keys == ("0", "1", "2", "3")
    assert probs(result) == pytest.approx([0.001, 0.2, 0.5, 0.299])
    assert result.observed_mass == pytest.approx(1.0)


class _SplitB(FakeBackend):
    """A tokenizer that spells the letter B with two byte tokens, bare or spaced."""

    async def tokenize(self, text: str) -> list[int]:
        return [*await super().tokenize(text.replace("B", "")), *[998, 999] * text.count("B")]


async def test_label_without_a_single_token_surface_is_scored_not_dropped() -> None:
    # Neither "B" nor " B" is one token: the constraint cannot express B, so it is echo-scored.
    backend = _SplitB(
        capabilities=caps(mask_reflected_in_logprobs=False),
        next_token={" A": lp(0.5), " C": lp(0.2), " **": lp(0.3)},
        sequences={"B": [("B", lp(0.01))], " B": [(" ", lp(0.5)), ("B", lp(0.2))]},
        vocab=LETTER_VOCAB,
    )
    result = await run(S2, backend, choice(3))
    assert result.raw_logprobs == pytest.approx((lp(0.5), lp(0.11), lp(0.2)))
    assert result.missing == ("opt-1",)
    assert not result.truncated


# --- missing labels ---------------------------------------------------------------------------

_TRUNCATING = {" A": lp(0.5), " B": lp(0.3), " **": lp(0.15), " C": lp(0.05)}
"""With a top-3 cap, C falls off the list."""


async def test_missing_label_is_echo_scored_on_the_same_scale() -> None:
    backend = FakeBackend(
        capabilities=caps(top_logprobs_max=3),
        next_token=_TRUNCATING,
        sequences={" C": [(" C", lp(0.05))], "C": [("C", -math.inf)]},
        vocab=LETTER_VOCAB,
    )
    result = await run(S1, backend, choice(3))
    assert probs(result) == pytest.approx([0.5 / 0.85, 0.3 / 0.85, 0.05 / 0.85])
    assert result.missing == ("opt-2",)
    assert not result.truncated
    assert result.n_backend_calls == 3  # first token + both surfaces of C
    assert any("echo-scored" in w for w in result.warnings)


async def test_echo_fill_is_rescaled_into_a_reflected_constraint() -> None:
    # Constrained logprobs are renormalized over the allowed tokens; echo is not. The anchor
    # surface converts echo into the constrained scale, so the result is the exact
    # constrained distribution even though C never appeared.
    table = {" A": lp(0.4), " B": lp(0.3), " C": lp(0.1), " **": lp(0.2)}
    backend = FakeBackend(
        capabilities=caps(top_logprobs_max=2),
        next_token=table,
        sequences={**{s: [(s, table[s])] for s in (" A", " C")}, "C": [("C", -math.inf)]},
        vocab=LETTER_VOCAB,
    )
    result = await run(S2, backend, choice(3))
    assert probs(result) == pytest.approx([0.5, 0.375, 0.125], rel=1e-12)
    assert result.missing == ("opt-2",)


async def test_without_echo_a_missing_label_gets_the_upper_bound() -> None:
    backend = FakeBackend(
        capabilities=caps(top_logprobs_max=3, echo_prompt_logprobs=False),
        next_token=_TRUNCATING,
        vocab=LETTER_VOCAB,
    )
    result = await run(S1, backend, choice(3))
    assert result.raw_logprobs[2] == pytest.approx(lp(0.15))  # smallest returned logprob
    assert probs(result) == pytest.approx([0.5 / 0.95, 0.3 / 0.95, 0.15 / 0.95])
    assert result.truncated
    assert result.missing == ("opt-2",)
    assert any("upper bound" in w for w in result.warnings)


async def test_explicit_token_capability_is_skipped_with_a_warning() -> None:
    backend = FakeBackend(
        capabilities=caps(top_logprobs_max=3, explicit_token_logprobs=True),
        next_token=_TRUNCATING,
        sequences={" C": [(" C", lp(0.05))], "C": [("C", -math.inf)]},
        vocab=LETTER_VOCAB,
    )
    result = await run(S1, backend, choice(3))
    assert any("explicit-token" in w for w in result.warnings)
    assert result.raw_logprobs[2] == pytest.approx(lp(0.05))


_six_logprobs = st.lists(
    st.floats(min_value=-20.0, max_value=0.0, allow_nan=False), min_size=6, max_size=6
)


@settings(max_examples=60, deadline=None)
@given(
    labels=_six_logprobs,
    junk=st.lists(st.floats(min_value=-20.0, max_value=0.0), max_size=4),
    cap=st.integers(1, 10),
)
def test_upper_bound_never_lifts_a_missing_label_above_an_observed_one(
    labels: list[float], junk: list[float], cap: int
) -> None:
    table = {f" {chr(65 + i)}": v for i, v in enumerate(labels)}
    table.update({f" junk{i}": v for i, v in enumerate(junk)})
    backend = FakeBackend(
        capabilities=caps(top_logprobs_max=cap, echo_prompt_logprobs=False),
        next_token=table,
        vocab=[*LETTER_VOCAB, *(f" junk{i}" for i in range(len(junk)))],
    )
    result = asyncio.run(run(S1, backend, choice(6)))
    pairs = list(zip(result.keys, result.raw_logprobs, strict=True))
    observed = [v for k, v in pairs if k not in result.missing]
    missing = [v for k, v in pairs if k in result.missing]
    assert all(math.isfinite(v) for v in missing)  # never a silent zero
    if observed and missing:
        assert max(missing) <= min(observed)
    assert math.fsum(probs(result)) == pytest.approx(1.0)
    assert result.truncated == bool(missing)


@settings(max_examples=60, deadline=None)
@given(
    labels=_six_logprobs,
    junk=st.dictionaries(
        st.sampled_from([" **", "\n", " "]), st.floats(min_value=-20.0, max_value=0.0)
    ),
    reflected=st.booleans(),
)
def test_constrained_result_is_exact_and_unchanged_by_invalid_tokens(
    labels: list[float], junk: dict[str, float], reflected: bool
) -> None:
    table = {f" {chr(65 + i)}": v for i, v in enumerate(labels)}
    norm = math.log(math.fsum(math.exp(x) for x in labels))
    expected = [math.exp(v - norm) for v in labels]
    for extra in ({}, junk):
        backend = FakeBackend(
            capabilities=caps(mask_reflected_in_logprobs=reflected),
            next_token={**extra, **table},
            vocab=LETTER_VOCAB,
        )
        result = asyncio.run(run(S2, backend, choice(6)))
        assert probs(result) == pytest.approx(expected, rel=1e-9, abs=1e-12)
        assert all(p >= 0.0 for p in probs(result))


# --- observed mass and limits -------------------------------------------------------------------


async def test_low_observed_mass_warns_under_raw_logprobs_only() -> None:
    table = {" **": lp(0.7), " A": lp(0.2), " B": lp(0.1)}
    raw = await run(S1, FakeBackend(next_token=table, vocab=LETTER_VOCAB), choice(2))
    assert raw.observed_mass == pytest.approx(0.3)
    assert any("wanted to write something else" in w for w in raw.warnings)
    processed = FakeBackend(
        capabilities=caps(logprobs_mode="processed_logprobs"), next_token=table, vocab=LETTER_VOCAB
    )
    assert (await run(S1, processed, choice(2))).warnings == ()


async def test_more_options_than_letters_is_rejected() -> None:
    backend = FakeBackend(next_token={" A": 0.0}, vocab=LETTER_VOCAB)
    with pytest.raises(LabelCapacityError):
        await run(S1, backend, choice(53))


@pytest.mark.parametrize(("n", "fits"), [(32, True), (33, False)])
def test_first_token_strategies_claim_exactness_only_when_every_surface_fits(
    n: int, fits: bool
) -> None:
    assert S1.supports(VLLM_LIKE, choice(n)) is fits
    assert S2.supports(VLLM_LIKE, choice(n)) is fits
    assert not S2.supports(caps(structured_choice=False), choice(2))


LUNA_LIKE = caps(
    top_logprobs_max=5,
    structured_choice=False,
    echo_prompt_logprobs=False,
    assistant_prefill=False,
    local_tokenizer=False,
    top_logprobs_exact=False,
)
"""A hosted chat API as gpt-6-luna measured: no prefill, echo or tokenizer; top 5, cut by
probability."""


async def test_no_prefill_no_tokenizer_merges_surfaces_and_bounds_a_missing_label() -> None:
    table = {"A": lp(0.6), "B": lp(0.2), "**": lp(0.1), " A": lp(0.05), "The": lp(0.04)}
    table["C"] = lp(0.005)  # sixth: not returned
    # No vocab: any tokenize call raises, so the plan must not need a tokenizer.
    backend = FakeBackend(capabilities=LUNA_LIKE, next_token=table)
    renderer = PromptRenderer("state_first", prefill=False)
    result = await S1.score(backend, renderer, STATE, choice(3))

    (_, call), *_ = backend.calls
    prompt = call["prompt"]
    assert not prompt.has_prefill
    assert prompt.messages[-1].content.endswith("Respond with the letter only.")
    assert "-noprefill-" in prompt.template_id
    # "C" is not returned: it gets the leftover mass 1 - 0.99, below the last entry (0.04).
    assert result.missing == ("opt-2",)
    assert result.truncated
    raw = [0.65, 0.2, 0.01]
    assert probs(result) == pytest.approx([p / sum(raw) for p in raw], rel=1e-9)
    assert result.observed_mass == pytest.approx(0.85)


async def test_no_prefill_refuses_more_labels_than_the_top_k_shows() -> None:
    backend = FakeBackend(capabilities=LUNA_LIKE, next_token={"A": 0.0})
    renderer = PromptRenderer("state_first", prefill=False)
    assert S1.supports(LUNA_LIKE, choice(5))
    assert not S1.supports(LUNA_LIKE, choice(6))
    with pytest.raises(CapabilityError, match="do not fit"):
        await S1.score(backend, renderer, STATE, choice(6))
    assert backend.calls == []
