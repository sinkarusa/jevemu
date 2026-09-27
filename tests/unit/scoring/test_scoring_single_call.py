"""``auto_single``: one model call per question; code labels above 32 options."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
from scoring_helpers import LETTER_VOCAB, RENDERER, STATE, VLLM_LIKE, caps, choice, lp

from jevemu.backends.base import CapabilityError, RenderedPrompt
from jevemu.backends.fake import FakeBackend
from jevemu.render import CODE_LABELS, LabelCapacityError
from jevemu.scoring import NoEchoAutoStrategy, ScoreResult, SingleCallStrategy
from jevemu.scoring.first_token import grammar_first_tokens
from jevemu.scoring.single_call import single_token_codes

WIDE = caps(top_logprobs_max=576)
"""vLLM 0.30.0 launched with ``--max-logprobs 576``."""

SPLIT_CODES = ("AB", "AD")
"""``AB`` is one token only bare, ``AD`` only space-prefixed: neither is usable."""

USABLE = tuple(c for c in CODE_LABELS[:60] if c not in SPLIT_CODES)


def _vocab(n_codes: int = 60) -> list[str]:
    """Single characters (so anything tokenizes) plus the first ``n_codes`` candidate codes,
    both surfaces each, except the half-single :data:`SPLIT_CODES`."""
    codes = CODE_LABELS[:n_codes]
    whole = [s for c in codes if c not in SPLIT_CODES for s in (c, " " + c)]
    return [*LETTER_VOCAB, *whole, "AB", " AD"]


def _codes_table(weights: dict[str, float], allowed: Sequence[str] | None) -> dict[str, float]:
    """Every allowed code surface at probability 1e-6 with ``weights`` on top, as a reflected
    mask shows them all, plus a token the constraint masks."""
    assert allowed is not None, "codes are always read under the choice constraint"
    table = {surface: lp(1e-6) for surface in allowed}
    table.update({surface: lp(p) for surface, p in weights.items()})
    table["\n"] = lp(0.2)
    return table


async def test_codes_skip_labels_without_two_single_token_surfaces() -> None:
    backend = FakeBackend(capabilities=WIDE, vocab=_vocab())
    codes = await single_token_codes(backend, 40)
    assert codes == USABLE[:40]
    assert codes[:3] == ("AA", "AC", "AE")


async def test_too_few_single_token_codes_raise() -> None:
    backend = FakeBackend(capabilities=WIDE, vocab=_vocab(n_codes=36))
    with pytest.raises(LabelCapacityError, match="codes"):
        await single_token_codes(backend, 40)


async def test_codes_are_read_in_one_constrained_call_with_surfaces_merged() -> None:
    weights = {" AC": 0.5, "AC": 0.1, " AA": 0.2}

    def table(prompt: RenderedPrompt, allowed: Sequence[str] | None) -> dict[str, float]:
        return _codes_table(weights, allowed)

    backend = FakeBackend(capabilities=WIDE, next_token=table, vocab=_vocab())
    result = await SingleCallStrategy().score(backend, RENDERER, STATE, choice(40))

    calls = [args for name, args in backend.calls if name == "next_token_logprobs"]
    assert len(calls) == 1
    assert result.n_backend_calls == 1
    (call,) = calls
    assert set(call["allowed"]) == {p + c for c in USABLE[:40] for p in ("", " ")}
    assert call["top_k"] == grammar_first_tokens(call["allowed"])
    text = "\n".join(m.content for m in call["prompt"].messages)
    assert "AA) opt-0\nAC) opt-1\nAE) opt-2" in text
    assert "AB)" not in text
    assert (result.strategy, result.label_scheme) == ("constrained", "codes")
    assert not result.missing
    assert not result.truncated
    probs = result.probabilities
    assert math.fsum(probs.values()) == pytest.approx(1.0)
    total = 0.8 + 1e-6 + 2 * 38 * 1e-6  # the weights, bare "AA", both surfaces of 38 codes
    assert probs["opt-1"] == pytest.approx(0.6 / total)
    assert probs["opt-0"] == pytest.approx((0.2 + 1e-6) / total)
    assert probs["opt-2"] == pytest.approx(2e-6 / total)


async def test_unseen_codes_get_the_upper_bound_without_echo_calls() -> None:
    def table(prompt: RenderedPrompt, allowed: Sequence[str] | None) -> dict[str, float]:
        return {" AC": lp(0.7), " AA": lp(0.3)}  # the other 38 codes never come back

    # The backend can echo; auto would echo-score the missing codes.
    backend = FakeBackend(capabilities=WIDE, next_token=table, vocab=_vocab())
    result = await SingleCallStrategy().score(backend, RENDERER, STATE, choice(40))
    reads = [name for name, _ in backend.calls if name != "tokenize"]
    assert reads == ["next_token_logprobs"]
    assert result.n_backend_calls == 1
    assert result.truncated
    assert len(result.missing) == 38
    assert any("echo off" in w for w in result.warnings)


async def test_codes_that_do_not_fit_the_top_k_raise_before_any_model_call() -> None:
    backend = FakeBackend(capabilities=caps(top_logprobs_max=64), vocab=_vocab())
    # 80 code surfaces + the space + "A", "B", " A", " B" (grammar prefixes) = 85
    with pytest.raises(CapabilityError, match="need 85 top logprobs"):
        await SingleCallStrategy().score(backend, RENDERER, STATE, choice(40))
    assert not [name for name, _ in backend.calls if name == "next_token_logprobs"]


async def test_codes_need_the_mask_reflected() -> None:
    unreflected = caps(WIDE, mask_reflected_in_logprobs=False)
    backend = FakeBackend(capabilities=unreflected, vocab=_vocab())
    with pytest.raises(CapabilityError, match="reflected"):
        await SingleCallStrategy().score(backend, RENDERER, STATE, choice(40))
    assert not backend.calls


async def test_letter_questions_score_exactly_as_auto_noecho() -> None:
    def table(prompt: RenderedPrompt, allowed: Sequence[str] | None) -> dict[str, float]:
        return {" B": lp(0.6), "B": lp(0.1), " C": lp(0.2), "\n": lp(0.1)}  # A, D, E unseen

    results: list[ScoreResult] = []
    calls = []
    for strategy in (SingleCallStrategy(), NoEchoAutoStrategy()):
        backend = FakeBackend(capabilities=VLLM_LIKE, next_token=table, vocab=LETTER_VOCAB)
        results.append(await strategy.score(backend, RENDERER, STATE, choice(5)))
        calls.append(backend.calls)
    assert results[0] == results[1]
    assert calls[0] == calls[1]
    assert results[0].label_scheme == "letters"
    assert results[0].n_backend_calls == 1


async def test_letter_questions_that_do_not_fit_one_read_raise() -> None:
    # auto_noecho would fall back to the multi-call trie here.
    backend = FakeBackend(capabilities=caps(top_logprobs_max=5), vocab=LETTER_VOCAB)
    with pytest.raises(CapabilityError, match="one call"):
        await SingleCallStrategy().score(backend, RENDERER, STATE, choice(4))
    assert not backend.calls
