"""S3 token trie on scripted multi-token vocabularies."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
from scoring_helpers import RENDERER, STATE, VLLM_LIKE, caps, lp

from jevemu.backends.base import RenderedPrompt
from jevemu.backends.fake import FakeBackend
from jevemu.scoring import TrieStrategy
from jevemu.types import ChoiceQuestion, ScoreQuestion

PETS = ChoiceQuestion(
    instructions="Which one?", criteria={"cat": None, "cat food": None, "dog": None, "dove": None}
)
PET_VOCAB = [" ca", "t", " food", " do", "g", "ve", " x", "\n", ".", "or"]
"""``" cat"`` = ``" ca" + "t"``; ``" cat food"`` continues it; ``" dog"``/``" dove"`` split."""


def _pets(prompt: RenderedPrompt, allowed: Sequence[str] | None) -> dict[str, float]:
    return {
        "Answer:": {" ca": lp(0.6), " do": lp(0.3), " x": lp(0.1)},
        "Answer: cat": {" food": lp(0.25), "\n": lp(0.5), ".": lp(0.25)},
        "Answer: do": {"g": lp(0.2), "ve": lp(0.6), "or": lp(0.2)},
    }[prompt.prefill]


def prefills(backend: FakeBackend) -> list[str]:
    return [args["prompt"].prefill for name, args in backend.calls if name == "next_token_logprobs"]


async def test_chain_rule_over_multi_token_labels_with_an_end_outcome() -> None:
    backend = FakeBackend(capabilities=VLLM_LIKE, next_token=_pets, vocab=PET_VOCAB)
    result = await TrieStrategy().score(backend, RENDERER, STATE, PETS)
    # Root: cat-branch 0.6/0.9, dog-branch 0.3/0.9. After " cat" the label may end (anything
    # but " food": 0.75) or continue. After " do": g 0.2/0.8, ve 0.6/0.8.
    assert result.keys == ("cat", "cat food", "dog", "dove")
    assert [math.exp(v) for v in result.logprobs] == pytest.approx(
        [2 / 3 * 0.75, 2 / 3 * 0.25, 1 / 3 * 0.25, 1 / 3 * 0.75], rel=1e-12
    )
    assert result.label_scheme == "keys"
    # Only decision points are queried: " ca" -> "t" has one outcome and costs nothing.
    assert sorted(prefills(backend)) == ["Answer:", "Answer: cat", "Answer: do"]
    assert result.n_backend_calls == 3
    assert not result.truncated


async def test_digits_share_the_space_and_cost_one_query() -> None:
    def table(prompt: RenderedPrompt, allowed: Sequence[str] | None) -> dict[str, float]:
        assert prompt.prefill == "Answer: "
        return {"0": lp(0.1), "1": lp(0.6), "2": lp(0.3)}

    backend = FakeBackend(capabilities=VLLM_LIKE, next_token=table, vocab=[" ", "0", "1", "2"])
    question = ScoreQuestion(instructions="Rate.", criteria=["low", "mid", "high"])
    result = await TrieStrategy().score(backend, RENDERER, STATE, question)
    assert result.probabilities == pytest.approx({"0": 0.1, "1": 0.6, "2": 0.3})
    assert result.n_backend_calls == 1


def _weights(depth: int) -> dict[str, float]:
    if depth == 0:
        return {"0": 0.9, "1": 0.09999, "2": 0.00001}
    return {str(d): float(d + 1) for d in range(10)}


def _synthetic(prompt: RenderedPrompt, allowed: Sequence[str] | None) -> dict[str, float]:
    partial = prompt.prefill.removeprefix("Answer: opt ")
    return {token: math.log(w) for token, w in _weights(len(partial)).items()}


def _expected_255() -> list[float]:
    """Exact trie probabilities: children renormalized over the digits that lead to a label."""
    tau = 1e-4
    out: list[float] = []
    for i in range(255):
        digits = f"{i:03d}"
        p = 1.0
        prefix = ""
        for depth, digit in enumerate(digits):
            valid = sorted({f"{j:03d}"[depth] for j in range(255) if f"{j:03d}".startswith(prefix)})
            weights = _weights(depth)
            if p < tau:
                break  # pruned: the label keeps its subtree's mass as an upper bound
            p *= weights[digit] / math.fsum(weights[d] for d in valid)
            prefix += digit
        out.append(p)
    total = math.fsum(out)
    return [p / total for p in out]


async def test_255_options_prune_below_tau_and_report_the_pruned_mass() -> None:
    keys = [f"opt {i:03d}" for i in range(255)]
    question = ChoiceQuestion(instructions="Pick.", criteria=dict.fromkeys(keys))
    backend = FakeBackend(
        capabilities=VLLM_LIKE,
        next_token=_synthetic,
        vocab=[" opt", " ", *"0123456789"],
    )
    result = await TrieStrategy(tau=1e-4).score(backend, RENDERER, STATE, question)
    assert result.keys == tuple(keys)
    assert [math.exp(v) for v in result.logprobs] == pytest.approx(_expected_255(), rel=1e-9)
    # "opt 2xx" is reached with 1e-5 < tau: its 55 labels are not queried.
    assert result.truncated
    (pruned,) = [w for w in result.warnings if "pruned" in w]
    assert "1 subtrees" in pruned
    assert "55 labels" in pruned
    assert "1e-05" in pruned
    # Queried: " opt " (3 children), "0" and "1" (10 each), and their 20 children.
    assert result.n_backend_calls == 1 + 2 + 20
    assert math.exp(result.raw_logprobs[keys.index("opt 254")]) == pytest.approx(1e-5)


async def test_children_missing_from_top_k_are_filled_not_zeroed() -> None:
    backend = FakeBackend(
        capabilities=caps(top_logprobs_max=2, echo_prompt_logprobs=False, structured_choice=False),
        next_token={" a": lp(0.5), " b": lp(0.3), " c": lp(0.1), " z": lp(0.1)},
        vocab=[" a", " b", " c", " z"],
    )
    question = ChoiceQuestion(instructions="Pick.", criteria=dict.fromkeys(["a", "b", "c"]))
    result = await TrieStrategy().score(backend, RENDERER, STATE, question)
    assert result.probabilities == pytest.approx({"a": 0.5 / 1.1, "b": 0.3 / 1.1, "c": 0.3 / 1.1})
    assert result.missing == ("c",)
    assert result.truncated


class _SentencePieceLike(FakeBackend):
    """Decodes a lone token without its leading space, as SentencePiece tokenizers do."""

    async def detokenize(self, ids: Sequence[int]) -> str:
        return (await super().detokenize(ids)).lstrip()


async def test_tokens_that_cannot_be_addressed_by_prefill_text_are_rejected() -> None:
    backend = _SentencePieceLike(capabilities=VLLM_LIKE, next_token=_pets, vocab=PET_VOCAB)
    with pytest.raises(ValueError, match="does not decode token by token"):
        await TrieStrategy().score(backend, RENDERER, STATE, PETS)
