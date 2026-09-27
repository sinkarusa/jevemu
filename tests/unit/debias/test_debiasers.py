"""Debiasers on a FakeBackend whose model multiplies each option's content weight by a bias
on its *position* (letter), so the true answer distribution is known exactly."""

from __future__ import annotations

import math
import re
import string
from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from jevemu.backends.base import RenderedPrompt
from jevemu.backends.fake import FakeBackend
from jevemu.debias import (
    BatchDebiaser,
    ContextualDebiaser,
    LabelPriors,
    PermutationDebiaser,
    PriDeDebiaser,
    cyclic_orders,
    debiaser_from_spec,
    fit_batch_priors,
    free_positions,
    permuted_question,
    prior_key,
)
from jevemu.emulator import Emulator
from jevemu.eval.crossfit import CalItem, apply_label_priors
from jevemu.render import CHOICE_LABELS
from jevemu.types import ChoiceAnswer, ChoiceQuestion, SystemOneRequest

WEIGHTS = {"apple": 0.5, "banana": 0.3, "cherry": 0.15, "date": 0.05, "I don't know": 0.1}
"""Content weights: the unbiased model's preference (apple > banana > cherry > date)."""
BIAS = {"A": 12.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0}
"""Position bias: the model over-picks whatever sits at A."""
OPTION = re.compile(r"^([A-Z])\) (\S+)(?:: (.+))?$")
VOCAB = [
    *CHOICE_LABELS,
    *(" " + label for label in CHOICE_LABELS),
    *("Yes", " Yes", "No", " No"),
    *string.printable,
]


def biased_model(prompt: RenderedPrompt, allowed: Sequence[str] | None) -> Mapping[str, float]:
    """``P(letter) ∝ weight(content at letter) * BIAS[letter]``; contents weigh the same under
    the content-free state ``"N/A"``."""
    text = "\n".join(message.content for message in prompt.messages)
    content_free = "State:\nN/A" in text
    table = {}
    for line in text.splitlines():
        match = OPTION.match(line)
        if match:
            letter, name, description = match.groups()
            weight = 1.0 if content_free else WEIGHTS[description or name]
            table[" " + letter] = math.log(weight * BIAS[letter])
    return table


def backend() -> FakeBackend:
    return FakeBackend(next_token=biased_model, vocab=VOCAB)


def question(*contents: str) -> ChoiceQuestion:
    return ChoiceQuestion(
        instructions="Which fruit?",
        criteria={CHOICE_LABELS[i]: text for i, text in enumerate(contents)},
    )


def truth(q: ChoiceQuestion) -> dict[str, float]:
    total = sum(WEIGHTS[str(text)] for text in q.criteria.values())
    return {key: WEIGHTS[str(text)] / total for key, text in q.criteria.items()}


async def ask(emulator: Emulator, q: ChoiceQuestion, state: str = "A basket.") -> ChoiceAnswer:
    response = await emulator.system_one(SystemOneRequest(state=state, questions={"answer": q}))
    answer = response.answers["answer"]
    assert isinstance(answer, ChoiceAnswer)
    return answer


WRONG_AT_A = question("date", "cherry", "banana", "apple")
"""Raw argmax is ``A`` (date, 0.05 x 12 beats apple's 0.5)."""


async def test_the_injected_bias_flips_the_raw_answer() -> None:
    answer = await ask(Emulator(backend()), WRONG_AT_A)
    assert answer.choice == "A"


# --- positions -----------------------------------------------------------------------------


def test_cyclic_orders_move_every_free_option_through_every_free_position() -> None:
    free = (True, True, True, True, False)  # E is "I don't know"
    orders = cyclic_orders(free)
    assert orders[0] == (0, 1, 2, 3, 4)
    assert len(orders) == 4
    for position in range(4):
        assert sorted(order[position] for order in orders) == [0, 1, 2, 3]
    assert all(order[4] == 4 for order in orders)


def test_permuted_letter_keyed_question_moves_descriptions_not_keys() -> None:
    q = question("date", "cherry", "banana", "apple")
    moved = permuted_question(q, (3, 0, 1, 2))
    assert list(moved.criteria) == ["A", "B", "C", "D"]
    assert list(moved.criteria.values()) == ["apple", "date", "cherry", "banana"]


def test_permuted_named_question_reorders_the_options() -> None:
    q = ChoiceQuestion(instructions="Topic?", criteria={"World": None, "Sports": "games"})
    assert list(permuted_question(q, (1, 0)).criteria.items()) == [
        ("Sports", "games"),
        ("World", None),
    ]


def test_idk_option_is_fixed_and_part_of_the_prior_key() -> None:
    q = question("date", "cherry", "banana", "apple", "I don't know")
    free = free_positions(q)
    assert free == (True, True, True, True, False)
    assert prior_key(q, free).endswith("|fixed=4")
    assert prior_key(WRONG_AT_A, free_positions(WRONG_AT_A)) != prior_key(q, free)


# --- permutation ---------------------------------------------------------------------------


async def test_permutation_restores_the_content_ranking() -> None:
    answer = await ask(Emulator(backend(), debiaser=PermutationDebiaser()), WRONG_AT_A)
    probs = answer.probabilities
    assert answer.choice == "D"  # apple
    assert probs["D"] > probs["C"] > probs["B"] > probs["A"]  # apple > banana > cherry > date


async def test_permutation_is_invariant_to_rotating_the_input_order() -> None:
    emulator = Emulator(backend(), debiaser=PermutationDebiaser())
    first = await ask(emulator, question("date", "cherry", "banana", "apple"))
    rotated = await ask(emulator, question("banana", "apple", "date", "cherry"))
    by_content = dict(zip(WRONG_AT_A.criteria.values(), first.probabilities.values(), strict=True))
    rotated_by_content = dict(
        zip(("banana", "apple", "date", "cherry"), rotated.probabilities.values(), strict=True)
    )
    assert rotated_by_content == pytest.approx(by_content, abs=1e-12)


async def test_permutation_diagnostics_count_every_presentation() -> None:
    fake = backend()
    emulator = Emulator(fake, debiaser=PermutationDebiaser(), include_diagnostics=True)
    plain = Emulator(backend())
    q = question("date", "cherry", "banana", "apple", "I don't know")
    request = SystemOneRequest(state="A basket.", questions={"answer": q})
    response = await emulator.system_one(request)
    baseline = await plain.system_one(request)
    assert response.x_jevemu is not None
    diag = response.x_jevemu["answer"]
    assert diag.debiaser == PermutationDebiaser().id
    assert (diag.permutations, diag.debias_calls, diag.n_backend_calls) == (4, 3, 4)
    assert [score.order[-1] for score in diag.permutation_scores] == ["E"] * 4  # IDK stays last
    assert diag.permutation_scores[0].order == ["A", "B", "C", "D", "E"]
    baseline_answer = baseline.answers["answer"]
    assert isinstance(baseline_answer, ChoiceAnswer)
    assert diag.raw_probabilities == pytest.approx(baseline_answer.probabilities)
    assert response.usage.input_tokens == 4 * baseline.usage.input_tokens  # same-length prompts


async def test_questions_above_max_options_pass_through_with_a_warning() -> None:
    emulator = Emulator(
        backend(), debiaser=PermutationDebiaser(max_options=3), include_diagnostics=True
    )
    response = await emulator.system_one(
        SystemOneRequest(state="A basket.", questions={"answer": WRONG_AT_A})
    )
    assert response.x_jevemu is not None
    diag = response.x_jevemu["answer"]
    assert diag.debias_calls == 0
    assert any("max_options=3" in warning for warning in diag.warnings)


# --- pride -------------------------------------------------------------------------------------


async def test_pride_estimates_the_position_prior_and_removes_it_exactly() -> None:
    debiaser = PriDeDebiaser(alpha=0.0)  # only the first question of the key is permuted
    emulator = Emulator(backend(), debiaser=debiaser, include_diagnostics=True)
    await ask(emulator, question("apple", "banana", "cherry", "date"))  # estimation question
    q = question("date", "banana", "cherry", "apple")
    response = await emulator.system_one(SystemOneRequest(state="x", questions={"answer": q}))
    answer = response.answers["answer"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.probabilities == pytest.approx(truth(q), abs=1e-9)
    assert response.x_jevemu is not None
    diag = response.x_jevemu["answer"]
    assert diag.debias_calls == 0
    total = sum(BIAS[k] for k in "ABCD")
    assert diag.debias_prior == pytest.approx({k: BIAS[k] / total for k in "ABCD"}, abs=1e-9)


async def test_pride_with_fitted_priors_needs_no_extra_calls() -> None:
    total = sum(BIAS[k] for k in "ABCD")
    key = prior_key(WRONG_AT_A, free_positions(WRONG_AT_A))
    priors = LabelPriors("pride", {key: tuple(BIAS[k] / total for k in "ABCD")}, {key: 1})
    fake = backend()
    answer = await ask(Emulator(fake, debiaser=PriDeDebiaser(priors=priors)), WRONG_AT_A)
    assert answer.probabilities == pytest.approx(truth(WRONG_AT_A), abs=1e-9)
    assert [name for name, _ in fake.calls].count("next_token_logprobs") == 1


async def test_offline_priors_reproduce_what_the_deployed_debiaser_answers() -> None:
    # scripts/calibrate.py registry --priors fits the deployment calibrator on
    # apply_label_priors(raw probabilities); it must see exactly the deployed debiaser's output.
    q = question("date", "cherry", "banana", "apple", "I don't know")
    free = free_positions(q)
    key = prior_key(q, free)
    priors = LabelPriors("pride", {key: (0.4, 0.3, 0.2, 0.1)}, {key: 25})
    raw = np.array(list((await ask(Emulator(backend()), q)).probabilities.values()))
    item = CalItem(
        item_id="q",
        benchmark="b",
        question_id="q",
        stratum="",
        signature=key,
        prior_key=key,
        free=free,
        keys=tuple(q.criteria),
        gold=0,
        probs=raw,
        returned=raw,
        raw=raw,
    )
    offline = apply_label_priors([item], priors)["q"]
    assert offline.tolist() != pytest.approx(raw.tolist(), abs=1e-3)  # the prior moved it
    for debiaser in (PriDeDebiaser(priors=priors), BatchDebiaser(priors)):
        deployed = await ask(Emulator(backend(), debiaser=debiaser), q)
        assert list(deployed.probabilities.values()) == pytest.approx(offline.tolist(), abs=1e-12)


# --- contextual --------------------------------------------------------------------------------


async def test_contextual_divides_out_the_content_free_distribution_once_per_question() -> None:
    emulator = Emulator(backend(), debiaser=ContextualDebiaser(), include_diagnostics=True)
    request = SystemOneRequest(state="A basket.", questions={"answer": WRONG_AT_A})
    first = await emulator.system_one(request)
    second = await emulator.system_one(request.model_copy(update={"state": "Another basket."}))
    for response in (first, second):
        answer = response.answers["answer"]
        assert isinstance(answer, ChoiceAnswer)
        assert answer.probabilities == pytest.approx(truth(WRONG_AT_A), abs=1e-9)
    assert first.x_jevemu is not None
    assert second.x_jevemu is not None
    assert first.x_jevemu["answer"].debias_calls == 1
    assert second.x_jevemu["answer"].debias_calls == 0  # cached per question


# --- batch -------------------------------------------------------------------------------------


async def test_batch_prior_from_a_recorded_batch_fixes_every_biased_answer() -> None:
    contents = ["date", "cherry", "banana", "apple"]
    batch = [question(*(contents[(j + s) % 4] for j in range(4))) for s in range(4)]
    plain = Emulator(backend())
    raw = [await ask(plain, q) for q in batch]
    apple = ["ABCD"[[str(t) for t in q.criteria.values()].index("apple")] for q in batch]
    assert [a.choice for a in raw] != apple  # the bias wins somewhere
    priors = fit_batch_priors(
        (prior_key(q, free_positions(q)), list(a.probabilities.values()), free_positions(q))
        for q, a in zip(batch, raw, strict=True)
    )
    debiased = Emulator(backend(), debiaser=BatchDebiaser(priors))
    assert [(await ask(debiased, q)).choice for q in batch] == apple


# --- specs -------------------------------------------------------------------------------------


def test_spec_parameters_reach_the_debiaser() -> None:
    permutation = debiaser_from_spec("permutation:n=2,max_options=none")
    assert isinstance(permutation, PermutationDebiaser)
    assert (permutation.n_permutations, permutation.max_options) == (2, None)
    contextual = debiaser_from_spec("contextual:content_free=N/A|[MASK]")
    assert isinstance(contextual, ContextualDebiaser)
    assert contextual.content_free == ("N/A", "[MASK]")
    with pytest.raises(ValueError, match="needs priors"):
        debiaser_from_spec("batch")
    with pytest.raises(ValueError, match="bad parameter"):
        debiaser_from_spec("pride:beta=1")


def test_label_priors_round_trip_through_json() -> None:
    priors = LabelPriors("batch", {"choice:4:x": (0.4, 0.2, 0.2, 0.2)}, {"choice:4:x": 7})
    assert LabelPriors.from_json(priors.to_json()) == priors
    with pytest.raises(ValueError, match="sum to 1"):
        LabelPriors("batch", {"k": (0.5, 0.2)}, {"k": 1})
