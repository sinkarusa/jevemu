"""The emulator facade on a scripted FakeBackend: answer assembly, calibration, rounding,
diagnostics, usage, per-question independence, concurrency and all-or-nothing errors."""

from __future__ import annotations

import asyncio
import json
import math
import string
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from jevemu.backends.base import Backend, RenderedPrompt
from jevemu.backends.fake import FakeBackend
from jevemu.calibrate import CalibrationKey, CalibratorRegistry, TemperatureCalibrator
from jevemu.emulator import Emulator, QuestionError
from jevemu.render import CHOICE_LABELS, PromptRenderer
from jevemu.scoring import ConstrainedStrategy, EchoStrategy, ScoreResult
from jevemu.types import (
    ChoiceAnswer,
    ChoiceQuestion,
    JSONish,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
)

DOC_REQUESTS = sorted(
    (Path(__file__).parents[1] / "golden" / "fixtures" / "jev_docs").glob("*__request.json")
)

VOCAB: list[str] = [
    *CHOICE_LABELS,
    *(" " + label for label in CHOICE_LABELS),
    "Yes",
    " Yes",
    "No",
    " No",
    *string.printable,
]
"""Letters and Yes/No are one token bare and spaced; a spaced digit is ``" "`` + digit (Qwen3)."""

Table = Mapping[str, float]


def lp(p: float) -> float:
    return math.log(p) if p > 0 else -math.inf


def letters(*probs: float) -> Table:
    return {" " + CHOICE_LABELS[i]: lp(p) for i, p in enumerate(probs)}


def digits(*probs: float) -> Table:
    return {str(i): lp(p) for i, p in enumerate(probs)}


def yes(p: float) -> Table:
    return {" Yes": lp(p), " No": lp(1.0 - p)}


def prompt_text(prompt: RenderedPrompt) -> str:
    return "\n".join(message.content for message in prompt.messages)


def by_marker(
    tables: Mapping[str, Table],
) -> Callable[[RenderedPrompt, Sequence[str] | None], Table]:
    """Next-token table of the question whose marker (e.g. its instructions) is in the prompt."""

    def next_token(prompt: RenderedPrompt, allowed: Sequence[str] | None) -> Table:
        text = prompt_text(prompt)
        (table,) = [table for marker, table in tables.items() if marker in text]
        return table

    return next_token


def fake(tables: Mapping[str, Table]) -> FakeBackend:
    return FakeBackend(next_token=by_marker(tables), vocab=VOCAB)


def request(questions: Mapping[str, Question], state: JSONish = "A ticket.") -> SystemOneRequest:
    return SystemOneRequest(state=state, questions=dict(questions))


COLOR = ChoiceQuestion(
    instructions="Which color?", criteria={"red": None, "green": "grass", "blue": None}
)
SEVERITY = ScoreQuestion(instructions="How severe?", criteria=["none", "some", {"level": "max"}])
URGENT = NoulQuestion(instructions="Is it urgent?")


async def test_answers_are_assembled_per_question_type() -> None:
    backend = fake(
        {
            "Which color?": letters(0.2, 0.5, 0.3),
            "How severe?": digits(0.1, 0.3, 0.6),
            "Is it urgent?": yes(0.8),
        }
    )
    response = await Emulator(backend).system_one(
        request({"color": COLOR, "severity": SEVERITY, "urgent": URGENT})
    )

    color = response.answers["color"]
    assert isinstance(color, ChoiceAnswer)
    assert list(color.probabilities) == ["red", "green", "blue"]  # question order
    assert list(color.probabilities.values()) == pytest.approx([0.2, 0.5, 0.3])
    assert color.choice == "green"
    assert color.confidence == pytest.approx((3 * 0.5 - 1) / 2)  # mode_distance = peak_linear

    severity = response.answers["severity"]
    assert isinstance(severity, ScoreAnswer)
    assert list(severity.probabilities) == ["0", "1", "2"]
    assert severity.score == pytest.approx(0.3 + 2 * 0.6)
    assert severity.legend == {"0": "none", "1": "some", "2": {"level": "max"}}
    # mode_distance over ordered levels: 1 - E|X - mode| / uniform spread ((K^2 - 1) / 4K)
    assert severity.confidence == pytest.approx(1 - (0.1 * 2 + 0.3 * 1) / (8 / 12))

    urgent = response.answers["urgent"]
    assert isinstance(urgent, NoulAnswer)
    assert urgent.noul == pytest.approx(0.8)
    assert response.model == "jevemu/fake-model"


async def test_a_tie_goes_to_the_option_listed_first() -> None:
    question = ChoiceQuestion(instructions="Pick.", criteria={"z": None, "y": None, "x": None})
    backend = fake({"Pick.": letters(0.2, 0.4, 0.4)})
    response = await Emulator(backend).system_one(request({"q": question}))
    answer = response.answers["q"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "y"


def test_confidence_function_is_pluggable_and_told_which_answers_are_ordinal() -> None:
    ordinal_flags: list[bool] = []

    def spread(probs: Sequence[float], *, ordinal: bool = False) -> float:
        ordinal_flags.append(ordinal)
        return max(probs) - min(probs)

    backend = fake({"Which color?": letters(0.2, 0.5, 0.3), "How severe?": digits(0.1, 0.3, 0.6)})
    emulator = Emulator(backend, confidence_fn=spread)
    answers = emulator.system_one_sync(request({"color": COLOR, "severity": SEVERITY})).answers
    color, severity = answers["color"], answers["severity"]
    assert isinstance(color, ChoiceAnswer)
    assert isinstance(severity, ScoreAnswer)
    assert color.confidence == pytest.approx(0.3)
    assert severity.confidence == pytest.approx(0.5)
    assert sorted(ordinal_flags) == [False, True]  # Score levels are ordinal, Choice is not


def _key(emulator: Emulator, question: Question) -> CalibrationKey:
    return CalibrationKey.for_question(
        backend="fake",
        model="fake-model",
        template_id=emulator.renderer.template_id,
        question=question,
    )


async def test_calibrator_softens_probabilities_but_keeps_the_argmax() -> None:
    backend = fake({"Which color?": letters(0.1, 0.7, 0.2)})
    registry = CalibratorRegistry()
    emulator = Emulator(backend, calibrators=registry, include_diagnostics=True)
    registry.register(_key(emulator, COLOR), TemperatureCalibrator(2.0))

    response = await emulator.system_one(request({"color": COLOR}))

    answer = response.answers["color"]
    assert isinstance(answer, ChoiceAnswer)
    raw = [0.1, 0.7, 0.2]
    tempered = [math.sqrt(p) for p in raw]  # softmax(log p / 2)
    expected = [t / sum(tempered) for t in tempered]
    assert list(answer.probabilities.values()) == pytest.approx(expected)
    assert answer.choice == "green"
    assert answer.probabilities["green"] < 0.7
    assert response.x_jevemu is not None
    diagnostics = response.x_jevemu["color"]
    assert diagnostics.calibrator == "temperature:exact"
    assert list(diagnostics.raw_probabilities.values()) == pytest.approx(raw)


@pytest.mark.parametrize(
    ("register", "label", "softened"),
    [
        ("fallback", "temperature:temperature_fallback", True),
        ("other_question", "identity:identity", False),
    ],
)
async def test_calibrator_resolution_is_reported(register: str, label: str, softened: bool) -> None:
    backend = fake({"Which color?": letters(0.1, 0.7, 0.2)})
    registry = CalibratorRegistry()
    emulator = Emulator(backend, calibrators=registry, include_diagnostics=True)
    if register == "fallback":
        registry.register_fallback(
            "fake-model", emulator.renderer.template_id, TemperatureCalibrator(3.0)
        )
    else:
        registry.register(_key(emulator, URGENT), TemperatureCalibrator(3.0))

    response = await emulator.system_one(request({"color": COLOR}))

    assert response.x_jevemu is not None
    assert response.x_jevemu["color"].calibrator == label
    answer = response.answers["color"]
    assert isinstance(answer, ChoiceAnswer)
    assert (answer.probabilities["green"] < 0.69) is softened


async def test_rounding_reports_jev_precision_but_chooses_on_unrounded_probabilities() -> None:
    backend = fake(
        {
            "Which color?": letters(0.446, 0.454, 0.100),
            "How severe?": digits(0.123, 0.456, 0.421),
            "Is it urgent?": yes(0.9876),
        }
    )
    response = await Emulator(backend, round_to=0.01).system_one(
        request({"color": COLOR, "severity": SEVERITY, "urgent": URGENT})
    )

    color = response.answers["color"]
    assert isinstance(color, ChoiceAnswer)
    assert color.probabilities == {"red": 0.45, "green": 0.45, "blue": 0.1}
    assert color.choice == "green"  # the unrounded argmax, not the first of the rounded tie
    assert color.confidence == 0.18  # (3 * 0.454 - 1) / 2 = 0.181

    severity = response.answers["severity"]
    assert isinstance(severity, ScoreAnswer)
    assert severity.probabilities == {"0": 0.12, "1": 0.46, "2": 0.42}
    assert severity.score == 1.3  # 0.456 + 2 * 0.421 = 1.298, from unrounded probabilities

    urgent = response.answers["urgent"]
    assert isinstance(urgent, NoulAnswer)
    assert urgent.noul == 0.99


async def test_diagnostics_describe_each_question() -> None:
    backend = fake({"Which color?": letters(0.2, 0.5, 0.3), "Is it urgent?": yes(0.8)})
    emulator = Emulator(backend, include_diagnostics=True)
    response = await emulator.system_one(request({"color": COLOR, "urgent": URGENT}))

    assert response.x_jevemu is not None
    assert set(response.x_jevemu) == {"color", "urgent"}
    color = response.x_jevemu["color"]
    assert (color.backend, color.backend_model, color.vllm_version) == (
        "fake",
        "fake-model",
        "fake",
    )
    assert (color.strategy, color.label_scheme, color.permutations) == (
        "constrained",
        "letters",
        1,
    )
    assert color.calibrator is None
    assert color.n_backend_calls == 1
    assert color.missing_labels == []
    assert color.observed_mass == pytest.approx(1.0)
    assert list(color.raw_probabilities.values()) == pytest.approx([0.2, 0.5, 0.3])
    assert color.latency_ms >= 0.0
    assert response.x_jevemu["urgent"].label_scheme == "yes_no"
    assert emulator.info is not None
    assert emulator.info.model == "fake-model"


async def test_diagnostics_are_omitted_by_default() -> None:
    backend = fake({"Is it urgent?": yes(0.8)})
    response = await Emulator(backend).system_one(request({"urgent": URGENT}))
    assert response.x_jevemu is None
    assert "x_jevemu" not in response.model_dump(exclude_none=True)


async def test_usage_counts_prompt_tokens_and_one_generated_token_per_first_token_call() -> None:
    backend = fake(
        {
            "Which color?": letters(0.2, 0.5, 0.3),
            "How severe?": digits(0.1, 0.3, 0.6),
            "Is it urgent?": yes(0.8),
        }
    )
    response = await Emulator(backend).system_one(
        request({"color": COLOR, "severity": SEVERITY, "urgent": URGENT})
    )
    reads = [args["prompt"] for method, args in backend.calls if method == "next_token_logprobs"]
    assert len(reads) == 3
    # FakeBackend counts one prompt token per character of every message.
    assert response.usage.input_tokens == sum(
        len(message.content) for prompt in reads for message in prompt.messages
    )
    assert response.usage.output_tokens == 3


async def test_echo_generates_no_tokens() -> None:
    backend = FakeBackend(
        sequences=lambda prefix, continuation: [(continuation, -1.0)],
        vocab=VOCAB,
    )
    response = await Emulator(backend, strategy=EchoStrategy()).system_one(
        request({"color": COLOR})
    )
    assert response.usage.output_tokens == 0
    assert response.usage.input_tokens > 0


async def test_questions_never_see_each_other() -> None:
    fruit = ChoiceQuestion(instructions="Which fruit?", criteria={"apple": None, "pear": None})
    shade = ChoiceQuestion(instructions="Which shade?", criteria={"crimson": None, "teal": None})
    ripe = NoulQuestion(instructions="Is it ripe?", criteria={"true": "soft", "false": "hard"})
    markers = {"Which fruit?": letters(0.6, 0.4), "Which shade?": letters(0.3, 0.7)}
    backend = fake({**markers, "Is it ripe?": yes(0.5)})

    await Emulator(backend).system_one(request({"fruit": fruit, "shade": shade, "ripe": ripe}))

    owned = {
        "Which fruit?": ("apple", "pear"),
        "Which shade?": ("crimson", "teal"),
        "Is it ripe?": ("soft", "hard"),
    }
    reads = [args["prompt"] for method, args in backend.calls if method == "next_token_logprobs"]
    assert len(reads) == 3
    for prompt in reads:
        text = prompt_text(prompt)
        (mine,) = [marker for marker in owned if marker in text]
        foreign = [word for marker, words in owned.items() if marker != mine for word in words]
        assert all(word in text for word in owned[mine])
        assert not [word for word in foreign if word in text]


class _TrackingStrategy:
    """Constrained scoring that records how many questions are in flight at once."""

    name = "tracking"

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self._inner = ConstrainedStrategy()

    def supports(self, capabilities: object, question: Question) -> bool:
        return True

    async def score(
        self, backend: Backend, renderer: PromptRenderer, state: JSONish, question: Question
    ) -> ScoreResult:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.01)
            return await self._inner.score(backend, renderer, state, question)
        finally:
            self.active -= 1


async def test_concurrency_bound_holds_across_requests() -> None:
    backend = fake({"Is it urgent?": yes(0.8)})
    strategy = _TrackingStrategy()
    emulator = Emulator(backend, strategy=strategy, max_concurrency=2)
    questions = {f"q{i}": URGENT for i in range(4)}

    responses = await asyncio.gather(
        emulator.system_one(request(questions)), emulator.system_one(request(questions))
    )

    assert strategy.peak == 2
    assert all(set(r.answers) == set(questions) for r in responses)


async def test_warm_up_runs_once_for_concurrent_first_requests() -> None:
    backend = fake({"Is it urgent?": yes(0.8)})
    emulator = Emulator(backend)
    await asyncio.gather(*(emulator.system_one(request({"u": URGENT})) for _ in range(3)))
    assert [method for method, _ in backend.calls].count("health") == 1


async def test_a_failing_question_fails_the_request_and_is_named() -> None:
    def next_token(prompt: RenderedPrompt, allowed: Sequence[str] | None) -> Table:
        if "BROKEN" in prompt_text(prompt):
            raise RuntimeError("backend exploded")
        return yes(0.5)

    backend = FakeBackend(next_token=next_token, vocab=VOCAB)
    broken = NoulQuestion(instructions="BROKEN question")
    with pytest.raises(QuestionError, match=r"'bad'.*backend exploded") as info:
        await Emulator(backend).system_one(request({"good": URGENT, "bad": broken}))
    assert info.value.question_id == "bad"
    assert isinstance(info.value.__cause__, RuntimeError)


async def test_labels_without_a_single_token_surface_are_reported() -> None:
    vocab = [token for token in VOCAB if token not in {"Yes", " Yes", "No", " No"}]
    backend = FakeBackend(next_token=by_marker({"Is it urgent?": yes(0.8)}), vocab=vocab)
    emulator = Emulator(backend)
    await emulator.warm_up()
    assert emulator.unreadable_labels == ("Yes", "No")


def test_sync_calls_work_repeatedly() -> None:
    emulator = Emulator(fake({"Is it urgent?": yes(0.8)}))
    first = emulator.system_one_sync(request({"u": URGENT}))
    second = emulator.system_one_sync(request({"u": URGENT}))
    assert first.answers == second.answers


def _generic_table(prompt: RenderedPrompt, allowed: Sequence[str] | None) -> Table:
    """A skewed distribution over every letter, digit and Yes/No surface."""
    table = {" " + label: -0.1 * i for i, label in enumerate(CHOICE_LABELS)}
    table.update({digit: -0.3 * int(digit) for digit in string.digits})
    table.update({" Yes": -0.2, " No": -1.7})
    return table


DOC_EXAMPLES = {path.name: path.read_text() for path in DOC_REQUESTS}


@pytest.mark.parametrize("name", sorted(DOC_EXAMPLES))
async def test_every_doc_example_gets_a_valid_response(name: str) -> None:
    req = SystemOneRequest.model_validate_json(DOC_EXAMPLES[name])
    backend = FakeBackend(next_token=_generic_table, vocab=VOCAB)
    response = await Emulator(backend, include_diagnostics=True).system_one(req)

    wire = json.loads(response.model_dump_json())
    assert SystemOneResponse.model_validate(wire) == response
    assert list(response.answers) == list(req.questions)
    for question_id, question in req.questions.items():
        answer = response.answers[question_id]
        assert answer.type == question.type
        if isinstance(answer, NoulAnswer):
            continue
        assert math.fsum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-9)
        if isinstance(answer, ChoiceAnswer):
            assert isinstance(question, ChoiceQuestion)
            assert list(answer.probabilities) == list(question.criteria)
