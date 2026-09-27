"""Boundary and error behavior of the Jev wire-format types."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from jevemu.types import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
)


def _request(question: dict[str, Any], **top: Any) -> dict[str, Any]:
    return {"state": "some state", "questions": {"q": question}, **top}


def _choice(n_options: int, **fields: Any) -> dict[str, Any]:
    criteria = {f"opt{i}": f"option {i}" for i in range(n_options)}
    return {"type": "choice", "instructions": "Pick one", "criteria": criteria, **fields}


def _score(n_levels: int, **fields: Any) -> dict[str, Any]:
    criteria = [f"level {i}" for i in range(n_levels)]
    return {"type": "score", "instructions": "Rate it", "criteria": criteria, **fields}


def _noul(**fields: Any) -> dict[str, Any]:
    return {"type": "noul", "instructions": "Is it?", **fields}


QUESTION_MAKERS = [_noul, lambda **f: _choice(2, **f), lambda **f: _score(2, **f)]


def _score_answer(**fields: Any) -> dict[str, Any]:
    answer = {
        "type": "score",
        "score": 1.0,
        "legend": {"0": "low", "1": "mid", "2": "high"},
        "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0},
        "confidence": 1.0,
    }
    return answer | fields


def _choice_answer(**fields: Any) -> dict[str, Any]:
    answer = {
        "type": "choice",
        "choice": "a",
        "probabilities": {"a": 0.7, "b": 0.3},
        "confidence": 0.4,
    }
    return answer | fields


def _response(answer: dict[str, Any], **top: Any) -> dict[str, Any]:
    return {
        "model": "jev-1.13.0",
        "answers": {"q": answer},
        "usage": {"input_tokens": 10, "output_tokens": 2},
        **top,
    }


def _diagnostics(**fields: Any) -> dict[str, Any]:
    diagnostics = {
        "backend": "fake",
        "backend_model": "tiny",
        "vllm_version": "0.30.0",
        "strategy": "logprobs",
        "label_scheme": "letters",
        "observed_mass": 0.98,
        "raw_probabilities": {"a": 0.6, "b": 0.4},
        "n_backend_calls": 1,
        "latency_ms": 12.5,
    }
    return diagnostics | fields


def _round_trip(model_cls: type[Any], payload: dict[str, Any]) -> Any:
    return model_cls.model_validate(payload).model_dump(mode="json", exclude_unset=True)


# --- questions -------------------------------------------------------------------------------


@pytest.mark.parametrize("n_options", [2, 255])
def test_choice_option_count_in_range_accepted(n_options: int) -> None:
    SystemOneRequest.model_validate(_request(_choice(n_options)))


@pytest.mark.parametrize("n_options", [1, 256])
def test_choice_option_count_out_of_range_rejected(n_options: int) -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(_request(_choice(n_options)))


def test_choice_empty_option_key_rejected() -> None:
    question = _choice(2, criteria={"": "blank", "b": "bee"})
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(_request(question))


def test_choice_null_option_description_accepted() -> None:
    question = _choice(2, criteria={"a": None, "b": None})
    SystemOneRequest.model_validate(_request(question))


@pytest.mark.parametrize("n_levels", [2, 10])
def test_score_level_count_in_range_accepted(n_levels: int) -> None:
    SystemOneRequest.model_validate(_request(_score(n_levels)))


@pytest.mark.parametrize("n_levels", [1, 11])
def test_score_level_count_out_of_range_rejected(n_levels: int) -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(_request(_score(n_levels)))


def test_score_null_level_accepted() -> None:
    SystemOneRequest.model_validate(_request(_score(2, criteria=[None, "high"])))


@pytest.mark.parametrize(
    "criteria",
    [
        {"true": "yes", "false": "no"},
        {"true": None, "false": None},
    ],
)
def test_noul_criteria_accepted(criteria: dict[str, Any]) -> None:
    SystemOneRequest.model_validate(_request(_noul(criteria=criteria)))


@pytest.mark.parametrize(
    "criteria",
    [
        {"true": "yes", "maybe": "unsure"},
        {"True": "yes", "false": "no"},
        {"yes": "yes", "no": "no"},
    ],
)
def test_noul_criteria_unknown_key_rejected(criteria: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(_request(_noul(criteria=criteria)))


@pytest.mark.parametrize(
    "instructions",
    ["Is it?", {"question": "Is it?", "focus": ["x"]}, ["Is it?", "Look at x"], None],
)
@pytest.mark.parametrize("make_question", QUESTION_MAKERS)
def test_instructions_text_or_structure_accepted(instructions: Any, make_question: Any) -> None:
    SystemOneRequest.model_validate(_request(make_question(instructions=instructions)))


@pytest.mark.parametrize("instructions", [3, 1.5])
def test_instructions_number_rejected(instructions: Any) -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(_request(_noul(instructions=instructions)))


# --- request ---------------------------------------------------------------------------------


def test_request_empty_questions_rejected() -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "s", "questions": {}})


def test_request_unknown_top_level_field_rejected() -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(_request(_noul(), modle="jev-latest"))


@pytest.mark.parametrize("make_question", QUESTION_MAKERS)
def test_request_unknown_question_field_rejected(make_question: Any) -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(_request(make_question(temperature=0.5)))


@pytest.mark.parametrize("question_type", [None, "rank", "Choice"])
def test_request_missing_or_unknown_question_type_rejected(question_type: str | None) -> None:
    question = {"instructions": "Is it?"}
    if question_type is not None:
        question["type"] = question_type
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(_request(question))


@pytest.mark.parametrize("state", [42, 4.2])
def test_request_number_state_rejected(state: Any) -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": state, "questions": {"q": _noul()}})


def test_request_criteria_order_preserved() -> None:
    choice = _choice(2, criteria={"zeta": "z", "alpha": None, "mu": "m"})
    score = _score(2, criteria=["worst", "bad", "ok", "good", "best"])
    payload = {"state": "s", "questions": {"c": choice, "s": score}}

    dumped = _round_trip(SystemOneRequest, payload)

    assert list(dumped["questions"]["c"]["criteria"]) == ["zeta", "alpha", "mu"]
    assert dumped["questions"]["s"]["criteria"] == ["worst", "bad", "ok", "good", "best"]


# --- answers ---------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_noul_outside_unit_interval_rejected(bad: float) -> None:
    with pytest.raises(ValidationError):
        NoulAnswer.model_validate({"type": "noul", "noul": bad})


@pytest.mark.parametrize("good", [0.0, 1.0])
def test_noul_unit_interval_edges_accepted(good: float) -> None:
    NoulAnswer.model_validate({"type": "noul", "noul": good})


@pytest.mark.parametrize(
    "fields",
    [
        {"probabilities": {"a": 1.2, "b": -0.2}},
        {"probabilities": {"a": -0.1, "b": 1.1}, "choice": "b"},
        {"confidence": -0.01},
        {"confidence": 1.01},
    ],
)
def test_choice_answer_probability_or_confidence_outside_unit_interval_rejected(
    fields: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        ChoiceAnswer.model_validate(_choice_answer(**fields))


def test_choice_answer_choice_not_an_option_rejected() -> None:
    with pytest.raises(ValidationError):
        ChoiceAnswer.model_validate(_choice_answer(choice="c"))


@pytest.mark.parametrize(
    "legend",
    [
        {"1": "low", "2": "mid", "3": "high"},
        {"0": "low", "2": "mid", "3": "high"},
        {"0": "low", "1": "mid", "x": "high"},
        {"0": "low", "01": "mid", "2": "high"},
        {"1": "mid", "0": "low", "2": "high"},
        {"0": "only"},
        {str(i): f"level {i}" for i in range(11)},
    ],
    ids=["offset", "gap", "non-digit", "zero-padded", "out-of-order", "1-level", "11-level"],
)
def test_score_answer_legend_keys_not_level_indices_rejected(legend: dict[str, str]) -> None:
    probabilities = dict.fromkeys(legend, 0.0) | {next(iter(legend)): 1.0}
    answer = _score_answer(legend=legend, probabilities=probabilities, score=0.0)
    with pytest.raises(ValidationError):
        ScoreAnswer.model_validate(answer)


@pytest.mark.parametrize("n_levels", [2, 10])
def test_score_answer_legend_level_count_in_range_accepted(n_levels: int) -> None:
    legend = {str(i): f"level {i}" for i in range(n_levels)}
    probabilities = dict.fromkeys(legend, 0.0) | {str(n_levels - 1): 1.0}
    answer = _score_answer(legend=legend, probabilities=probabilities, score=n_levels - 1)
    ScoreAnswer.model_validate(answer)


def test_score_answer_null_legend_value_round_trips() -> None:
    answer = _score_answer(legend={"0": None, "1": "mid", "2": {"what": "high"}})

    assert _round_trip(ScoreAnswer, answer) == answer


@pytest.mark.parametrize(
    "probabilities",
    [
        {"0": 0.5, "1": 0.5},
        {"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.4},
        {"0": 0.2, "1": 0.3, "x": 0.5},
    ],
)
def test_score_answer_probability_keys_mismatching_legend_rejected(
    probabilities: dict[str, float],
) -> None:
    with pytest.raises(ValidationError):
        ScoreAnswer.model_validate(_score_answer(probabilities=probabilities, score=0.5))


@pytest.mark.parametrize(
    "fields", [{"confidence": 1.01}, {"probabilities": {"0": -0.1, "1": 1.1, "2": 0.0}}]
)
def test_score_answer_probability_or_confidence_outside_unit_interval_rejected(
    fields: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        ScoreAnswer.model_validate(_score_answer(**fields))


@pytest.mark.parametrize("score", [-0.01, 2.01])
def test_score_answer_score_outside_level_range_rejected(score: float) -> None:
    with pytest.raises(ValidationError):
        ScoreAnswer.model_validate(_score_answer(score=score))


@pytest.mark.parametrize("score", [0.0, 2.0])
def test_score_answer_score_at_level_range_edges_accepted(score: float) -> None:
    ScoreAnswer.model_validate(_score_answer(score=score))


@pytest.mark.parametrize(
    "usage",
    [{"input_tokens": -1, "output_tokens": 0}, {"input_tokens": 0, "output_tokens": -1}],
)
def test_response_negative_usage_rejected(usage: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        SystemOneResponse.model_validate(_response(_choice_answer(), usage=usage))


# --- response --------------------------------------------------------------------------------


def test_response_keeps_unknown_fields_on_round_trip() -> None:
    payload = _response(_choice_answer(rationale="because"), request_id="req_123")

    assert _round_trip(SystemOneResponse, payload) == payload


def test_response_answer_order_preserved() -> None:
    probabilities = {"b": 0.1, "c": 0.6, "a": 0.3}
    payload = _response(_choice_answer(choice="c", probabilities=probabilities))
    payload["answers"]["s"] = _score_answer()

    dumped = _round_trip(SystemOneResponse, payload)

    assert list(dumped["answers"]) == ["q", "s"]
    assert list(dumped["answers"]["q"]["probabilities"]) == ["b", "c", "a"]


@pytest.mark.parametrize(
    "diagnostics",
    [
        _diagnostics(),
        _diagnostics(permutations=6, missing_labels=["c"], calibrator="temp", cached_tokens=8),
    ],
)
def test_response_with_emulator_diagnostics_round_trips(diagnostics: dict[str, Any]) -> None:
    payload = _response(_choice_answer(), x_jevemu={"q": diagnostics})

    assert _round_trip(SystemOneResponse, payload) == payload


@pytest.mark.parametrize(
    "diagnostics",
    [
        _diagnostics(gpu="A100"),
        _diagnostics(observed_mass=1.5),
        _diagnostics(permutations=0),
    ],
)
def test_response_invalid_emulator_diagnostics_rejected(diagnostics: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SystemOneResponse.model_validate(_response(_choice_answer(), x_jevemu={"q": diagnostics}))
