"""Replays real vLLM answers (``emulator_vllm.jsonl``) through the emulator for Jev doc examples.

Recorded from vLLM 0.30.0 / Qwen3-0.6B with ``record_emulator_fixtures.py``. A ``FixtureMissing``
error (wrapped in ``QuestionError``) means the emulator now sends a different backend request:
re-record both files and review the diff of ``expected.json``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import pytest
from emulator_golden_cases import EXPECTED, FIXTURE, REQUESTS, load_request, make_emulator

from jevemu.backends.recorded import RecordedBackend
from jevemu.types import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse

EXPECTED_RESPONSES: dict[str, Any] = json.loads(EXPECTED.read_text("utf-8"))


def approx_floats(value: Any) -> Any:
    """``value`` with every float replaced by a tight ``pytest.approx``, at any depth."""
    if isinstance(value, float):
        return pytest.approx(value, rel=1e-12, abs=1e-15)
    if isinstance(value, dict):
        return {k: approx_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [approx_floats(v) for v in value]
    return value


def test_every_request_has_an_expected_response() -> None:
    assert sorted(EXPECTED_RESPONSES) == sorted(REQUESTS)


@pytest.mark.parametrize("name", REQUESTS)
async def test_replay_is_a_valid_and_stable_response(name: str) -> None:
    request = load_request(name)
    response = await make_emulator(RecordedBackend(FIXTURE)).system_one(request)

    body = json.loads(response.model_dump_json())
    assert SystemOneResponse.model_validate(body) == response
    assert list(response.answers) == list(request.questions)
    for answer in response.answers.values():
        if isinstance(answer, NoulAnswer):
            continue
        assert math.fsum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-12)
        if isinstance(answer, ChoiceAnswer):
            assert answer.probabilities[answer.choice] == max(answer.probabilities.values())
        if isinstance(answer, ScoreAnswer):
            expected_score = math.fsum(int(k) * p for k, p in answer.probabilities.items())
            assert answer.score == pytest.approx(expected_score)

    for diagnostics in body["x_jevemu"].values():
        assert diagnostics.pop("latency_ms") >= 0.0
    assert body == approx_floats(EXPECTED_RESPONSES[name])


def test_fixture_records_the_server_identity() -> None:
    header = json.loads(FIXTURE.read_text("utf-8").splitlines()[0])
    info = header["info"]
    assert info["backend"] == "vllm_http"
    assert info["engine_version"] == "0.30.0"
    assert re.fullmatch(r"[0-9a-f]{40}", info["model_revision"])
    assert re.fullmatch(r"vllm/vllm-openai:v[\d.]+@sha256:[0-9a-f]{64}", info["flags"]["image"])
    assert header["capabilities"]["mask_reflected_in_logprobs"] is True
