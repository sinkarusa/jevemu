"""The emulator and the typesafe-sdk facade against a live vLLM server (``scripts/serve_vllm.sh``).

Skipped unless ``JEVEMU_VLLM_URL`` is set; ``JEVEMU_VLLM_MODEL`` (default Qwen/Qwen3-0.6B) and
``JEVEMU_VLLM_MAX_LOGPROBS`` (default 576) must match the server launch. The emulator probes the
server's capabilities at warm-up (cached per server identity).
"""

from __future__ import annotations

import math
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from jevemu.backends.vllm_http import VLLMHTTPBackend
from jevemu.compat.typesafe import AsyncTypeSafeClient, Choice, Noul, Score
from jevemu.emulator import Emulator
from jevemu.types import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
)

URL = os.environ.get("JEVEMU_VLLM_URL", "")
MODEL = os.environ.get("JEVEMU_VLLM_MODEL", "Qwen/Qwen3-0.6B")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not URL, reason="set JEVEMU_VLLM_URL to run live vLLM tests"),
]

DOC_REQUESTS = {
    path.name.removesuffix("__request.json"): path.read_text(encoding="utf-8")
    for path in sorted(
        (Path(__file__).parents[1] / "golden" / "fixtures" / "jev_docs").glob("*__request.json")
    )
}


@pytest.fixture
async def backend() -> AsyncIterator[VLLMHTTPBackend]:
    async with VLLMHTTPBackend(URL, MODEL) as live:
        yield live


@pytest.mark.parametrize("name", sorted(DOC_REQUESTS))
async def test_every_doc_example_is_answered(backend: VLLMHTTPBackend, name: str) -> None:
    request = SystemOneRequest.model_validate_json(DOC_REQUESTS[name])
    emulator = Emulator(backend, include_diagnostics=True)
    response = await emulator.system_one(request)

    assert backend.capabilities.mask_reflected_in_logprobs  # probed at warm-up, not the default
    assert SystemOneResponse.model_validate_json(response.model_dump_json()) == response
    assert response.model == f"jevemu/{MODEL}"
    assert list(response.answers) == list(request.questions)
    assert response.x_jevemu is not None
    for question_id, question in request.questions.items():
        answer = response.answers[question_id]
        assert answer.type == question.type
        diagnostics = response.x_jevemu[question_id]
        assert diagnostics.vllm_version == "0.30.0"
        assert diagnostics.strategy == "constrained"  # every doc question fits the top-k
        if isinstance(answer, NoulAnswer):
            continue
        assert math.fsum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-9)
        if isinstance(answer, ChoiceAnswer):
            assert isinstance(question, ChoiceQuestion)
            assert list(answer.probabilities) == list(question.criteria)
            assert answer.probabilities[answer.choice] == max(answer.probabilities.values())
        if isinstance(answer, ScoreAnswer):
            assert isinstance(question, ScoreQuestion)
            assert 0.0 <= answer.score <= len(question.criteria) - 1
    # One constrained first-token read per question.
    assert response.usage.output_tokens == len(request.questions)
    assert response.usage.input_tokens > 0


async def test_sdk_quickstart_runs_against_the_server() -> None:
    async with AsyncTypeSafeClient(base_url=URL, backend_model=MODEL) as client:
        response = await client.system_one(
            state={"document": "I was charged twice. Please fix this ASAP."},
            questions={
                "billing": Noul(instructions="Is this ticket about billing?"),
                "tone": Choice(
                    instructions="What is the customer's tone?",
                    criteria={"calm": None, "frustrated": None, "angry": None},
                ),
                "urgency": Score(
                    instructions="How urgent is this ticket?",
                    criteria=["can wait", "this week", "today"],
                ),
            },
        )
        models = await client.models.list()

    # >=: Qwen3-0.6B's bf16 logits tie " Yes" and " No" exactly on this prompt (-0.6932 each).
    assert response.nouls["billing"].noul >= 0.5
    assert response.choices["tone"].choice in {"frustrated", "angry"}
    assert all(round(p, 2) == p for p in response.choices["tone"].probabilities.values())
    assert [m.name for m in models.models] == [response.model]
