"""Replays real vLLM answers (``scoring_vllm.jsonl``) through every scoring strategy.

Recorded from vLLM 0.30.0 / Qwen3-0.6B with ``record_scoring_fixtures.py``. A ``FixtureMissing``
error means a strategy now sends a different request (or a prompt template changed): re-record
both files and review the diff of ``expected.json``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import pytest
from scoring_golden_cases import CASES, EXPECTED, FIXTURE, pairs, strategies

from jevemu.backends.recorded import RecordedBackend
from jevemu.render import PromptRenderer, sequence_scheme_for
from jevemu.scoring import ScoreResult

EXPECTED_RESULTS: dict[str, Any] = json.loads(EXPECTED.read_text("utf-8"))


async def replay(case: str, name: str) -> ScoreResult:
    layout, state, question, _ = CASES[case]
    strategy = strategies()[name]
    return await strategy.score(RecordedBackend(FIXTURE), PromptRenderer(layout), state, question)


def test_every_pair_has_an_expected_result() -> None:
    assert sorted(EXPECTED_RESULTS) == sorted(f"{c}/{n}" for c, n in pairs())


@pytest.mark.parametrize(("case", "name"), pairs(), ids=[f"{c}/{n}" for c, n in pairs()])
async def test_replay_is_a_valid_and_stable_distribution(case: str, name: str) -> None:
    result = await replay(case, name)
    question = CASES[case][2]
    assert result.keys == sequence_scheme_for(question).keys  # question order, every scheme
    probabilities = [result.probabilities[k] for k in result.keys]
    assert all(p >= 0.0 and not math.isnan(p) for p in probabilities)
    assert math.fsum(probabilities) == pytest.approx(1.0, abs=1e-12)
    assert 0.0 <= result.observed_mass <= 1.0

    expected = EXPECTED_RESULTS[f"{case}/{name}"]
    assert result.strategy == expected["strategy"]
    assert list(result.keys) == expected["keys"]
    assert probabilities == pytest.approx(expected["probabilities"], rel=1e-12, abs=1e-15)
    assert result.observed_mass == pytest.approx(expected["observed_mass"], rel=1e-12)
    assert list(result.missing) == expected["missing"]
    assert result.truncated == expected["truncated"]
    assert result.n_backend_calls == expected["n_backend_calls"]
    assert list(result.warnings) == expected["warnings"]


@pytest.mark.parametrize(
    ("case", "answer"),
    [("choice_easy", "fruit"), ("noul_human", "true"), ("choice_60_countries", "France")],
)
async def test_every_strategy_gets_the_easy_questions_right(case: str, answer: str) -> None:
    for name in CASES[case][3] or strategies():
        result = await replay(case, name)
        assert max(result.probabilities, key=result.probabilities.__getitem__) == answer, name


async def test_digit_reads_follow_the_models_path_through_the_space() -> None:
    # The first-token strategies read digits after "Answer: "; echo scores " 3" as " " + "3".
    # Forcing the bare digit right after "Answer:" read 0.976 for "3" on this prompt.
    echo = (await replay("score_damage", "echo_sum")).probabilities
    for name in ("first_token", "constrained", "trie"):
        read = (await replay("score_damage", name)).probabilities
        assert max(abs(read[k] - echo[k]) for k in echo) < 0.06, name


async def test_trie_resolves_a_label_that_prefixes_other_labels() -> None:
    # "Beaver" prefixes "Beaver Dam Logistics": echo's open-ended sum lets the short option
    # absorb the long one, while the trie scores the end of "Beaver" explicitly.
    trie = (await replay("choice_prefix_keys", "trie")).probabilities
    echo = (await replay("choice_prefix_keys", "echo_sum")).probabilities
    assert trie["Beaver Dam Logistics"] > 0.99
    assert echo["Beaver"] > 0.4


def test_fixture_records_the_server_identity() -> None:
    header = json.loads(FIXTURE.read_text("utf-8").splitlines()[0])
    info = header["info"]
    assert info["backend"] == "vllm_http"
    assert info["engine_version"] == "0.30.0"
    assert re.fullmatch(r"[0-9a-f]{40}", info["model_revision"])
    assert re.fullmatch(r"vllm/vllm-openai:v[\d.]+@sha256:[0-9a-f]{64}", info["flags"]["image"])
    assert header["capabilities"]["mask_reflected_in_logprobs"] is True
