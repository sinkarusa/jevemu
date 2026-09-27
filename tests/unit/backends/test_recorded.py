from __future__ import annotations

import asyncio
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import pytest

from jevemu.backends.base import ChatMessage, RenderedPrompt
from jevemu.backends.fake import DEFAULT_CAPABILITIES, FakeBackend
from jevemu.backends.recorded import (
    FixtureMissing,
    FixtureSchemaError,
    RecordedBackend,
    RecordingBackend,
)

PROMPT = RenderedPrompt(
    messages=(ChatMessage("user", "Is it?"), ChatMessage("assistant", "Answer:")),
    template_id="t@1",
)
CAPS = dataclasses.replace(DEFAULT_CAPABILITIES, top_logprobs_max=5)


def make_fake() -> FakeBackend:
    return FakeBackend(
        capabilities=CAPS,
        next_token={" Yes": -0.2, " No": -1.8, " Maybe": -math.inf},
        sequences={" yes": [(" yes", -0.3)], " never": [(" never", -math.inf)]},
        vocab=["Answer", ":", " ", "A"],
        model="m",
    )


async def record(path: Path) -> list[Any]:
    rec = RecordingBackend(make_fake(), path)
    return [
        await rec.next_token_logprobs(PROMPT, allowed=None, top_k=3),
        await rec.next_token_logprobs(PROMPT, allowed=[" Yes", " No"], top_k=3),
        await rec.sequence_logprobs(PROMPT, [" yes", " never"]),
        await rec.tokenize("Answer: A"),
        await rec.detokenize([0, 1]),
    ]


async def test_record_then_replay_returns_equal_results(tmp_path: Path) -> None:
    path = tmp_path / "fx.jsonl"
    live = await record(path)
    replay = RecordedBackend(path)
    replayed: list[Any] = [
        await replay.next_token_logprobs(PROMPT, allowed=None, top_k=3),
        await replay.next_token_logprobs(PROMPT, allowed=(" Yes", " No"), top_k=3),
        await replay.sequence_logprobs(PROMPT, (" yes", " never")),
        await replay.tokenize("Answer: A"),
        await replay.detokenize((0, 1)),
    ]
    assert replayed == live
    assert replayed[0].top[-1].logprob == -math.inf
    assert replayed[2][1].total == -math.inf
    assert replay.capabilities == CAPS
    assert await replay.health() == await make_fake().health()
    # The fixture must be strict JSON (no bare -Infinity tokens).
    for line in path.read_text().splitlines():
        json.loads(line, parse_constant=lambda c: pytest.fail(f"non-standard constant {c}"))


async def test_replay_miss_raises_fixture_missing(tmp_path: Path) -> None:
    path = tmp_path / "fx.jsonl"
    await record(path)
    replay = RecordedBackend(path)
    with pytest.raises(FixtureMissing) as info:
        await replay.next_token_logprobs(PROMPT, allowed=None, top_k=4)
    assert info.value.method == "next_token_logprobs"
    assert info.value.key in str(info.value)
    other_template = dataclasses.replace(PROMPT, template_id="t@2")
    with pytest.raises(FixtureMissing):
        await replay.next_token_logprobs(other_template, allowed=None, top_k=3)


async def test_schema_version_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "fx.jsonl"
    await record(path)
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    header["schema_version"] += 1
    path.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")
    with pytest.raises(FixtureSchemaError, match="schema_version"):
        RecordedBackend(path)


async def test_recording_appends_to_matching_fixture(tmp_path: Path) -> None:
    path = tmp_path / "fx.jsonl"
    await RecordingBackend(make_fake(), path).tokenize("A")
    await RecordingBackend(make_fake(), path).tokenize("Answer")
    replay = RecordedBackend(path)
    assert await replay.tokenize("A") == [3]
    assert await replay.tokenize("Answer") == [0]


async def test_recording_refuses_fixture_from_other_backend(tmp_path: Path) -> None:
    path = tmp_path / "fx.jsonl"
    await RecordingBackend(make_fake(), path).tokenize("A")
    other = FakeBackend(capabilities=DEFAULT_CAPABILITIES, vocab=["A"])
    with pytest.raises(FixtureSchemaError, match="header"):
        await RecordingBackend(other, path).tokenize("A")


async def test_concurrent_calls_are_all_recorded_whole_and_replayable(tmp_path: Path) -> None:
    path = tmp_path / "fx.jsonl"
    # Long prompts give multi-kilobyte records, so interleaved writes would corrupt lines.
    prompts = [
        RenderedPrompt(
            messages=(ChatMessage("user", f"{i} " + "x" * 5000), ChatMessage("assistant", "A:")),
            template_id=f"t@{i}",
        )
        for i in range(64)
    ]
    rec = RecordingBackend(
        FakeBackend(
            capabilities=CAPS,
            next_token=lambda prompt, _allowed: {prompt.template_id: -0.1, " No": -2.4},
            sequences={" yes": [(" yes", -0.3)], " no": [(" no", -1.2)]},
        ),
        path,
    )
    dists, scores = await asyncio.gather(
        asyncio.gather(*(rec.next_token_logprobs(p, allowed=None, top_k=2) for p in prompts)),
        asyncio.gather(*(rec.sequence_logprobs(p, [" yes", " no"]) for p in prompts)),
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + 2 * len(prompts)
    records = [json.loads(line) for line in lines]
    assert records[0]["kind"] == "header"
    assert all(record["kind"] == "call" for record in records[1:])

    replay = RecordedBackend(path)
    assert [await replay.next_token_logprobs(p, allowed=None, top_k=2) for p in prompts] == dists
    assert [await replay.sequence_logprobs(p, [" yes", " no"]) for p in prompts] == scores
    assert [dist.sampled.token for dist in dists] == [p.template_id for p in prompts]
