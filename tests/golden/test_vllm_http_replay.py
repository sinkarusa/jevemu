"""Replays real vLLM request/response pairs (tests/golden/vllm/*.json) through VLLMHTTPBackend.

Each exchange is mocked with an exact request-body match, so a replay also fails if the adapter
starts sending something the recorded server never accepted. Re-record with
``scripts/record_vllm_fixtures.py``; after reviewing a response-shape change, refresh the drift
snapshot with ``JEVEMU_UPDATE_VLLM_SCHEMA=1 uv run pytest tests/golden/test_vllm_http_replay.py``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import respx

from jevemu.backends.base import ChatMessage, RenderedPrompt
from jevemu.backends.vllm_http import (
    DEFAULT_CAPABILITIES,
    LOGPROB_FLOOR,
    VLLMHTTPBackend,
    VLLMHTTPError,
)

FIXTURE_DIR = Path(__file__).parent / "vllm"
SCHEMA_PATH = FIXTURE_DIR / "_schema.json"
BASE = "http://vllm.test"


def load(name: str) -> dict[str, Any]:
    fixture: dict[str, Any] = json.loads((FIXTURE_DIR / f"{name}.json").read_text("utf-8"))
    return fixture


FIXTURE_NAMES = sorted(p.stem for p in FIXTURE_DIR.glob("*.json") if not p.stem.startswith("_"))


def prompt_of(fixture: dict[str, Any]) -> RenderedPrompt:
    spec = fixture["call"]["prompt"]
    return RenderedPrompt(
        messages=tuple(ChatMessage(m["role"], m["content"]) for m in spec["messages"]),
        template_id=spec["template_id"],
    )


def backend_for(fixture: dict[str, Any]) -> VLLMHTTPBackend:
    meta = fixture["meta"]
    caps = dataclasses.replace(DEFAULT_CAPABILITIES, **fixture["call"]["capabilities"])
    return VLLMHTTPBackend(
        BASE,
        meta["model"],
        capabilities=caps,
        image=meta["image"],
        model_revision=meta["model_revision"],
    )


@contextmanager
def replay(fixture: dict[str, Any]) -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE, assert_all_called=True) as router:
        for exchange in fixture["exchanges"]:
            request, response = exchange["request"], exchange["response"]
            match = {} if request["json"] is None else {"json": request["json"]}
            route = router.route(method=request["method"], path=request["path"], **match)
            if "json" in response:
                route.respond(response["status"], json=response["json"])
            else:
                route.respond(response["status"], text=response["text"])
        yield router


def response_json(fixture: dict[str, Any], path: str) -> list[dict[str, Any]]:
    return [e["response"]["json"] for e in fixture["exchanges"] if e["request"]["path"] == path]


def logsumexp(values: list[float]) -> float:
    peak = max(values)
    return peak + math.log(math.fsum(math.exp(v - peak) for v in values))


def first_token_top(raw: dict[str, Any]) -> dict[int, float]:
    """Recorded top-k by token id (the adapter asks for ``"token_id:N"`` keys)."""
    top: dict[str, float] = raw["choices"][0]["logprobs"]["top_logprobs"][0]
    return {int(key.removeprefix("token_id:")): value for key, value in top.items()}


async def test_unconstrained_first_token_parses_recorded_top_k() -> None:
    fixture = load("first_token")
    (raw,) = response_json(fixture, "/v1/completions")
    raw_top = first_token_top(raw)
    with replay(fixture):
        dist = await backend_for(fixture).next_token_logprobs(
            prompt_of(fixture), allowed=None, top_k=fixture["call"]["top_k"]
        )
    assert not dist.constrained
    assert len(dist.top) == fixture["call"]["top_k"]
    assert [t.logprob for t in dist.top] == sorted((t.logprob for t in dist.top), reverse=True)
    assert {(t.token_id, t.logprob) for t in dist.top} <= set(raw_top.items())
    assert (dist.sampled.token, dist.sampled.logprob) == (dist.top[0].token, dist.top[0].logprob)
    assert dist.prompt_tokens == raw["usage"]["prompt_tokens"]
    assert dist.cached_tokens == raw["usage"]["prompt_tokens_details"]["cached_tokens"]


async def test_constrained_first_token_maps_masked_floor_to_neg_inf() -> None:
    fixture = load("first_token_constrained")
    allowed = fixture["call"]["allowed"]
    (raw,) = response_json(fixture, "/v1/completions")
    raw_top = first_token_top(raw)
    floored = sum(v <= LOGPROB_FLOOR for v in raw_top.values())
    with replay(fixture):
        dist = await backend_for(fixture).next_token_logprobs(
            prompt_of(fixture), allowed=allowed, top_k=fixture["call"]["top_k"]
        )
    finite = [t for t in dist.top if math.isfinite(t.logprob)]
    assert dist.constrained
    assert floored > 0
    assert sum(t.logprob == -math.inf for t in dist.top) == floored
    # vLLM 0.30.0 applies the choice mask before computing raw logprobs, on /v1/completions too.
    assert sorted(t.token for t in finite) == sorted(allowed)
    assert logsumexp([t.logprob for t in finite]) == pytest.approx(0.0, abs=1e-4)
    assert dist.sampled.token in allowed
    assert dist.cached_tokens == raw["usage"]["prompt_tokens_details"]["cached_tokens"] > 0


async def test_top_logprobs_above_server_cap_raises_http_error() -> None:
    fixture = load("top_logprobs_above_cap_error")
    with replay(fixture), pytest.raises(VLLMHTTPError) as info:
        await backend_for(fixture).next_token_logprobs(
            prompt_of(fixture), allowed=None, top_k=fixture["call"]["top_k"]
        )
    assert info.value.status_code == 400
    assert "greater than max allowed: 64" in info.value.detail


async def test_echo_scores_only_continuation_tokens_in_input_order() -> None:
    fixture = load("echo_sequence_logprobs")
    continuations = fixture["call"]["continuations"]
    echoes = {
        tuple(e["request"]["json"]["prompt"]): e["response"]["json"]
        for e in fixture["exchanges"]
        if e["request"]["path"] == "/v1/completions"
    }
    with replay(fixture):
        scores = await backend_for(fixture).sequence_logprobs(prompt_of(fixture), continuations)
    assert [s.continuation for s in scores] == continuations
    by_text = {s.continuation: s for s in scores}
    for prompt, raw in echoes.items():
        tokens = raw["choices"][0]["logprobs"]["tokens"]
        values = raw["choices"][0]["logprobs"]["token_logprobs"]
        assert values[0] is None  # vLLM's null first prompt token never reaches a SeqScore
        assert len(tokens) == len(prompt)  # max_tokens=0: no generated token appended
        score = by_text[tokens[-1]]
        assert score.tokens == (score.continuation,)
        assert score.token_logprobs == (values[-1],)
    assert by_text[" Paris"].total > by_text[" London"].total


async def test_tokenize_detokenize_round_trip() -> None:
    tok, detok = load("tokenize"), load("detokenize")
    with replay(tok):
        ids = await backend_for(tok).tokenize(tok["call"]["text"])
    with replay(detok):
        text = await backend_for(detok).detokenize(ids)
    assert ids == detok["call"]["ids"]
    assert text == tok["call"]["text"]


async def test_health_reports_server_identity_and_discovered_flags() -> None:
    fixture = load("health")
    meta = fixture["meta"]
    with replay(fixture):
        info = await backend_for(fixture).health()
    assert (info.backend, info.model, info.engine_version) == (
        "vllm_http",
        meta["model"],
        meta["vllm_version"],
    )
    assert info.model_revision == meta["model_revision"]
    assert info.flags["image"] == meta["image"]
    assert info.flags["enable-prefix-caching"] == "true"  # from /metrics cache_config_info
    assert info.flags["block-size"] == "16"
    assert info.flags["max-model-len"] == "8192"
    assert info.flags["max-logprobs"] == "64"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_records_server_identity(name: str) -> None:
    meta = load(name)["meta"]
    assert meta["vllm_version"] == "0.30.0"
    assert meta["model"]
    assert re.fullmatch(r"[0-9a-f]{40}", meta["model_revision"])
    assert re.fullmatch(r"vllm/vllm-openai:v[\d.]+@sha256:[0-9a-f]{64}", meta["image"])


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_no_recorded_request_has_guided_fields(name: str) -> None:
    def keys(value: Any) -> Iterator[str]:
        if isinstance(value, dict):
            for key, inner in value.items():
                yield key
                yield from keys(inner)
        elif isinstance(value, list):
            for inner in value:
                yield from keys(inner)

    bodies = [e["request"]["json"] for e in load(name)["exchanges"]]
    assert not [k for k in keys(bodies) if k.startswith("guided_")]


def _shape(value: Any) -> Any:
    """JSON shape: object keys (token-id maps collapsed), merged list element shapes, types."""
    if isinstance(value, dict):
        if value and all(k.isdigit() or k.startswith("token_id:") for k in value):
            return {"<token_id>": _merge([_shape(v) for v in value.values()])}
        return {k: _shape(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return _merge([_shape(v) for v in value])
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def _merge(shapes: list[Any]) -> list[Any]:
    unique = {json.dumps(s, sort_keys=True): s for s in shapes}
    return [unique[k] for k in sorted(unique)]


def _fixture_shapes() -> dict[str, list[Any]]:
    shapes = {}
    for name in FIXTURE_NAMES:
        exchanges = sorted(
            load(name)["exchanges"],
            key=lambda e: (e["request"]["path"], json.dumps(e["request"]["json"], sort_keys=True)),
        )
        shapes[name] = [
            {
                "request": [e["request"]["method"], e["request"]["path"]],
                "status": e["response"]["status"],
                "response": _shape(e["response"].get("json", e["response"].get("text"))),
            }
            for e in exchanges
        ]
    return shapes


def test_response_shapes_match_snapshot() -> None:
    """Drift guard: fails when re-recorded fixtures change any response shape."""
    shapes = _fixture_shapes()
    if os.environ.get("JEVEMU_UPDATE_VLLM_SCHEMA") == "1":
        SCHEMA_PATH.write_text(json.dumps(shapes, indent=1, sort_keys=True) + "\n", "utf-8")
    assert shapes == json.loads(SCHEMA_PATH.read_text("utf-8"))
