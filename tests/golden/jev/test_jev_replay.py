"""Replays real Jev responses (``systemone/*.json``, ``models.json``) through ``JevClient``.

Fixtures are recorded by ``tests/live_jev/test_jev_live.py`` (``uv run pytest -m live_jev``):
the request body actually sent, the raw response, latency and date; never credentials.
``_schema.json`` snapshots the response shapes, so a re-recording fails here when Jev changes
its wire format.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest
import respx

from jevemu.cache import ResponseCache, canonical_json
from jevemu.errors import JevVersionDrift
from jevemu.jev_client import (
    API_KEY_ENV,
    JevClient,
    cost_usd,
    estimate_input_tokens,
    read_env_file,
)
from jevemu.types import DEFAULT_MODEL, SystemOneRequest

HERE = Path(__file__).parent
REPO = HERE.parents[2]
FIXTURES = sorted((HERE / "systemone").glob("*.json"))
BASE = "https://jev.test"


def load(path: Path) -> dict[str, Any]:
    fixture: dict[str, Any] = json.loads(path.read_text("utf-8"))
    return fixture


def client() -> JevClient:
    return JevClient("replay-key", base_url=BASE, cache=ResponseCache.in_memory())


def test_fixtures_cover_every_question_type() -> None:
    types = {answer["type"] for p in FIXTURES for answer in load(p)["response"]["answers"].values()}
    assert types == {"noul", "choice", "score"}


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
async def test_replay_is_pinned_billed_once_then_cached(path: Path) -> None:
    fixture = load(path)
    request = SystemOneRequest.model_validate(fixture["request"])
    with respx.mock(base_url=BASE) as router:
        route = router.post("/v1/systemone").respond(json=fixture["response"])
        jev = client()
        response, record = await jev.system_one_with_meta(request)
        replay, cached = await jev.system_one_with_meta(request)

    # The wire body is byte-stable, so cache keys built from it survive a re-run.
    assert route.calls[0].request.content.decode() == canonical_json(fixture["request"])
    assert route.call_count == 1
    assert response.model == DEFAULT_MODEL
    assert set(response.answers) == set(request.questions)
    assert record.cost_usd == pytest.approx(cost_usd(fixture["response"]["usage"]["input_tokens"]))
    assert replay == response
    assert (cached.cached, cached.cost_usd) == (True, 0.0)


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
async def test_drifted_replay_raises(path: Path) -> None:
    fixture = load(path)
    drifted = {**fixture["response"], "model": "jev-1.14.0"}
    with respx.mock(base_url=BASE) as router:
        router.post("/v1/systemone").respond(json=drifted)
        with pytest.raises(JevVersionDrift):
            await client().system_one(SystemOneRequest.model_validate(fixture["request"]))


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_budget_estimate_bounds_billed_tokens(path: Path) -> None:
    fixture = load(path)
    billed = fixture["response"]["usage"]["input_tokens"]
    assert estimate_input_tokens(canonical_json(fixture["request"])) >= billed


async def test_models_catalogue_replays() -> None:
    fixture = load(HERE / "models.json")
    with respx.mock(base_url=BASE) as router:
        router.get("/v1/models").respond(json=fixture["response"])
        models = await client().list_models()
    assert {"jev-latest", "jev-preview"} <= {m.name for m in models}


def test_fixtures_hold_no_credentials() -> None:
    key = os.environ.get(API_KEY_ENV) or read_env_file(REPO / ".env").get(API_KEY_ENV)
    for path in [*FIXTURES, HERE / "models.json"]:
        text = path.read_text("utf-8")
        assert not re.search(r"(?i)bearer|authorization|api[_-]?key", text), path.name
        if key:
            assert key not in text, path.name


def _answer_shapes() -> dict[str, Any]:
    top: set[str] = set()
    usage: set[str] = set()
    answers: dict[str, set[str]] = {}
    for path in FIXTURES:
        response = load(path)["response"]
        top |= set(response)
        usage |= set(response["usage"])
        for answer in response["answers"].values():
            answers.setdefault(answer["type"], set()).update(answer)
    return {
        "response": sorted(top),
        "usage": sorted(usage),
        "answers": {t: sorted(keys) for t, keys in sorted(answers.items())},
    }


def test_response_shapes_match_snapshot() -> None:
    """Drift guard: fails when re-recorded fixtures change any response field set."""
    assert _answer_shapes() == json.loads((HERE / "_schema.json").read_text("utf-8"))
