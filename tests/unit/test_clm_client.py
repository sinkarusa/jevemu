from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx

from jevemu.bench.manifest import describe_system
from jevemu.cache import ResponseCache
from jevemu.clm_client import API_KEY_ENV, HEAD_SHA256, MODEL, ClmClient
from jevemu.errors import JevVersionDrift
from jevemu.jev_client import JevClient
from jevemu.types import ChoiceQuestion, SystemOneRequest

BASE = "http://clm.test"


@pytest.fixture(autouse=True)
def no_ambient_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE, assert_all_called=False) as mock:
        yield mock


def request() -> SystemOneRequest:
    return SystemOneRequest(
        state="My payouts failed.",
        questions={
            "dept": ChoiceQuestion(instructions="Which team?", criteria={"a": "A", "b": "B"})
        },
    )


def clm_reply(model: str = MODEL, tokens: int = 812) -> dict[str, Any]:
    """The shape of upstream ``clm.engine.Engine.answer`` (unrounded probabilities)."""
    return {
        "model": model,
        "answers": {
            "dept": {
                "type": "choice",
                "choice": "a",
                "confidence": 0.4127,
                "probabilities": {"a": 0.7063514, "b": 0.2936486},
            }
        },
        "usage": {"billing_units": 1, "input_tokens": tokens, "output_tokens": 0},
    }


@pytest.mark.parametrize("model", ["clm-latest", "clm-raw", "jev-1.13.0"])
def test_unpinned_models_are_refused(model: str) -> None:
    with pytest.raises(ValueError, match="versioned CLM"):
        ClmClient(model=model, cache=ResponseCache.in_memory())


async def test_keyless_call_is_free_and_pinned(router: respx.MockRouter) -> None:
    route = router.post("/v1/systemone").mock(return_value=httpx.Response(200, json=clm_reply()))
    async with ClmClient(base_url=BASE, cache=ResponseCache.in_memory()) as clm:
        response, record = await clm.system_one_with_meta(request())
        _, replay = await clm.system_one_with_meta(request())

    call = route.calls.last.request
    assert "authorization" not in call.headers
    assert json.loads(call.content)["model"] == MODEL
    assert response.usage.input_tokens == 812
    assert (record.cached, record.cost_usd, clm.spent_usd) == (False, 0.0, 0.0)
    assert (replay.cached, route.call_count) == (True, 1)


async def test_key_from_environment_is_sent(
    router: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_ENV, "clm-secret")
    route = router.post("/v1/systemone").mock(return_value=httpx.Response(200, json=clm_reply()))
    async with ClmClient(base_url=BASE, cache=ResponseCache.in_memory()) as clm:
        await clm.system_one(request())
    assert route.calls.last.request.headers["authorization"] == "Bearer clm-secret"


async def test_shared_cache_never_crosses_jev_and_clm(router: respx.MockRouter) -> None:
    jev_body = {**clm_reply(model="jev-1.13.0"), "usage": {"input_tokens": 300, "output_tokens": 0}}
    router.post("/v1/systemone").mock(
        side_effect=[httpx.Response(200, json=jev_body), httpx.Response(200, json=clm_reply())]
    )
    cache = ResponseCache.in_memory()
    async with JevClient("tsk-test", base_url=BASE, cache=cache) as jev:
        await jev.system_one(request())
    async with ClmClient(base_url=BASE, cache=cache) as clm:
        response, record = await clm.system_one_with_meta(request())
    assert (response.model, record.cached) == (MODEL, False)


def models(*names: str) -> httpx.Response:
    entries = [{"name": n, "description": "d", "release_date": "2026-09-19"} for n in names]
    return httpx.Response(200, json={"models": entries})


async def test_manifest_requires_the_pinned_model_to_be_served(router: respx.MockRouter) -> None:
    router.get("/v1/models").mock(side_effect=[models(MODEL, "clm-raw"), models("clm-latest")])
    async with ClmClient(base_url=BASE, cache=ResponseCache.in_memory()) as clm:
        description = await describe_system(clm)
        with pytest.raises(JevVersionDrift):
            await describe_system(clm)
    assert (description["kind"], description["model"]) == ("clm", MODEL)
    assert description["head"]["sha256"] == HEAD_SHA256
