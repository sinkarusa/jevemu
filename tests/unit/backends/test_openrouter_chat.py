"""OpenRouterChatBackend over a mocked OpenRouter: provider routing, the structured read, the
price snapshot from the endpoints listing, billed spend, identity checks, and a runner end to
end."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from jevemu.backends.chat_api import AnswerFormat, parse_structured_response
from jevemu.backends.openrouter_chat import OpenRouterChatBackend
from jevemu.bench import BenchItem, BenchmarkSpec
from jevemu.bench.manifest import RunManifest
from jevemu.bench.runner import run_split
from jevemu.cache import ResponseCache
from jevemu.emulator import Emulator, QuestionError
from jevemu.errors import ChatAPIQuotaExceeded, ChatAPIResponseError
from jevemu.eval.costs import PriceBook
from jevemu.jev_client import RetryPolicy
from jevemu.scoring import ConstrainedStrategy
from jevemu.types import ChoiceAnswer, ChoiceQuestion, SystemOneRequest

KEY = "sk-or-test-secret"
MODEL = "deepseek/deepseek-v4.1-flash"
QUESTION = ChoiceQuestion(instructions="Pick one.", criteria={"x": "ex", "y": "why", "z": None})
QUOTE = {' "': math.log(0.8), " ": math.log(0.2)}
"""The value's opening quote: 0.2 of the mass on more whitespace first (a later value start)."""
VALUE = {"B": math.log(0.7), "A": math.log(0.2), "C": math.log(0.05), ",": -9999.0}
"""The value token's top logprobs; the mask lists "," at the provider's floor."""
BILLED = 0.000123


def endpoint(
    provider: str,
    tag: str,
    pricing: dict[str, Any],
    *,
    logprobs: bool = True,
) -> dict[str, Any]:
    params = ["max_tokens", "temperature", "reasoning"]
    return {
        "name": f"{provider} | {MODEL}-20260910",
        "provider_name": provider,
        "tag": tag,
        "pricing": pricing,
        "supported_parameters": params + (["logprobs", "top_logprobs"] if logprobs else []),
    }


def listing(*, wafer_prompt: str = "0.000000099") -> dict[str, Any]:
    return {
        "data": {
            "id": MODEL,
            "endpoints": [
                endpoint(
                    "Wafer",
                    "wafer",
                    {"prompt": wafer_prompt, "completion": "0.0000006", "input_cache_read": "6e-8"},
                ),
                endpoint(
                    "DeepSeek",
                    "deepseek",
                    {
                        "prompt": "0.00000015",
                        "completion": "0.0000006",
                        "input_cache_read": "0.000000003",
                        "overrides": [{"utc_start": 100, "prompt": "0.0000003"}],
                    },
                ),
                endpoint(
                    "DeepInfra",
                    "deepinfra/fp8",
                    {"prompt": "0.000009", "completion": "0.000009"},
                    logprobs=False,
                ),
            ],
        }
    }


Steps = list[tuple[str, dict[str, float]]]
"""A reply's tokens, each with its top logprobs (its own logprob is its entry there)."""

MAKORA_STEPS: Steps = [
    ("{", {"{": 0.0}),
    (' "', {' "': 0.0}),
    ("answer", {"answer": 0.0}),
    ('":', {'":': 0.0}),
    (' "', QUOTE),
    ("B", VALUE),
    ('"', {'"': 0.0}),
    (" }", {" }": 0.0}),
]
"""``{ "answer": "B" }`` as DeepSeek V4.1 Flash tokenizes it on Makora."""


def completion(
    *,
    provider: str = "Wafer",
    reasoning_tokens: int = 0,
    cost: float | None = BILLED,
    steps: Steps = MAKORA_STEPS,
) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": 900,
        "completion_tokens": 9,
        "total_tokens": 909,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }
    if cost is not None:
        usage["cost"] = cost
    return {
        "id": "gen-1",
        "object": "chat.completion",
        "model": f"{MODEL}-20260910",
        "provider": provider,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(t for t, _ in steps)},
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": token,
                            "logprob": top[token],
                            "top_logprobs": [{"token": t, "logprob": v} for t, v in top.items()],
                        }
                        for token, top in steps
                    ]
                },
            }
        ],
        "usage": usage,
    }


class Api:
    """Mock OpenRouter: answers each chat request with ``respond(n, body)``, recording bodies."""

    def __init__(self, respond: Callable[[int, dict[str, Any]], httpx.Response]) -> None:
        self.bodies: list[dict[str, Any]] = []
        self._respond = respond

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {KEY}"
        body = json.loads(request.content)
        self.bodies.append(body)
        return self._respond(len(self.bodies), body)


def ok(_: int, __: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=completion())


def backend(api: Api, **kwargs: Any) -> OpenRouterChatBackend:
    kwargs.setdefault("cache", ResponseCache.in_memory())
    kwargs.setdefault("endpoints", listing())
    return OpenRouterChatBackend(
        MODEL, api_key=KEY, transport=httpx.MockTransport(api.handler), **kwargs
    )


async def ask(system: Emulator) -> tuple[ChoiceAnswer, Any]:
    response = await system.system_one(
        SystemOneRequest(state="some state", questions={"q": QUESTION})
    )
    answer = response.answers["q"]
    assert isinstance(answer, ChoiceAnswer)
    return answer, response


async def test_pinned_structured_request_disables_reasoning_and_spends_what_was_billed() -> None:
    api = Api(ok)
    cache = ResponseCache.in_memory()
    async with backend(api, cache=cache, providers=["wafer"]) as router:
        emulator = Emulator(router, strategy=ConstrainedStrategy(), include_diagnostics=True)
        answer, response = await ask(emulator)
        info = await router.health()

    (body,) = api.bodies
    assert body["messages"][-1]["role"] == "user"
    assert body["reasoning"] == {"enabled": False}
    assert (body["max_tokens"], body["top_logprobs"], body["logprobs"]) == (16, 20, True)
    assert body["provider"] == {
        "require_parameters": True,
        "order": ["wafer"],
        "allow_fallbacks": False,
    }
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["answer"]["enum"] == ["A", "B", "C"]
    # Read given that the value starts after this ' "': the whitespace alternative is left out.
    assert list(answer.probabilities.values()) == pytest.approx(
        [p / 0.95 for p in (0.2, 0.7, 0.05)]
    )
    diagnostics = response.x_jevemu["q"]
    assert diagnostics.observed_mass == pytest.approx(0.95)
    assert diagnostics.api.cost_usd == pytest.approx(BILLED)
    assert diagnostics.api.providers == ["Wafer"]
    assert router.spent_usd == pytest.approx(BILLED)
    assert info.model_revision == f"{MODEL}-20260910"

    # A replay from the response cache needs no key and spends nothing.
    offline = OpenRouterChatBackend(
        MODEL, endpoints=listing(), providers=["wafer"], cache=cache, api_key=None
    )
    again, _ = await ask(Emulator(offline, strategy=ConstrainedStrategy()))
    assert again == answer
    assert len(api.bodies) == 1


def test_a_forced_quote_left_out_of_the_logprobs_still_locates_the_value() -> None:
    # Makora leaves tokens its grammar forces out of the logprobs: '{"answer":  \t"B"}' came back
    # as '{"', 'answer', '":', ' ', '\t', 'B', '"}'.
    steps: Steps = [
        ('{"', {'{"': 0.0}),
        ("answer", {"answer": 0.0}),
        ('":', {'":': 0.0}),
        (" ", {" ": 0.0}),
        ("\t", {"\t": math.log(0.9), '"': math.log(0.1)}),
        ("B", VALUE),
        ('"}', {'"}': 0.0}),
    ]
    dist = parse_structured_response(completion(steps=steps), AnswerFormat(("A", "B", "C")))
    assert [t.token for t in dist.top] == ["B", "A", "C"]
    assert [math.exp(t.logprob) for t in dist.top] == pytest.approx([0.7, 0.2, 0.05])
    assert dist.sampled.token == "B"


LOOP = [("\r", {"\r": 0.0}), ("\n", {"\n": 0.0})] * 6
"""Makora's whitespace loop under the schema: ``\\r\\n`` until ``max_tokens``."""


@pytest.mark.parametrize(
    "stuck",
    [
        pytest.param([("{\n\n", {"{\n\n": 0.0}), *LOOP], id="before-the-key"),
        pytest.param([*MAKORA_STEPS[:4], (" ", {" ": 0.0}), *LOOP], id="before-the-value"),
    ],
)
async def test_a_reply_stuck_before_its_value_is_sent_again_with_room_billed_and_not_cached(
    stuck: Steps,
) -> None:
    replies = [completion(steps=stuck), completion()]
    api = Api(lambda n, _: httpx.Response(200, json=replies[n - 1]))
    cache = ResponseCache.in_memory()
    async with backend(api, cache=cache, providers=["wafer"]) as router:
        emulator = Emulator(router, strategy=ConstrainedStrategy(), include_diagnostics=True)
        answer, response = await ask(emulator)

    first, resent = api.bodies
    assert (first["max_tokens"], resent["max_tokens"]) == (16, 64)
    assert {**resent, "max_tokens": 16} == first
    assert answer.choice == "y"
    diagnostics = response.x_jevemu["q"]
    assert diagnostics.n_backend_calls == 1
    assert diagnostics.api.cost_usd == pytest.approx(2 * BILLED)
    assert router.spent_usd == pytest.approx(2 * BILLED)
    assert router.empty_replies == 1
    assert len(cache) == 1


def test_price_is_the_pinned_endpoints_or_else_the_highest_over_providers_and_hours() -> None:
    cache = ResponseCache.in_memory()
    wafer = OpenRouterChatBackend(MODEL, endpoints=listing(), providers=["wafer"], cache=cache)
    assert wafer.price.input_usd_per_mtok == 0.099
    assert wafer.price.output_usd_per_mtok == 0.6
    assert wafer.price.cached_input_usd_per_mtok == 0.06
    assert wafer.price.price_id == f"openrouter:{MODEL}@wafer"

    # Unpinned: DeepSeek's peak-hour input rate; DeepInfra (no logprobs) is never routed to.
    anyone = OpenRouterChatBackend(MODEL, endpoints=listing(), cache=cache)
    assert anyone.price.input_usd_per_mtok == 0.3
    assert anyone.price.output_usd_per_mtok == 0.6
    assert anyone.price.cached_input_usd_per_mtok == 0.06

    with pytest.raises(ValueError, match="no OpenRouter endpoint"):
        OpenRouterChatBackend(MODEL, endpoints=listing(), providers=["deepinfra"], cache=cache)


async def test_a_response_from_an_unpinned_provider_or_with_reasoning_is_rejected() -> None:
    for bad in (completion(provider="DeepSeek"), completion(reasoning_tokens=12)):
        api = Api(lambda n, _, bad=bad: httpx.Response(200, json=bad))
        cache = ResponseCache.in_memory()
        async with backend(api, cache=cache, providers=["wafer"]) as router:
            with pytest.raises(QuestionError) as info:
                await ask(Emulator(router, strategy=ConstrainedStrategy()))
        assert isinstance(info.value.__cause__, ChatAPIResponseError)
        assert router.spent_usd == pytest.approx(BILLED)  # billed all the same
        assert len(cache) == 0


async def test_exhausted_credit_stops_without_retrying_or_leaking_the_key() -> None:
    def broke(_: int, __: dict[str, Any]) -> httpx.Response:
        return httpx.Response(402, json={"error": {"code": 402, "message": f"no credit {KEY}"}})

    api = Api(broke)
    async with backend(api, retry=RetryPolicy(max_attempts=4)) as router:
        with pytest.raises(QuestionError) as info:
            await ask(Emulator(router, strategy=ConstrainedStrategy()))
    assert isinstance(info.value.__cause__, ChatAPIQuotaExceeded)
    assert KEY not in str(info.value.__cause__)
    assert len(api.bodies) == 1


def _spec(n: int) -> BenchmarkSpec:
    source = {"revision": "0" * 40, "source_sha256": "1" * 64}
    items = [
        BenchItem(
            item_id=f"toy:{i}",
            state=f"toy state {i}",
            question=QUESTION,
            gold="y",
            metadata={"question_id": str(i), "stratum": "s", **source},
        )
        for i in range(n)
    ]
    return BenchmarkSpec("toy", "choice", lambda cfg: items, {"test": "test"}, "CC0")


async def test_runs_are_priced_by_the_recorded_snapshot_and_resume_after_a_repricing(
    tmp_path: Path,
) -> None:
    spec = _spec(4)
    api = Api(ok)
    cache = ResponseCache.in_memory()

    async def run(limit: int | None, endpoints: dict[str, Any]) -> Any:
        async with backend(api, cache=cache, providers=["wafer"], endpoints=endpoints) as router:
            emulator = Emulator(router, strategy=ConstrainedStrategy(), include_diagnostics=True)
            return await run_split(
                emulator, "or", spec, "select", tmp_path, concurrency=1, limit=limit
            )

    first = await run(1, listing())
    assert first.n_ok == 1
    done = await run(None, listing(wafer_prompt="0.0000002"))  # Wafer repriced meanwhile
    assert done.status == "complete"
    assert done.n_called == done.n_items - 1

    manifest = RunManifest.load(tmp_path / "or")
    assert manifest is not None
    price = PriceBook.default().resolve(manifest.system)
    assert price is not None
    assert price.price_id == f"openrouter:{MODEL}@wafer"
    assert price.input_usd_per_mtok == 0.099  # the first invocation's snapshot
    assert manifest.totals.spent_usd == pytest.approx(done.n_items * BILLED)


async def test_recorded_spend_includes_failed_attempts_and_passes(tmp_path: Path) -> None:
    spec = _spec(2)  # its select split holds one item
    stuck = completion(steps=[*MAKORA_STEPS[:4], (" ", {" ": 0.0}), *LOOP])
    # The first pass is stuck on all 3 attempts; the second answers.
    api = Api(lambda n, _: httpx.Response(200, json=stuck if n <= 3 else completion()))

    async def run() -> Any:
        async with backend(api, providers=["wafer"]) as router:
            emulator = Emulator(router, strategy=ConstrainedStrategy(), include_diagnostics=True)
            return await run_split(emulator, "or", spec, "select", tmp_path, concurrency=1)

    first = await run()
    assert (first.n_errors, first.spent_usd) == (1, pytest.approx(3 * BILLED))
    assert (await run()).status == "complete"

    manifest = RunManifest.load(tmp_path / "or")
    assert manifest is not None
    assert len(api.bodies) == 4
    assert manifest.totals.spent_usd == pytest.approx(4 * BILLED)
