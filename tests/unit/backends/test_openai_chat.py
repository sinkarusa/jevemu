"""OpenAIChatBackend over a mocked Chat Completions API: the structured read, requests,
accounting, cache, budget, retries, and a runner end to end (Emulator + backend + run_split)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from jevemu.backends.chat_api import parse_chat_response
from jevemu.backends.openai_chat import OpenAIChatBackend
from jevemu.bench import BenchItem, BenchmarkSpec
from jevemu.bench.manifest import RunManifest
from jevemu.bench.runner import load_records, records_path, run_split
from jevemu.cache import ResponseCache
from jevemu.emulator import Emulator
from jevemu.errors import ChatAPIBudgetExceeded, ChatAPIQuotaExceeded, ChatAPIResponseError
from jevemu.eval.costs import GPT_6_LUNA, PriceBook, TokenUsage
from jevemu.jev_client import RetryPolicy
from jevemu.scoring import ConstrainedStrategy, FirstTokenStrategy
from jevemu.types import ChoiceAnswer, ChoiceQuestion, SystemOneRequest

KEY = "sk-test-secret"
QUESTION = ChoiceQuestion(instructions="Pick one.", criteria={"x": "ex", "y": "why", "z": None})
TOP = {"B": math.log(0.6), " B": math.log(0.05), "A": math.log(0.25), "**": math.log(0.08)}
"""First-token top logprobs the mock returns, fewer than requested as gpt-6-luna does: they hold
0.98 of the mass, so "C" is absent and can hold at most the leftover 0.02."""

Steps = list[tuple[str, dict[str, float]]]
"""A reply's tokens, each with its top logprobs (its own logprob is its entry there, else 0)."""


def structured(steps: Steps, *, reasoning_tokens: int = 0) -> dict[str, Any]:
    """A structured reply as gpt-6-luna gives it, one entry per ``steps`` token."""
    reply = completion()
    choice = reply["choices"][0]
    choice["message"]["content"] = "".join(token for token, _ in steps)
    choice["finish_reason"] = "stop"
    choice["logprobs"]["content"] = [
        {
            "token": token,
            "logprob": top.get(token, 0.0),
            "top_logprobs": [{"token": t, "logprob": v} for t, v in top.items()],
        }
        for token, top in steps
    ]
    reply["usage"]["completion_tokens"] = 11
    reply["usage"]["completion_tokens_details"]["reasoning_tokens"] = reasoning_tokens
    return reply


def luna_steps(value: dict[str, float]) -> Steps:
    """``{"answer":"<value>"}`` as gpt-6-luna tokenizes it, ``value`` the value token's top."""
    sampled = max(value, key=value.__getitem__)
    head: Steps = [(t, {t: 0.0}) for t in ('{"', "answer", '":"')]
    return [*head, (sampled, value), ('"}', {'"}': 0.0})]


def completion(
    *,
    model: str = "gpt-6-luna-2026-09-01",
    prompt_tokens: int = 1200,
    cached: int = 0,
    written: int = 1024,
    top: dict[str, float] = TOP,
) -> dict[str, Any]:
    sampled = max(top, key=top.__getitem__)
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": model,
        "system_fingerprint": "fp_test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": sampled},
                "finish_reason": "length",
                "logprobs": {
                    "content": [
                        {
                            "token": sampled,
                            "logprob": top[sampled],
                            "top_logprobs": [{"token": t, "logprob": v} for t, v in top.items()],
                        }
                    ]
                },
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 4,
            "total_tokens": prompt_tokens + 4,
            "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }


class Api:
    """Mock API: answers each request with ``respond(n, body)``, recording the bodies."""

    def __init__(self, respond: Callable[[int, dict[str, Any]], httpx.Response]) -> None:
        self.bodies: list[dict[str, Any]] = []
        self._respond = respond

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {KEY}"
        body = json.loads(request.content)
        self.bodies.append(body)
        return self._respond(len(self.bodies), body)


def ok(_: int, __: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=completion())


def backend(api: Api, **kwargs: Any) -> OpenAIChatBackend:
    kwargs.setdefault("cache", ResponseCache.in_memory())
    return OpenAIChatBackend(api_key=KEY, transport=httpx.MockTransport(api.handler), **kwargs)


async def ask(system: Emulator) -> tuple[ChoiceAnswer, Any]:
    response = await system.system_one(
        SystemOneRequest(state="some state", questions={"q": QUESTION})
    )
    answer = response.answers["q"]
    assert isinstance(answer, ChoiceAnswer)
    return answer, response


async def test_structured_read_sends_the_labels_as_an_enum_and_bounds_a_missing_one() -> None:
    # gpt-6-luna's top list at the value holds only labels, cut by probability: 0.98 listed.
    reply = structured(luna_steps({"B": math.log(0.7), "A": math.log(0.28)}))
    api = Api(lambda n, _: httpx.Response(200, json=reply))
    cache = ResponseCache.in_memory()
    async with backend(api, cache=cache) as luna:
        emulator = Emulator(luna, strategy=ConstrainedStrategy(), include_diagnostics=True)
        answer, response = await ask(emulator)

    (body,) = api.bodies
    assert body["messages"][-1]["role"] == "user"
    assert body["messages"][-1]["content"].endswith("Respond with the letter only.")
    assert body["reasoning_effort"] == "none"
    assert (body["max_completion_tokens"], body["top_logprobs"], body["logprobs"]) == (16, 5, True)
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string", "enum": ["A", "B", "C"]}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }
    assert "-noprefill-" in emulator.renderer.template_id

    # Under the mask the leftover 0.02 can only be "C"'s.
    assert list(answer.probabilities.values()) == pytest.approx([0.28, 0.7, 0.02])
    assert answer.choice == "y"
    diagnostics = response.x_jevemu["q"]
    assert diagnostics.strategy == "constrained"
    assert diagnostics.missing_labels == ["z"]
    assert diagnostics.n_backend_calls == 1
    usage = response.usage
    assert (usage.input_tokens, usage.output_tokens) == (1200, 11)
    expected = GPT_6_LUNA.cost(TokenUsage(1200, 11, cached_input_tokens=0, cache_write_tokens=1024))
    assert diagnostics.api.cost_usd == pytest.approx(expected)
    assert diagnostics.api.system_fingerprints == ["fp_test"]

    # A replay is served from the response cache: no key, no request, no spend.
    offline = OpenAIChatBackend(
        cache=cache, api_key=None, transport=httpx.MockTransport(api.handler)
    )
    replay = Emulator(offline, strategy=ConstrainedStrategy(), include_diagnostics=True)
    again, replayed = await ask(replay)
    assert again == answer
    assert len(api.bodies) == 1
    assert replayed.x_jevemu["q"].api.response_cache_hits == 1


async def test_a_merged_value_token_leaves_labels_behind_the_bare_quote_to_the_leftover() -> None:
    # The reply took '":"A' (quote and label in one token); "B" could only follow the bare '":"'
    # (0.3), so it may hold up to that much: more than the smallest listed label ("C", 0.1).
    head = {'":"A': math.log(0.6), '":"': math.log(0.3), '":"C': math.log(0.1)}
    steps: Steps = [('{"', {'{"': 0.0}), ("answer", {"answer": 0.0}), ('":"A', head)]
    api = Api(lambda n, _: httpx.Response(200, json=structured([*steps, ('"}', {'"}': 0.0})])))
    async with backend(api) as luna:
        emulator = Emulator(luna, strategy=ConstrainedStrategy(), include_diagnostics=True)
        answer, response = await ask(emulator)
    raw = [0.6, 0.3, 0.1]
    assert list(answer.probabilities.values()) == pytest.approx([p / sum(raw) for p in raw])
    assert response.x_jevemu["q"].missing_labels == ["y"]


async def test_emulator_reads_the_first_token_of_a_free_reply() -> None:
    api = Api(ok)
    async with backend(api) as luna:
        emulator = Emulator(luna, strategy=FirstTokenStrategy(), include_diagnostics=True)
        answer, response = await ask(emulator)
    (body,) = api.bodies
    assert "response_format" not in body
    # "B" and " B" merge; "C" gets the leftover mass, tighter than the smallest entry (0.05).
    raw = [0.25, 0.6 + 0.05, 0.02]
    assert list(answer.probabilities.values()) == pytest.approx([p / sum(raw) for p in raw])
    assert response.x_jevemu["q"].strategy == "first_token"


async def test_budget_refuses_before_sending() -> None:
    api = Api(ok)
    async with backend(api, max_usd=1e-7) as luna:
        emulator = Emulator(luna, strategy=FirstTokenStrategy())
        with pytest.raises(Exception, match="OpenAI budget") as info:
            await ask(emulator)
    assert isinstance(info.value.__cause__, ChatAPIBudgetExceeded)
    assert api.bodies == []


async def test_rate_limits_are_retried_after_the_servers_delay_but_exhausted_credit_is_not() -> (
    None
):
    def limited(n: int, _: dict[str, Any]) -> httpx.Response:
        if n == 1:
            return httpx.Response(429, headers={"retry-after-ms": "1500"}, json={"error": {}})
        return httpx.Response(200, json=completion())

    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    api = Api(limited)
    retry = RetryPolicy(initial_backoff_s=0.01)
    async with backend(api, retry=retry, sleep=sleep) as luna:
        await ask(Emulator(luna, strategy=FirstTokenStrategy()))
    assert len(api.bodies) == 2
    assert max(slept) >= 1.5

    broke = Api(
        lambda n, _: httpx.Response(
            429, json={"error": {"type": "insufficient_quota", "message": f"key {KEY} broke"}}
        )
    )
    async with backend(broke, retry=retry, sleep=sleep) as luna:
        with pytest.raises(Exception, match="429") as info:
            await ask(Emulator(luna, strategy=FirstTokenStrategy()))
    assert isinstance(info.value.__cause__, ChatAPIQuotaExceeded)
    assert KEY not in str(info.value.__cause__)
    assert len(broke.bodies) == 1


def test_float32_rounding_above_zero_reads_as_certainty_but_a_real_positive_is_rejected() -> None:
    # As gpt-6-luna reported a near-certain token on 2026-09-25.
    dist = parse_chat_response(completion(top={"J": 3.814697265625e-06}))
    assert dist.top[0].logprob == 0.0
    assert dist.sampled.logprob == 0.0
    with pytest.raises(ChatAPIResponseError, match="not a log-probability"):
        parse_chat_response(completion(top={"J": 0.5}))


async def test_an_empty_reply_is_resent_billed_and_never_cached() -> None:
    empty = completion()
    empty["choices"][0]["message"]["content"] = ""
    empty["choices"][0]["logprobs"]["content"] = []
    api = Api(lambda n, _: httpx.Response(200, json=empty if n == 1 else completion()))
    cache = ResponseCache.in_memory()
    async with backend(api, cache=cache) as luna:
        answer, response = await ask(
            Emulator(luna, strategy=FirstTokenStrategy(), include_diagnostics=True)
        )
        assert luna.empty_replies == 1
    assert answer.choice == "y"
    per_call = GPT_6_LUNA.cost(TokenUsage(1200, 4, cached_input_tokens=0, cache_write_tokens=1024))
    assert response.x_jevemu["q"].api.cost_usd == pytest.approx(2 * per_call)
    # Only the usable reply was cached: a replay needs no request.
    async with backend(api, cache=cache) as luna:
        again, _ = await ask(Emulator(luna, strategy=FirstTokenStrategy()))
    assert again == answer
    assert len(api.bodies) == 2


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        ({**structured(luna_steps({"B": 0.0})), "model": "gpt-6-astra"}, "pinned"),
        (structured(luna_steps({"B": 0.0}), reasoning_tokens=7), "reasoning"),
        # The value token's list is another position's (it lacks the token generated there).
        (structured([*luna_steps({"B": 0.0})[:3], ("B", {'":"': 0.0})]), "stale"),
    ],
    ids=["another-model", "reasoning", "stale-list"],
)
async def test_an_unusable_response_is_rejected_billed_and_not_cached(
    reply: dict[str, Any], error: str
) -> None:
    api = Api(lambda n, _: httpx.Response(200, json=reply))
    cache = ResponseCache.in_memory()
    async with backend(api, cache=cache) as luna:
        with pytest.raises(Exception, match=error) as info:
            await ask(Emulator(luna, strategy=ConstrainedStrategy()))
        assert luna.spent_usd > 0
    assert isinstance(info.value.__cause__, ChatAPIResponseError)
    assert len(cache) == 0


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


async def test_runner_stops_at_the_budget_resumes_from_the_cache_and_prices_by_tokens(
    tmp_path: Path,
) -> None:
    spec = _spec(12)
    api = Api(ok)
    cache = ResponseCache.in_memory()
    per_call = GPT_6_LUNA.cost(TokenUsage(1200, 4, cached_input_tokens=0, cache_write_tokens=1024))

    async def run(max_usd: float) -> Any:
        async with backend(api, cache=cache, max_usd=max_usd) as luna:
            emulator = Emulator(luna, strategy=FirstTokenStrategy(), include_diagnostics=True)
            return await run_split(emulator, "luna", spec, "select", tmp_path, concurrency=1)

    stopped = await run(3.5 * per_call)
    assert stopped.status == "stopped_budget"
    assert 0 < stopped.n_ok < stopped.n_items
    done = await run(1.0)
    assert done.status == "complete"
    assert len(api.bodies) == done.n_items  # nothing answered twice

    records = load_records(records_path(tmp_path, "luna", "toy", "select"))
    assert all(
        r.cost_usd == pytest.approx(per_call) and r.cached is False for r in records.values()
    )
    manifest = RunManifest.load(tmp_path / "luna")
    assert manifest is not None
    assert manifest.system["price_id"] == "gpt-6-luna"
    assert PriceBook.default().resolve(manifest.system) == GPT_6_LUNA
    assert manifest.totals.spent_usd == pytest.approx(done.n_items * per_call)
