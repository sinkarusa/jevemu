from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from jevemu.cache import ResponseCache, canonical_json
from jevemu.errors import (
    JevAPIKeyMissing,
    JevAuthError,
    JevBudgetExceeded,
    JevConnectionError,
    JevHTTPError,
    JevOverloaded,
    JevRateLimited,
    JevValidationError,
    JevVersionDrift,
)
from jevemu.jev_client import (
    API_KEY_ENV,
    JevCallRecord,
    JevClient,
    RetryPolicy,
    TokenBucket,
    cost_usd,
    estimate_input_tokens,
    load_env_file,
    parse_retry_after,
    probe_nondeterminism,
    read_env_file,
    wire_body,
)
from jevemu.types import ChoiceQuestion, NoulQuestion, SystemOneRequest

BASE = "https://jev.test"
KEY = "tsk-UNIT-TEST-SECRET-0123456789abcdef"
PINNED = "jev-1.13.0"


class FakeTime:
    """Monotonic clock that only moves when something sleeps on it."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def no_ambient_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)


@pytest.fixture
def ft() -> FakeTime:
    return FakeTime()


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE, assert_all_called=False) as mock:
        yield mock


def client(ft: FakeTime, **kwargs: Any) -> JevClient:
    options: dict[str, Any] = {
        "base_url": BASE,
        "cache": ResponseCache.in_memory(),
        "clock": ft.clock,
        "sleep": ft.sleep,
    }
    options.update(kwargs)
    return JevClient(options.pop("api_key", KEY), **options)


def choice_request(
    state: str = "My payouts failed.", order: tuple[str, ...] = ("a", "b")
) -> SystemOneRequest:
    return SystemOneRequest(
        state=state,
        questions={
            "dept": ChoiceQuestion(
                instructions="Which team?", criteria={o: f"option {o}" for o in order}
            )
        },
    )


def ok(
    *, model: str = PINNED, p_a: float = 0.7, noul: float | None = None, tokens: int = 300
) -> dict[str, Any]:
    answers: dict[str, Any] = {
        "dept": {
            "type": "choice",
            "choice": "a" if p_a >= 0.5 else "b",
            "probabilities": {"a": p_a, "b": round(1 - p_a, 2)},
            "confidence": round(abs(2 * p_a - 1), 2),
        }
    }
    if noul is not None:
        answers["urgent"] = {"type": "noul", "noul": noul}
    return {
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": tokens, "output_tokens": 30},
    }


def reply(status: int = 200, body: Any = None, **headers: str) -> httpx.Response:
    return httpx.Response(status, json=ok() if body is None else body, headers=headers)


def sent(route: respx.Route, i: int = -1) -> Any:
    return json.loads(route.calls[i].request.content)


def files_text(directory: Path) -> list[str]:
    return [p.read_bytes().decode("latin-1") for p in directory.iterdir()]


# --- retries ------------------------------------------------------------------------------------


async def test_429_waits_at_least_retry_after_then_succeeds(
    ft: FakeTime, router: respx.MockRouter
) -> None:
    route = router.post("/v1/systemone").mock(
        side_effect=[reply(429, {"detail": "slow down"}, **{"retry-after": "3"}), reply()]
    )
    jev = client(ft, retry=RetryPolicy(initial_backoff_s=1.0))
    response, record = await jev.system_one_with_meta(choice_request())

    assert route.call_count == 2
    assert record.attempts == 2
    assert response.answers["dept"].type == "choice"
    (backoff,) = ft.sleeps
    assert 3.0 <= backoff <= 3.5  # retry-after, plus at most half the first backoff as jitter


async def test_529_backs_off_exponentially(ft: FakeTime, router: respx.MockRouter) -> None:
    route = router.post("/v1/systemone").mock(side_effect=[reply(529), reply(529), reply()])
    jev = client(ft, retry=RetryPolicy(initial_backoff_s=1.0, max_backoff_s=60.0))
    _, record = await jev.system_one_with_meta(choice_request())

    assert route.call_count == 3
    assert record.attempts == 3
    first, second = ft.sleeps
    assert 0.5 <= first <= 1.0
    assert 1.0 <= second <= 2.0


@pytest.mark.parametrize(("status", "error"), [(429, JevRateLimited), (529, JevOverloaded)])
async def test_exhausted_retries_raise_typed_error_and_cost_nothing(
    ft: FakeTime, router: respx.MockRouter, status: int, error: type[JevHTTPError]
) -> None:
    route = router.post("/v1/systemone").respond(
        status, json={"detail": "busy"}, headers={"retry-after": "2"}
    )
    jev = client(ft, retry=RetryPolicy(max_attempts=3))
    with pytest.raises(error) as info:
        await jev.system_one(choice_request())

    assert route.call_count == 3
    assert info.value.attempts == 3
    assert info.value.retry_after == 2.0  # type: ignore[attr-defined]
    assert jev.spent_usd == 0.0
    assert len(jev.cache) == 0


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (401, {"detail": {"error_type": "authentication_error"}}, JevAuthError),
        (
            422,
            {
                "detail": [
                    {"type": "missing", "loc": ["body", "questions"], "msg": "Field required"}
                ]
            },
            JevValidationError,
        ),
        (
            400,
            {"detail": {"error_type": "api_usage_error", "message": "Invalid request."}},
            JevValidationError,
        ),
        (500, {"detail": "boom"}, JevHTTPError),
    ],
)
async def test_non_retryable_statuses_raise_at_once_with_body(
    ft: FakeTime, router: respx.MockRouter, status: int, body: Any, error: type[JevHTTPError]
) -> None:
    route = router.post("/v1/systemone").respond(
        status, json=body, headers={"x-typesafe-request-id": "req_1"}
    )
    jev = client(ft)
    with pytest.raises(JevHTTPError) as info:
        await jev.system_one(choice_request())

    assert type(info.value) is error
    assert (info.value.status_code, info.value.body) == (status, body)
    assert info.value.request_id == "req_1"
    assert route.call_count == 1
    assert ft.sleeps == []


async def test_transport_failure_raises_and_releases_budget_reservation(
    ft: FakeTime, router: respx.MockRouter
) -> None:
    request = choice_request()
    estimate = cost_usd(estimate_input_tokens(canonical_json(wire_body(request, PINNED))))
    route = router.post("/v1/systemone").mock(side_effect=[httpx.ConnectError("refused"), reply()])
    jev = client(ft, max_usd=estimate)  # room for exactly one request in flight

    with pytest.raises(JevConnectionError):
        await jev.system_one(request)
    await jev.system_one(request)
    assert route.call_count == 2


# --- version pin ----------------------------------------------------------------------------


async def test_request_is_sent_pinned_with_env_key(
    ft: FakeTime, router: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_ENV, KEY)
    route = router.post("/v1/systemone").mock(return_value=reply())
    request = SystemOneRequest(
        state="s",
        model="jev-latest",
        questions={"urgent": NoulQuestion(instructions="Urgent?")},
    )
    await client(ft, api_key=None).system_one_with_meta(request)

    call = route.calls.last.request
    assert call.headers["authorization"] == f"Bearer {KEY}"
    assert sent(route) == {
        "state": "s",
        "model": PINNED,
        "questions": {"urgent": {"type": "noul", "instructions": "Urgent?"}},
    }


@pytest.mark.parametrize("model", ["jev-latest", "jev-preview", "jev-1.13", ""])
def test_aliases_and_unversioned_models_are_refused(model: str) -> None:
    with pytest.raises(ValueError, match="versioned"):
        JevClient(KEY, model=model, cache=ResponseCache.in_memory())


async def test_version_drift_raises_is_billed_and_not_cached(
    ft: FakeTime, router: respx.MockRouter
) -> None:
    route = router.post("/v1/systemone").mock(return_value=reply(body=ok(model="jev-1.14.0")))
    jev = client(ft)
    for _ in range(2):
        with pytest.raises(JevVersionDrift) as info:
            await jev.system_one(choice_request())
        assert (info.value.expected, info.value.actual) == (PINNED, "jev-1.14.0")

    assert route.call_count == 2
    assert jev.spent_usd == pytest.approx(2 * cost_usd(300))


async def test_cached_drifted_answer_raises_on_strict_replay(
    ft: FakeTime, router: respx.MockRouter, caplog: pytest.LogCaptureFixture
) -> None:
    router.post("/v1/systemone").mock(return_value=reply(body=ok(model="jev-1.14.0")))
    cache = ResponseCache.in_memory()
    lenient = client(ft, cache=cache, strict_version=False)
    with caplog.at_level(logging.WARNING, logger="jevemu.jev_client"):
        response, record = await lenient.system_one_with_meta(choice_request())
    assert response.model == record.model == "jev-1.14.0"
    assert "jev-1.14.0" in caplog.text

    strict = client(ft, cache=cache, api_key=None)
    with pytest.raises(JevVersionDrift):
        await strict.system_one(choice_request())


# --- cache and accounting ---------------------------------------------------------------------


async def test_cache_hit_needs_no_network_nor_key_and_costs_nothing(
    ft: FakeTime, router: respx.MockRouter
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        ft.now += 0.25  # server round trip
        return reply(**{"x-typesafe-request-id": "req_42"})

    route = router.post("/v1/systemone").mock(side_effect=respond)
    cache = ResponseCache.in_memory()
    online = client(ft, cache=cache)
    first, paid = await online.system_one_with_meta(choice_request())

    offline = client(ft, cache=cache, api_key=None)
    again, free = await offline.system_one_with_meta(choice_request())

    assert route.call_count == 1
    assert again == first
    assert (paid.cached, paid.attempts, paid.request_id) == (False, 1, "req_42")
    assert paid.latency_ms == pytest.approx(250.0)
    assert paid.cost_usd == pytest.approx(300 * 0.042 / 1e6)
    assert (free.cached, free.attempts, free.cost_usd) == (True, 0, 0.0)
    assert free.latency_ms == paid.latency_ms
    assert free.input_tokens == paid.input_tokens == 300
    assert offline.spent_usd == 0.0
    assert (offline.cache_hits, offline.network_calls) == (1, 0)


async def test_cache_miss_without_key_fails_before_network(
    ft: FakeTime, router: respx.MockRouter
) -> None:
    route = router.post("/v1/systemone").mock(return_value=reply())
    with pytest.raises(JevAPIKeyMissing):
        await client(ft, api_key=None).system_one(choice_request())
    assert route.call_count == 0


async def test_option_order_is_a_different_cached_request(
    ft: FakeTime, router: respx.MockRouter
) -> None:
    route = router.post("/v1/systemone").mock(return_value=reply())
    jev = client(ft)
    _, ab = await jev.system_one_with_meta(choice_request(order=("a", "b")))
    _, ba = await jev.system_one_with_meta(choice_request(order=("b", "a")))

    assert route.call_count == 2
    assert not ab.cached
    assert not ba.cached
    assert list(sent(route, 1)["questions"]["dept"]["criteria"]) == ["b", "a"]


# --- budget ----------------------------------------------------------------------------------


def estimate_of(request: SystemOneRequest) -> float:
    return cost_usd(estimate_input_tokens(canonical_json(wire_body(request, PINNED))))


async def test_budget_guard_refuses_before_sending(ft: FakeTime, router: respx.MockRouter) -> None:
    route = router.post("/v1/systemone").mock(return_value=reply())
    request = choice_request()
    jev = client(ft, max_usd=estimate_of(request) * 0.99)
    with pytest.raises(JevBudgetExceeded):
        await jev.system_one(request)
    assert route.call_count == 0
    assert jev.spent_usd == 0.0


async def test_budget_counts_actual_spend_and_spares_cache_hits(
    ft: FakeTime, router: respx.MockRouter
) -> None:
    route = router.post("/v1/systemone").mock(return_value=reply(body=ok(tokens=300)))
    first, second = choice_request("state one"), choice_request("state two")
    assert estimate_of(first) == estimate_of(second)
    jev = client(ft, max_usd=cost_usd(300) + estimate_of(second) * 0.99)

    await jev.system_one(first)
    assert jev.spent_usd == pytest.approx(cost_usd(300))
    with pytest.raises(JevBudgetExceeded) as info:
        await jev.system_one(second)
    assert info.value.committed_usd == pytest.approx(cost_usd(300))
    _, record = await jev.system_one_with_meta(first)  # cached: allowed with no budget left

    assert record.cached
    assert route.call_count == 1


async def test_budget_reserves_estimates_of_requests_in_flight(
    ft: FakeTime, router: respx.MockRouter
) -> None:
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0)  # let the second request try while the first is in flight
        return reply()

    route = router.post("/v1/systemone").mock(side_effect=slow)
    a, b = choice_request("state one"), choice_request("state two")
    jev = client(ft, max_usd=estimate_of(a) * 1.5)

    results = await asyncio.gather(jev.system_one(a), jev.system_one(b), return_exceptions=True)
    assert sum(isinstance(r, JevBudgetExceeded) for r in results) == 1
    assert route.call_count == 1


# --- rate limiting ------------------------------------------------------------------------------


async def test_token_bucket_spaces_concurrent_starts(ft: FakeTime) -> None:
    bucket = TokenBucket(1200, clock=ft.clock, sleep=ft.sleep)
    starts: list[float] = []

    async def start() -> None:
        await bucket.acquire()
        starts.append(ft.now)

    await asyncio.gather(*(start() for _ in range(5)))
    gaps = [b - a for a, b in pairwise(starts)]
    assert gaps == pytest.approx([0.05] * 4)

    ft.now += 10.0  # idle: no debt carried over
    assert await bucket.acquire() == 0.0


async def test_token_bucket_burst_then_steady_rate(ft: FakeTime) -> None:
    bucket = TokenBucket(60, burst=3, clock=ft.clock, sleep=ft.sleep)
    waits = [await bucket.acquire() for _ in range(5)]
    assert waits == pytest.approx([0.0, 0.0, 0.0, 1.0, 1.0])


async def test_client_request_starts_respect_max_rpm(
    ft: FakeTime, router: respx.MockRouter
) -> None:
    starts: list[float] = []

    def respond(request: httpx.Request) -> httpx.Response:
        starts.append(ft.now)
        return reply()

    router.post("/v1/systemone").mock(side_effect=respond)
    jev = client(ft, max_rpm=60)
    for i in range(3):
        await jev.system_one(choice_request(f"state {i}"))
    assert [b - a for a, b in pairwise(starts)] == pytest.approx([1.0, 1.0])


# --- secrets --------------------------------------------------------------------------------------


async def test_api_key_never_reaches_logs_records_errors_or_cache(
    ft: FakeTime, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cache_path = tmp_path / "jev.sqlite"
    records: list[JevCallRecord] = []
    errors: list[Exception] = []
    with respx.mock(base_url=BASE) as mock, caplog.at_level(logging.DEBUG):
        route = mock.post("/v1/systemone").mock(
            side_effect=[
                reply(429, {"detail": f"slow down {KEY}"}, **{"retry-after": "1"}),
                reply(),
                reply(401, {"detail": {"message": f"bad key {KEY}"}}),
                httpx.Response(422, text=f"validation failed for key={KEY}"),
            ]
        )
        jev = client(ft, cache=ResponseCache(cache_path))
        records.append((await jev.system_one_with_meta(choice_request("one")))[1])
        records.append((await jev.system_one_with_meta(choice_request("one")))[1])
        for state in ("two", "three"):
            with pytest.raises(JevHTTPError) as info:
                await jev.system_one(choice_request(state))
            errors.append(info.value)
        jev.cache.close()

    assert route.calls[0].request.headers["authorization"] == f"Bearer {KEY}"
    surfaces = [
        caplog.text,
        repr(jev),
        *map(repr, records),
        *map(str, errors),
        *(repr(getattr(e, "body", None)) for e in errors),
        *files_text(tmp_path),
    ]
    assert not [s for s in surfaces if KEY in s]
    assert errors[0].body == {"detail": {"message": "bad key ***"}}  # type: ignore[attr-defined]


# --- models ---------------------------------------------------------------------------------------


async def test_list_models_parses_catalogue(ft: FakeTime, router: respx.MockRouter) -> None:
    router.get("/v1/models").respond(
        json={
            "models": [
                {"name": "jev-latest", "description": "d", "release_date": "2026-09-10"},
                {"name": "jev-preview", "description": "p", "release_date": "2026-09-10"},
            ]
        }
    )
    models = await client(ft).list_models()
    assert [m.name for m in models] == ["jev-latest", "jev-preview"]


# --- nondeterminism probe -----------------------------------------------------------------------


async def test_probe_reports_max_delta_per_answer_and_keeps_cached_answer(
    ft: FakeTime, router: respx.MockRouter
) -> None:
    bodies = iter(
        [
            ok(p_a=0.70, noul=0.40),  # baseline, cached
            ok(p_a=0.72, noul=0.40),
            ok(p_a=0.65, noul=0.50),
            ok(p_a=0.70, noul=0.45),
        ]
    )
    route = router.post("/v1/systemone").mock(side_effect=lambda _: reply(body=next(bodies)))
    requests = [
        SystemOneRequest(
            state=f"ticket {i}",
            questions={
                "dept": ChoiceQuestion(instructions="Team?", criteria={"a": None, "b": None}),
                "urgent": NoulQuestion(instructions="Urgent?"),
            },
        )
        for i in range(10)
    ]
    jev = client(ft)
    report = await probe_nondeterminism(jev, requests, fraction=0.05, repeats=3, seed=1)

    assert (report.requests_total, report.requests_sampled) == (10, 1)
    assert route.call_count == 4
    deltas = {a.question_id: (a.n_samples, a.max_abs_delta) for a in report.answers}
    assert deltas == {"dept": (4, pytest.approx(0.07)), "urgent": (4, pytest.approx(0.10))}
    assert report.max_abs_delta == pytest.approx(0.10)
    assert report.cost_usd == pytest.approx(4 * cost_usd(300))

    probed = next(r for r in requests if wire_body(r, PINNED)["state"] == sent(route, 0)["state"])
    replay, record = await jev.system_one_with_meta(probed)
    assert record.cached
    assert replay.answers["dept"].probabilities["a"] == 0.70  # type: ignore[union-attr]


async def test_probe_rejects_bad_parameters() -> None:
    jev = JevClient(KEY, cache=ResponseCache.in_memory())
    with pytest.raises(ValueError, match="fraction"):
        await probe_nondeterminism(jev, [choice_request()], fraction=1.5)
    with pytest.raises(ValueError, match="repeats"):
        await probe_nondeterminism(jev, [choice_request()], repeats=0)


# --- helpers ----------------------------------------------------------------------------------

NOW = datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("3", 3.0),
        (" 0.5 ", 0.5),
        ("-4", 0.0),
        (format_datetime(NOW + timedelta(seconds=10), usegmt=True), 10.0),
        (format_datetime(NOW - timedelta(seconds=10), usegmt=True), 0.0),
        ("soon", None),
        ("nan", None),
        (None, None),
    ],
)
def test_parse_retry_after(value: str | None, seconds: float | None) -> None:
    assert parse_retry_after(value, now=NOW) == seconds


def test_env_file_parsing_and_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "\n"
        f"{API_KEY_ENV}={KEY}\n"
        'export QUOTED="a # not a comment"\n'
        "SINGLE='x=y'\n"
        "INLINE=value # trailing comment\n"
        "not a line\n"
        "1BAD=x\n",
        "utf-8",
    )
    assert read_env_file(env) == {
        API_KEY_ENV: KEY,
        "QUOTED": "a # not a comment",
        "SINGLE": "x=y",
        "INLINE": "value",
    }
    for name in (API_KEY_ENV, "QUOTED", "SINGLE", "INLINE"):
        monkeypatch.setenv(name, "")  # registers the variable so teardown removes it
        monkeypatch.delenv(name)
    monkeypatch.setenv("SINGLE", "kept")

    assert load_env_file(env) == [API_KEY_ENV, "QUOTED", "INLINE"]
    assert os.environ[API_KEY_ENV] == KEY
    assert os.environ["SINGLE"] == "kept"
    assert read_env_file(tmp_path / "missing") == {}
    assert load_env_file(tmp_path / "missing") == []
