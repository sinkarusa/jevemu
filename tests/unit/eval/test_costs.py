from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from jevemu.bench.manifest import Invocation
from jevemu.bench.runner import ItemError, ItemRecord
from jevemu.eval.costs import (
    GPT_6_LUNA,
    GPT_6_LUNA_BATCH,
    GPU_USD_PER_HOUR_ENV,
    GPU_WATTS_ENV,
    JEV_1_13_0,
    USD_PER_WH_ENV,
    GpuTimePrice,
    PriceBook,
    TokenUsage,
    item_costs,
    record_usage,
)
from jevemu.types import NoulAnswer, Usage

T0 = datetime(2026, 9, 25, tzinfo=timezone.utc)
JEV_USD_PER_TOKEN = 0.042e-6


def _record(
    item_id: str, usage: Usage | None, *, cost_usd: float | None = None, cached: bool | None = None
) -> ItemRecord:
    if usage is None:
        return ItemRecord(
            item_id=item_id,
            question_type="noul",
            gold=True,
            error=ItemError(type="JevConnectionError", message="read timeout"),
        )
    return ItemRecord(
        item_id=item_id,
        question_type="noul",
        gold=True,
        answer=NoulAnswer(noul=0.9),
        usage=usage,
        cost_usd=cost_usd,
        cached=cached,
    )


def _invocation(seconds: float | None, start: float = 0.0) -> Invocation:
    started = T0 + timedelta(seconds=start)
    return Invocation(
        started_at=started,
        finished_at=None if seconds is None else started + timedelta(seconds=seconds),
        git_commit=None,
        git_dirty=None,
        concurrency=1,
        limit=None,
    )


def test_luna_prices_cached_input_cache_writes_and_output_at_their_own_rates() -> None:
    usage = TokenUsage(
        input_tokens=200_000,
        output_tokens=10_000,
        cached_input_tokens=50_000,
        cache_write_tokens=20_000,
    )
    # 130k uncached x $0.10 + 50k cached x $0.01 + 20k written x $0.125 + 10k output x $0.50
    standard = (130_000 * 0.10 + 50_000 * 0.01 + 20_000 * 0.125 + 10_000 * 0.50) / 1e6
    assert GPT_6_LUNA.cost(usage) == pytest.approx(standard)
    assert GPT_6_LUNA_BATCH.cost(usage) == pytest.approx(standard / 2)


def test_luna_long_context_tier_reprices_the_whole_request_above_272k_input_tokens() -> None:
    at = TokenUsage(input_tokens=272_000, output_tokens=10_000)
    above = TokenUsage(input_tokens=272_001, output_tokens=10_000)
    assert GPT_6_LUNA.cost(at) == pytest.approx((272_000 * 0.10 + 10_000 * 0.50) / 1e6)
    assert GPT_6_LUNA.cost(above) == pytest.approx((272_001 * 0.10 * 2 + 10_000 * 0.50 * 1.5) / 1e6)
    cached = TokenUsage(input_tokens=300_000, output_tokens=0, cached_input_tokens=100_000)
    assert GPT_6_LUNA_BATCH.cost(cached) == pytest.approx(
        (200_000 * 0.10 + 100_000 * 0.01) * 2 * 0.5 / 1e6
    )


def test_jev_items_cost_their_recorded_charge_and_cache_hits_the_table_price() -> None:
    records = [
        _record(
            "billed", Usage(input_tokens=1000, output_tokens=800), cost_usd=1000 * JEV_USD_PER_TOKEN
        ),
        _record("hit", Usage(input_tokens=2000, output_tokens=900), cost_usd=0.0, cached=True),
        _record(
            "near", Usage(input_tokens=1000, output_tokens=0), cost_usd=1005 * JEV_USD_PER_TOKEN
        ),
        _record(
            "off", Usage(input_tokens=1000, output_tokens=0), cost_usd=1020 * JEV_USD_PER_TOKEN
        ),
        _record("failed", None),
    ]
    costs = item_costs(records, JEV_1_13_0)
    assert [c.cost_usd for c in costs] == pytest.approx(
        [t * JEV_USD_PER_TOKEN for t in (1000, 2000, 1005, 1020, 0)]
    )
    assert [c.billed for c in costs] == [True, False, True, True, False]
    assert [c.price_mismatch for c in costs] == [False, False, False, True, False]


def test_gpu_time_is_split_by_tokens_and_items_without_usage_get_the_mean_share() -> None:
    records = [
        _record("a", Usage(input_tokens=90, output_tokens=10)),
        _record("b", Usage(input_tokens=290, output_tokens=10)),
        _record("c", None),
    ]
    # 100 s + 20 s; the unfinished invocation has no end and adds nothing
    invocations = [_invocation(100.0), _invocation(20.0, start=500.0), _invocation(None, 900.0)]
    costs = item_costs(records, GpuTimePrice(0.36), invocations=invocations)
    assert [c.gpu_seconds for c in costs] == pytest.approx([20.0, 60.0, 40.0])
    assert math.fsum(c.cost_usd or 0.0 for c in costs) == pytest.approx(120 / 3600 * 0.36)

    unrated = item_costs(records, GpuTimePrice(None), invocations=invocations)
    assert [c.gpu_seconds for c in unrated] == pytest.approx([20.0, 60.0, 40.0])
    assert [c.cost_usd for c in unrated] == [None, None, None]

    no_usage = item_costs(records[2:] * 4, GpuTimePrice(0.36), invocations=invocations)
    assert [c.gpu_seconds for c in no_usage] == pytest.approx([30.0] * 4)
    assert [c.gpu_seconds for c in item_costs(records, GpuTimePrice(0.36))] == [None] * 3


def test_usage_extra_fields_carry_prompt_cache_tokens_through_the_records_file() -> None:
    usage = Usage(
        input_tokens=500, output_tokens=5, cached_input_tokens=300, cache_write_tokens=100
    )
    record = ItemRecord.model_validate_json(_record("a", usage).model_dump_json())
    assert record_usage(record) == TokenUsage(500, 5, 300, 100)
    too_many = _record("b", Usage(input_tokens=100, output_tokens=0, cached_input_tokens=101))
    with pytest.raises(ValueError, match="exceed"):
        record_usage(too_many)


def test_price_book_resolution_and_gpu_rate_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (GPU_WATTS_ENV, USD_PER_WH_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(GPU_USD_PER_HOUR_ENV, "0.5")
    assert PriceBook.default().gpu.usd_per_gpu_hour == 0.5
    assert PriceBook.default(0.3).gpu.usd_per_gpu_hour == 0.3
    # An explicit rate beats the electricity estimate.
    assert PriceBook.default(gpu_watts=350.0).gpu.usd_per_gpu_hour == 0.5
    monkeypatch.delenv(GPU_USD_PER_HOUR_ENV)
    # No rate: the electricity estimate, 410 W x $0.00027/Wh = $0.1107/h.
    book = PriceBook.default()
    assert book.gpu.usd_per_gpu_hour == pytest.approx(0.1107)
    assert book.gpu.cost(3600.0) == pytest.approx(0.1107)
    monkeypatch.setenv(GPU_WATTS_ENV, "350")
    monkeypatch.setenv(USD_PER_WH_ENV, "0.0002")
    assert PriceBook.default().gpu.usd_per_gpu_hour == pytest.approx(0.07)
    assert PriceBook.default(gpu_watts=100.0).gpu.usd_per_gpu_hour == pytest.approx(0.02)
    with pytest.raises(ValueError, match="watts"):
        GpuTimePrice.from_power(-1.0, 0.0002)
    assert book.resolve({"kind": "emulator", "backend_model": "Qwen/Qwen3.6-27B"}) is book.gpu
    assert book.resolve({"kind": "jev", "model": "jev-1.13.0"}) is JEV_1_13_0
    luna_batch = {"kind": "openai", "model": "gpt-6-luna", "price_id": "gpt-6-luna:batch"}
    assert book.resolve(luna_batch) is GPT_6_LUNA_BATCH
    assert book.resolve({"kind": "jev", "model": "jev-9.9.9"}) is None
