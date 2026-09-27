from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from jevemu.bench.compare import JEV_THROUGHPUT_NOTE, stats, stats_table
from jevemu.bench.manifest import Invocation, RunManifest, SplitRun, run_key
from jevemu.bench.runner import ItemError, ItemRecord
from jevemu.eval.costs import PriceBook
from jevemu.eval.splits import SplitMetadata
from jevemu.types import NoulAnswer, Usage

T0 = datetime(2026, 9, 25, tzinfo=timezone.utc)


def _answer(item_id: str, correct: bool, tokens: int, **fields: Any) -> ItemRecord:
    return ItemRecord(
        item_id=item_id,
        question_type="noul",
        gold=True,
        answer=NoulAnswer(noul=0.9 if correct else 0.1),
        latency_ms=100.0,
        usage=Usage(input_tokens=tokens, output_tokens=0),
        **fields,
    )


def _failed(item_id: str) -> ItemRecord:
    return ItemRecord(
        item_id=item_id,
        question_type="noul",
        gold=True,
        error=ItemError(type="ReadTimeout", message="timeout"),
    )


def _invocations(*seconds: float) -> list[Invocation]:
    return [
        Invocation(
            started_at=T0 + timedelta(hours=i),
            finished_at=T0 + timedelta(hours=i, seconds=s),
            git_commit=None,
            git_dirty=None,
            concurrency=16,
            limit=None,
        )
        for i, s in enumerate(seconds)
    ]


def _write_run(
    run_dir: Path,
    system: dict[str, Any],
    benchmarks: dict[str, tuple[list[ItemRecord], list[Invocation]]],
) -> Path:
    run_dir.mkdir(parents=True)
    manifest = RunManifest(system_id=run_dir.name, system=system, jevemu_version="test")
    for name, (records, invocations) in benchmarks.items():
        lines = "".join(record.model_dump_json() + "\n" for record in records)
        (run_dir / f"{run_key(name, 'select')}.jsonl").write_text(lines, encoding="utf-8")
        metadata = SplitMetadata(
            split_version="1",
            benchmark=name,
            question_type="noul",
            license="CC0",
            split="select",
            seed=0,
            screen_size=None,
            revision="0" * 40,
            source_sha256="1" * 64,
            n_benchmark=2 * len(records),
            n_items=len(records),
            items_sha256="2" * 64,
        )
        manifest.runs[run_key(name, "select")] = SplitRun(
            benchmark=name,
            split="select",
            split_metadata=metadata,
            status="complete",
            n_items=len(records),
            n_ok=sum(r.ok for r in records),
            n_errors=sum(not r.ok for r in records),
            spent_usd=math.fsum(r.cost_usd or 0.0 for r in records),
            invocations=invocations,
        )
    manifest.save(run_dir)
    return run_dir


def test_emulator_gpu_cost_sums_over_benchmarks_and_accuracy_is_the_macro(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path / "emu",
        {"kind": "emulator", "backend_model": "Qwen/Qwen3.6-27B"},
        {
            "toya": (
                [_answer("a0", True, 100), _answer("a1", False, 300), _failed("a2")],
                _invocations(100.0, 20.0),
            ),
            "toyb": ([_answer(f"b{i}", True, 50) for i in range(3)], _invocations(60.0)),
        },
    )
    result = stats(run, PriceBook.default(0.36))
    toya, toyb = result.benchmarks["toya"], result.benchmarks["toyb"]
    assert (toya.gpu_seconds, toyb.gpu_seconds) == pytest.approx((120.0, 60.0))
    assert result.total.gpu_seconds == pytest.approx(180.0)
    assert result.total.cost_usd == pytest.approx(180 / 3600 * 0.36)
    assert (toya.cost_usd or 0.0) + (toyb.cost_usd or 0.0) == pytest.approx(result.total.cost_usd)
    assert result.total.cost_per_1k_items == pytest.approx(180 / 3600 * 0.36 / 6 * 1000)
    assert result.total.cost_per_correct == pytest.approx(180 / 3600 * 0.36 / 4)
    assert (toya.accuracy, toyb.accuracy, result.total.accuracy) == pytest.approx((0.5, 1.0, 0.75))
    assert result.total.mean_input_tokens == pytest.approx(550 / 5)

    unrated = stats(run, PriceBook())
    assert unrated.total.gpu_seconds == pytest.approx(180.0)
    assert unrated.total.cost_usd is None


def test_cost_per_correct_is_undefined_without_a_correct_answer(tmp_path: Path) -> None:
    usd = 1000 * 0.042e-6
    run = _write_run(
        tmp_path / "jev",
        {"kind": "jev", "model": "jev-1.13.0"},
        {
            "toya": (
                [_answer(f"a{i}", False, 1000, cost_usd=usd, cached=False) for i in range(4)],
                [],
            )
        },
    )
    total = stats(run, PriceBook()).total
    assert total.cost_usd == pytest.approx(4 * usd)
    assert total.cost_per_1k_items == pytest.approx(usd * 1000)
    assert total.cost_per_correct is None


def test_a_system_without_a_price_keeps_its_tokens_but_no_cost(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path / "other",
        {"kind": "OtherClient", "model": "other-1"},
        {"toya": ([_answer("a0", True, 70), _answer("a1", True, 30)], _invocations(10.0))},
    )
    result = stats(run, PriceBook.default(0.36))
    assert result.price is None
    assert result.total.input_tokens == 100
    assert result.total.cost_usd is None
    assert result.total.gpu_seconds is None


def _timed(seconds: float, n_called: int, n_cache_hits: int = 0, hour: int = 0) -> Invocation:
    return Invocation(
        started_at=T0 + timedelta(hours=hour),
        finished_at=T0 + timedelta(hours=hour, seconds=seconds),
        git_commit=None,
        git_dirty=None,
        concurrency=16,
        limit=None,
        n_called=n_called,
        n_cache_hits=n_cache_hits,
    )


def test_throughput_pools_timed_invocations_and_skips_cache_hits(tmp_path: Path) -> None:
    toya = [_answer(f"a{i}", True, 10) for i in range(4)]
    run = _write_run(
        tmp_path / "jev",
        {"kind": "jev", "model": "jev-1.13.0"},
        {
            # a run with 3 cache hits (not counted), a retry, and a full replay from the
            # response cache (no call: its time is ignored)
            "toya": (toya, [_timed(10.0, 33, 3), _timed(5.0, 20, hour=1), _timed(1.0, 4, 4, 2)]),
            "toyb": ([_answer("b0", True, 10)], [_timed(25.0, 50)]),
            "toyc": ([_answer("c0", True, 10)], [_timed(0.5, 1, 1)]),
        },
    )
    result = stats(run, PriceBook())
    a, b, c = (result.benchmarks[n] for n in ("toya", "toyb", "toyc"))
    assert a.throughput_items == 50
    assert a.wall_seconds == pytest.approx(15.0)
    assert a.items_per_second == pytest.approx(50 / 15)
    assert b.items_per_second == pytest.approx(2.0)
    assert c.items_per_second is None
    assert result.total.items_per_second == pytest.approx(100 / 40)
    assert result.total.concurrency == (16,)

    table = stats_table([result])
    assert "| 2.50 † | 16 |" in table
    assert f"† {JEV_THROUGHPUT_NOTE}" in table
    assert "| — | — |" in result.to_markdown()
