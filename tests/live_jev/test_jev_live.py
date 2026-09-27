"""Live checks against the real Jev API (tier T4; costs well under one cent per run).

Skipped unless ``TYPESAFE_API_KEY`` is set in the environment or in the repo's ``.env`` (read
here without exporting it). Run with ``uv run pytest -m live_jev tests/live_jev -s``.

Each doc example in ``SOURCES`` is sent once through a fresh temporary cache with
``max_usd=MAX_USD``; the version pin is asserted and the golden fixtures in
``tests/golden/jev/`` are re-recorded (request, response, latency, date; no credentials).
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

import jevemu
from jevemu.cache import ResponseCache, canonical_json
from jevemu.jev_client import (
    API_KEY_ENV,
    DEFAULT_BASE_URL,
    JevCallRecord,
    JevClient,
    estimate_input_tokens,
    probe_nondeterminism,
    read_env_file,
)
from jevemu.types import DEFAULT_MODEL, SystemOneRequest

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "tests" / "golden" / "fixtures" / "jev_docs"
OUT = REPO / "tests" / "golden" / "jev"
MAX_USD = 1.0
SOURCES = (
    "api__01",  # noul, no criteria
    "api__02",  # noul with string criteria
    "api__03",  # choice, 3 options
    "api__04",  # score, 3 levels
    "introduction_quickstart__02",  # choice + score + noul in one request
    "primitives_choice__02",  # 5 choices, null option descriptions
    "primitives_choice__03",  # object instructions and option descriptions
    "primitives_score__03",  # object score levels
    "primitives_noul__02",  # object state and instructions
    "primitives_advanced__05",  # noul with object criteria
)

API_KEY = os.environ.get(API_KEY_ENV) or read_env_file(REPO / ".env").get(API_KEY_ENV, "")

pytestmark = [
    pytest.mark.live_jev,
    pytest.mark.skipif(not API_KEY, reason=f"set {API_KEY_ENV} (or add it to .env)"),
]


def write_fixture(path: Path, fixture: dict[str, Any]) -> None:
    text = json.dumps(fixture, indent=1, ensure_ascii=False) + "\n"
    assert API_KEY not in text
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, "utf-8")


def meta(**fields: Any) -> dict[str, Any]:
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": DEFAULT_BASE_URL,
        "jevemu_version": jevemu.__version__,
        **fields,
    }


def load_request(source: str) -> SystemOneRequest:
    return SystemOneRequest.model_validate_json((DOCS / f"{source}__request.json").read_bytes())


async def test_models_catalogue(tmp_path: Path) -> None:
    async with JevClient(API_KEY, cache=ResponseCache(tmp_path / "jev.sqlite")) as jev:
        models = await jev.list_models()
    names = {m.name for m in models}
    assert "jev-latest" in names
    write_fixture(
        OUT / "models.json",
        {"meta": meta(), "response": {"models": [m.model_dump() for m in models]}},
    )


async def test_doc_examples_answer_pinned_and_refresh_fixtures(tmp_path: Path) -> None:
    requests = [load_request(s) for s in SOURCES]
    async with JevClient(
        API_KEY, cache=ResponseCache(tmp_path / "jev.sqlite"), max_usd=MAX_USD
    ) as jev:
        results = await asyncio.gather(*(jev.system_one_with_meta(r) for r in requests))

        records: list[JevCallRecord] = []
        for source, request, (response, record) in zip(SOURCES, requests, results, strict=True):
            assert response.model == record.model == DEFAULT_MODEL
            assert set(response.answers) == set(request.questions)
            assert not record.cached
            assert record.cost_usd > 0
            entry = jev.cache.get(record.cache_key)
            assert entry is not None
            assert estimate_input_tokens(canonical_json(entry.request)) >= record.input_tokens
            write_fixture(
                OUT / "systemone" / f"{source}.json",
                {
                    "meta": meta(
                        source=f"tests/golden/fixtures/jev_docs/{source}__request.json",
                        model=record.model,
                        request_id=record.request_id,
                        latency_ms=round(record.latency_ms, 1),
                        input_tokens=record.input_tokens,
                        cost_usd=record.cost_usd,
                    ),
                    "request": entry.request,
                    "response": entry.response,
                },
            )
            records.append(record)

        replays = await asyncio.gather(*(jev.system_one_with_meta(r) for r in requests))
        assert all(rec.cached and rec.cost_usd == 0 for _, rec in replays)
        assert [r for r, _ in replays] == [r for r, _ in results]

        report = await probe_nondeterminism(jev, requests, fraction=0.1, repeats=3)
        spent = jev.spent_usd

    latencies = sorted(r.latency_ms for r in records)
    print(
        f"\nJev live: {len(records)} requests, spent ${spent:.8f} "
        f"(probe ${report.cost_usd:.8f}), latency ms min/median/max "
        f"{latencies[0]:.0f}/{latencies[len(latencies) // 2]:.0f}/{latencies[-1]:.0f}, "
        f"probe max |dp| {report.max_abs_delta:.3f} over {report.requests_sampled} request(s): "
        + ", ".join(f"{a.question_id}={a.max_abs_delta:.2f}" for a in report.answers)
    )
    assert spent <= MAX_USD
