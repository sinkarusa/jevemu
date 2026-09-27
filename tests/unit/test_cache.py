from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from jevemu.cache import CacheEntry, ResponseCache, cache_key, canonical_json

MODEL = "jev-1.13.0"
BODY: dict[str, Any] = {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": MODEL,
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "Payments", "technical": None},
        }
    },
}
RESPONSE: dict[str, Any] = {
    "model": MODEL,
    "answers": {
        "department": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.9, "technical": 0.1},
            "confidence": 0.8,
        }
    },
    "usage": {"input_tokens": 300, "output_tokens": 20},
}
WHEN = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


def entry(response: dict[str, Any] = RESPONSE, *, latency_ms: float = 123.5) -> CacheEntry:
    key = cache_key(canonical_json(BODY), MODEL)
    return CacheEntry(key, MODEL, BODY, response, latency_ms, WHEN)


def test_key_is_sha256_of_model_and_compact_ordered_json() -> None:
    # Changing the key format silently invalidates every on-disk cache (and re-pays for it).
    # sha256("jev-1.13.0\n" + compact JSON), checked with coreutils sha256sum.
    assert cache_key(canonical_json(BODY), MODEL) == (
        "24c5737ea947c994354a0544f5621e9468f5af7d7624bfbff48962a0438149a9"
    )


def test_option_order_and_model_are_part_of_the_key() -> None:
    swapped = copy.deepcopy(BODY)
    criteria = swapped["questions"]["department"]["criteria"]
    swapped["questions"]["department"]["criteria"] = dict(reversed(list(criteria.items())))
    keys = {
        cache_key(canonical_json(BODY), MODEL),
        cache_key(canonical_json(swapped), MODEL),
        cache_key(canonical_json(BODY), "jev-1.14.0"),
    }
    assert len(keys) == 3


def test_entries_survive_reopening_the_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "jev.sqlite"
    with ResponseCache(path) as cache:
        assert cache.put(entry())
    with ResponseCache(path) as reopened:
        got = reopened.get(entry().key)
    assert got == entry()
    assert got is not None
    assert got.created_at.utcoffset() == timedelta(0)


def test_first_answer_for_a_key_wins() -> None:
    cache = ResponseCache.in_memory()
    other = copy.deepcopy(RESPONSE)
    other["answers"]["department"]["probabilities"] = {"billing": 0.2, "technical": 0.8}
    assert cache.put(entry())
    assert not cache.put(entry(other, latency_ms=1.0))
    assert cache.get(entry().key) == entry()
    assert len(cache) == 1


def test_unknown_key_misses() -> None:
    cache = ResponseCache.in_memory()
    cache.put(entry())
    assert cache.get("0" * 64) is None
    assert "0" * 64 not in cache
    assert entry().key in cache


def test_naive_timestamps_are_rejected() -> None:
    naive = CacheEntry("k", MODEL, BODY, RESPONSE, 1.0, datetime(2026, 9, 24))
    with pytest.raises(ValueError, match="timezone-aware"):
        ResponseCache.in_memory().put(naive)
