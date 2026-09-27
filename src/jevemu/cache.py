"""Content-addressed SQLite cache of real Jev responses.

The key is ``sha256(model id + "\\n" + canonical request JSON)``. Canonical JSON is compact
(``separators=(",", ":")``, UTF-8, no NaN) and keeps key order exactly as sent: option order
in a Choice ``criteria`` map (and key order inside JSON states) can change Jev's answer, so
requests that differ only in order are different requests and must not share an entry.

Each entry stores the request body actually sent (no credentials), the response JSON, the
client-measured latency and a UTC timestamp. The first stored answer for a key wins; later
``put`` calls for the same key are ignored so a cached answer never silently changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

__all__ = [
    "MEMORY",
    "CacheEntry",
    "ResponseCache",
    "cache_key",
    "canonical_json",
    "default_cache_path",
]

MEMORY = ":memory:"
"""Pass as ``path`` for a process-local cache that is never written to disk."""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jev_responses (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL
)
"""


def default_cache_path() -> Path:
    """``~/.cache/jevemu/jev_cache.sqlite``."""
    return Path.home() / ".cache" / "jevemu" / "jev_cache.sqlite"


def canonical_json(body: Mapping[str, Any]) -> str:
    """Compact, order-preserving JSON: the exact bytes the client sends."""
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def cache_key(canonical_request: str, model: str) -> str:
    """Hex sha256 over the model id and the canonical request JSON."""
    return hashlib.sha256(f"{model}\n{canonical_request}".encode()).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    key: str
    model: str
    """Model id the request was pinned to (part of the key)."""
    request: dict[str, Any]
    response: dict[str, Any]
    latency_ms: float
    """Client-measured round trip of the HTTP attempt that produced ``response``."""
    created_at: datetime
    """When the response was received (timezone-aware UTC)."""


class ResponseCache:
    """SQLite store of Jev responses keyed by :func:`cache_key`.

    The database (and its parent directory) is created on first use, so constructing a cache
    touches nothing on disk. Safe to share between tasks and threads of one process; separate
    processes may share a file (WAL journal).
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path: Path | None = None if path == MEMORY else Path(path or default_cache_path())
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    @classmethod
    def in_memory(cls) -> ResponseCache:
        return cls(MEMORY)

    def __repr__(self) -> str:
        return f"ResponseCache({str(self.path) if self.path else MEMORY!r})"

    def __enter__(self) -> ResponseCache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            conn, self._conn = self._conn, None
            if conn is not None:
                conn.close()

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.path is None:
                conn = sqlite3.connect(MEMORY, check_same_thread=False)
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
                conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_SCHEMA)
            conn.commit()
            self._conn = conn
        return self._conn

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            row = (
                self._connection()
                .execute(
                    "SELECT key, model, request_json, response_json, latency_ms, created_at "
                    "FROM jev_responses WHERE key = ?",
                    (key,),
                )
                .fetchone()
            )
        if row is None:
            return None
        return CacheEntry(
            key=row[0],
            model=row[1],
            request=json.loads(row[2]),
            response=json.loads(row[3]),
            latency_ms=float(row[4]),
            created_at=datetime.fromisoformat(row[5]),
        )

    def put(self, entry: CacheEntry) -> bool:
        """Store ``entry``; returns False (and keeps the old entry) if the key exists."""
        if entry.created_at.tzinfo is None:
            raise ValueError("CacheEntry.created_at must be timezone-aware")
        with self._lock:
            conn = self._connection()
            cursor = conn.execute(
                "INSERT OR IGNORE INTO jev_responses "
                "(key, model, request_json, response_json, latency_ms, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    entry.key,
                    entry.model,
                    canonical_json(entry.request),
                    canonical_json(entry.response),
                    entry.latency_ms,
                    entry.created_at.isoformat(),
                ),
            )
            conn.commit()
            return cursor.rowcount == 1

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with self._lock:
            row = (
                self._connection()
                .execute("SELECT 1 FROM jev_responses WHERE key = ?", (key,))
                .fetchone()
            )
        return row is not None

    def __len__(self) -> int:
        with self._lock:
            (count,) = self._connection().execute("SELECT COUNT(*) FROM jev_responses").fetchone()
        return int(count)
