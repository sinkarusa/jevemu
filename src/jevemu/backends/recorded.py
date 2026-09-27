"""Backend-level record/replay, independent of the transport.

``RecordingBackend`` wraps any backend and appends every call to a JSONL fixture;
``RecordedBackend`` replays that fixture offline. Calls are matched by a stable request key:
the sha256 of the canonical JSON of the method name and its arguments.

File layout: the first line is a header record
``{"kind": "header", "schema_version": N, "info": {...}, "capabilities": {...}}``; every other
line is ``{"kind": "call", "method": ..., "key": ..., "args": {...}, "result": ...}``.
Non-finite floats are stored as the strings ``"-inf"``, ``"inf"`` and ``"nan"``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jevemu.backends.base import (
    Backend,
    BackendInfo,
    Capabilities,
    NextTokenDist,
    RenderedPrompt,
    SeqScore,
    TokenLogprob,
)

__all__ = [
    "SCHEMA_VERSION",
    "FixtureMissing",
    "FixtureSchemaError",
    "RecordedBackend",
    "RecordingBackend",
    "request_key",
]

SCHEMA_VERSION = 1
"""Bump whenever the record layout or the request-key derivation changes."""

_NON_FINITE = {"-inf": -math.inf, "inf": math.inf, "nan": math.nan}


class FixtureMissing(LookupError):
    """Replay found no recorded call for a request."""

    def __init__(self, method: str, key: str, args: Mapping[str, Any]) -> None:
        self.method = method
        self.key = key
        self.request_args = args
        super().__init__(
            f"no recorded {method} call with key {key} (args: {_canonical_json(args)}); "
            "re-record the fixture"
        )


class FixtureSchemaError(ValueError):
    """The fixture file is malformed or was written with a different schema version."""


def request_key(method: str, args: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of ``method`` and its JSON-encoded ``args``."""
    payload = _canonical_json({"method": method, "args": args})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- encoding -------------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _enc_float(value: float) -> float | str:
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def _dec_float(value: Any) -> float:
    if isinstance(value, str):
        try:
            return _NON_FINITE[value]
        except KeyError:
            raise FixtureSchemaError(f"invalid float encoding {value!r}") from None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FixtureSchemaError(f"expected a number, got {value!r}")
    return float(value)


def _enc_prompt(prompt: RenderedPrompt) -> dict[str, Any]:
    return {
        "messages": [{"role": m.role, "content": m.content} for m in prompt.messages],
        "template_id": prompt.template_id,
    }


def _enc_token(token: TokenLogprob) -> dict[str, Any]:
    return {"token": token.token, "token_id": token.token_id, "logprob": _enc_float(token.logprob)}


def _dec_token(raw: Mapping[str, Any]) -> TokenLogprob:
    return TokenLogprob(
        token=raw["token"], token_id=raw["token_id"], logprob=_dec_float(raw["logprob"])
    )


def _enc_dist(dist: NextTokenDist) -> dict[str, Any]:
    return {
        "top": [_enc_token(t) for t in dist.top],
        "sampled": _enc_token(dist.sampled),
        "constrained": dist.constrained,
        "prompt_tokens": dist.prompt_tokens,
        "cached_tokens": dist.cached_tokens,
    }


def _dec_dist(raw: Mapping[str, Any]) -> NextTokenDist:
    return NextTokenDist(
        top=tuple(_dec_token(t) for t in raw["top"]),
        sampled=_dec_token(raw["sampled"]),
        constrained=raw["constrained"],
        prompt_tokens=raw["prompt_tokens"],
        cached_tokens=raw["cached_tokens"],
    )


def _enc_scores(scores: Sequence[SeqScore]) -> list[dict[str, Any]]:
    return [
        {
            "continuation": s.continuation,
            "tokens": list(s.tokens),
            "token_logprobs": [_enc_float(lp) for lp in s.token_logprobs],
            "prompt_tokens": s.prompt_tokens,
        }
        for s in scores
    ]


def _dec_scores(raw: Sequence[Mapping[str, Any]]) -> list[SeqScore]:
    return [
        SeqScore(
            continuation=s["continuation"],
            tokens=tuple(s["tokens"]),
            token_logprobs=tuple(_dec_float(lp) for lp in s["token_logprobs"]),
            prompt_tokens=s.get("prompt_tokens"),  # absent in fixtures recorded before the field
        )
        for s in raw
    ]


def _header(info: BackendInfo, capabilities: Capabilities) -> dict[str, Any]:
    return {
        "kind": "header",
        "schema_version": SCHEMA_VERSION,
        "info": {
            "backend": info.backend,
            "model": info.model,
            "model_revision": info.model_revision,
            "engine_version": info.engine_version,
            "flags": dict(info.flags),
        },
        "capabilities": dataclasses.asdict(capabilities),
    }


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FixtureSchemaError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise FixtureSchemaError(f"{path}:{lineno}: record is not a JSON object")
            records.append(record)
    if not records or records[0].get("kind") != "header":
        raise FixtureSchemaError(f"{path}: first record must be the header")
    version = records[0].get("schema_version")
    if version != SCHEMA_VERSION:
        raise FixtureSchemaError(
            f"{path}: fixture schema_version {version!r} != supported {SCHEMA_VERSION}; "
            "re-record the fixture"
        )
    return records


def _init_fixture(path: Path, header: dict[str, Any]) -> None:
    """Start a new fixture with ``header``, or check an existing fixture's header against it."""
    if path.exists() and path.stat().st_size > 0:
        existing = _read_records(path)[0]
        if existing != header:
            raise FixtureSchemaError(
                f"{path}: existing header {existing} does not match the "
                f"backend being recorded {header}"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.write(_canonical_json(header) + "\n")


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


# --- backends -------------------------------------------------------------------------------


class RecordingBackend:
    """Forwards to ``inner`` and appends each call and result to the JSONL fixture at ``path``.

    A new file gets a header from ``inner.health()`` and ``inner.capabilities``. An existing
    fixture is appended to, provided its header matches the inner backend's.
    """

    def __init__(self, inner: Backend, path: str | os.PathLike[str]) -> None:
        self.inner = inner
        self.path = Path(path)
        self.capabilities = inner.capabilities
        self._header_ready = False
        self._header_lock = asyncio.Lock()
        self._append_lock = asyncio.Lock()

    async def next_token_logprobs(
        self, prompt: RenderedPrompt, *, allowed: Sequence[str] | None, top_k: int
    ) -> NextTokenDist:
        dist = await self.inner.next_token_logprobs(prompt, allowed=allowed, top_k=top_k)
        args = _next_token_args(prompt, allowed, top_k)
        await self._append("next_token_logprobs", args, _enc_dist(dist))
        return dist

    async def sequence_logprobs(
        self, prefix: RenderedPrompt, continuations: Sequence[str]
    ) -> list[SeqScore]:
        scores = await self.inner.sequence_logprobs(prefix, continuations)
        args = _sequence_args(prefix, continuations)
        await self._append("sequence_logprobs", args, _enc_scores(scores))
        return scores

    async def tokenize(self, text: str) -> list[int]:
        ids = await self.inner.tokenize(text)
        await self._append("tokenize", {"text": text}, list(ids))
        return ids

    async def detokenize(self, ids: Sequence[int]) -> str:
        text = await self.inner.detokenize(ids)
        await self._append("detokenize", {"ids": list(ids)}, text)
        return text

    async def health(self) -> BackendInfo:
        info = await self.inner.health()
        await self._ensure_header()
        return info

    async def _append(self, method: str, args: dict[str, Any], result: Any) -> None:
        await self._ensure_header()
        record = {
            "kind": "call",
            "method": method,
            "key": request_key(method, args),
            "args": args,
            "result": result,
        }
        line = _canonical_json(record) + "\n"
        # One write at a time keeps lines whole and in the order the calls completed.
        async with self._append_lock:
            await asyncio.to_thread(_append_line, self.path, line)

    async def _ensure_header(self) -> None:
        async with self._header_lock:
            if self._header_ready:
                return
            header = _header(await self.inner.health(), self.inner.capabilities)
            await asyncio.to_thread(_init_fixture, self.path, header)
            self._header_ready = True


class RecordedBackend:
    """Replays a fixture written by ``RecordingBackend``; raises ``FixtureMissing`` on a miss.

    ``capabilities`` and ``health()`` come from the fixture header. When a key was recorded
    more than once, the latest record wins.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        records = _read_records(self.path)
        header = records[0]
        try:
            raw_info = header["info"]
            self._info = BackendInfo(
                backend=raw_info["backend"],
                model=raw_info["model"],
                model_revision=raw_info["model_revision"],
                engine_version=raw_info["engine_version"],
                flags=dict(raw_info["flags"]),
            )
            self.capabilities = Capabilities(**header["capabilities"])
            self._results: dict[str, Any] = {
                record["key"]: record["result"] for record in records[1:]
            }
        except (KeyError, TypeError) as exc:
            raise FixtureSchemaError(f"{self.path}: malformed record: {exc!r}") from exc

    async def next_token_logprobs(
        self, prompt: RenderedPrompt, *, allowed: Sequence[str] | None, top_k: int
    ) -> NextTokenDist:
        raw = self._lookup("next_token_logprobs", _next_token_args(prompt, allowed, top_k))
        return _dec_dist(raw)

    async def sequence_logprobs(
        self, prefix: RenderedPrompt, continuations: Sequence[str]
    ) -> list[SeqScore]:
        return _dec_scores(self._lookup("sequence_logprobs", _sequence_args(prefix, continuations)))

    async def tokenize(self, text: str) -> list[int]:
        return [int(i) for i in self._lookup("tokenize", {"text": text})]

    async def detokenize(self, ids: Sequence[int]) -> str:
        return str(self._lookup("detokenize", {"ids": list(ids)}))

    async def health(self) -> BackendInfo:
        return self._info

    def _lookup(self, method: str, args: dict[str, Any]) -> Any:
        key = request_key(method, args)
        try:
            return self._results[key]
        except KeyError:
            raise FixtureMissing(method, key, args) from None


def _next_token_args(
    prompt: RenderedPrompt, allowed: Sequence[str] | None, top_k: int
) -> dict[str, Any]:
    return {
        "prompt": _enc_prompt(prompt),
        "allowed": None if allowed is None else list(allowed),
        "top_k": top_k,
    }


def _sequence_args(prefix: RenderedPrompt, continuations: Sequence[str]) -> dict[str, Any]:
    return {"prefix": _enc_prompt(prefix), "continuations": list(continuations)}
