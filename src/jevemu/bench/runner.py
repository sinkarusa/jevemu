"""Resumable runner: one system answers one benchmark split.

Layout under ``out_dir``:

- ``splits/<benchmark>.<split>.jsonl``: the frozen split
  (:class:`~jevemu.eval.splits.FrozenSplit`), written by the first run and read by every later
  one, so every system run into ``out_dir`` answers byte-identical items.
- ``<system_id>/<benchmark>.<split>.jsonl``: one :class:`ItemRecord` per item: gold, question
  type, the system's answer (or the error), latency, usage, cost and emulator diagnostics.
- ``<system_id>/manifest.json``: :class:`~jevemu.bench.manifest.RunManifest`.

Each item is sent as ``SystemOneRequest(state, questions={"answer": question})`` (the question
banks' request shape, so Jev cache keys match the bank runs). A
:class:`~jevemu.jev_client.JevClient` is called through ``system_one_with_meta``:
``latency_ms`` is Jev's recorded HTTP round trip (also for cache hits), ``wall_ms`` this call's
wall time, and ``cost_usd``/``cached`` come from its accounting. An emulator over a paid API
(diagnostics carrying :class:`~jevemu.types.ApiUsage`) gets the same fields from that block:
``latency_ms`` is the summed HTTP round trips, ``cost_usd`` the spend, ``cached`` whether every
call was a response-cache hit. Any other ``SystemOneClient`` gets ``latency_ms = wall_ms``
measured around ``system_one`` and no cost.

**Resuming.** Items with a successful record are never sent again; items whose latest attempt
failed are retried. Records are appended and flushed as items finish, so an interruption loses
at most the in-flight items. The file is rewritten (one record per item, split order) when a
run starts and when it ends. Resuming refuses a different system identity
(:func:`~jevemu.bench.manifest.system_differences`), a different split (``items_sha256``) or
records of items outside the split.

**Errors.** An exception answering one item (HTTP or validation error, emulator failure, an
answer of the wrong type) is recorded and the run continues. A paid-API error record carries
what its failed calls were billed (:class:`~jevemu.errors.ChatAPIResponseError` ``cost_usd``),
and a retried item's new record adds the spend of its failed record, so the records' spend is
what was billed.
:class:`~jevemu.errors.BudgetExceeded` (Jev's or the OpenAI backend's) stops the run cleanly: no
new items start, in-flight items finish, status ``stopped_budget``. :data:`FATAL_ERRORS` (bad
or missing API key, exhausted credit, version drift) stop the run the same way and are
re-raised once the state is saved. Both are unwrapped from the emulator's
:class:`~jevemu.emulator.QuestionError`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from jevemu.bench import BenchItem, BenchmarkSpec, Gold, QuestionType, get_benchmark
from jevemu.bench.manifest import (
    Invocation,
    RunManifest,
    RunStatus,
    SplitRun,
    describe_system,
    git_state,
    jevemu_version,
    run_key,
    system_differences,
)
from jevemu.emulator import QuestionError
from jevemu.errors import (
    BudgetExceeded,
    ChatAPIAuthError,
    ChatAPIKeyMissing,
    ChatAPIQuotaExceeded,
    ChatAPIResponseError,
    JevAPIKeyMissing,
    JevAuthError,
    JevVersionDrift,
)
from jevemu.eval.bank import IDK_TEXT, QUESTION_KEY
from jevemu.eval.splits import DEFAULT_SCREEN_SIZE, FrozenSplit, SplitName, freeze_split
from jevemu.jev_client import JevCallRecord, JevClient
from jevemu.types import (
    Answer,
    ChoiceQuestion,
    EmulatorDiagnostics,
    SystemOneClient,
    SystemOneRequest,
    Usage,
)

__all__ = [
    "FATAL_ERRORS",
    "SPLITS_DIR",
    "ItemError",
    "ItemRecord",
    "RecordFormatError",
    "RunResult",
    "answer_item",
    "frozen_split",
    "idk_key",
    "item_request",
    "load_records",
    "records_path",
    "run_split",
]

logger = logging.getLogger(__name__)

SPLITS_DIR = "splits"
FATAL_ERRORS: tuple[type[Exception], ...] = (
    JevAuthError,
    JevAPIKeyMissing,
    JevVersionDrift,
    ChatAPIAuthError,
    ChatAPIKeyMissing,
    ChatAPIQuotaExceeded,
)
"""Errors no other item can avoid: the run stops and re-raises them."""
_STOP_ERRORS: tuple[type[Exception], ...] = (BudgetExceeded, *FATAL_ERRORS)

_ERROR_MESSAGE_LIMIT = 1000
_PROGRESS_EVERY = 500


class RecordFormatError(ValueError):
    """A records file has a malformed line other than a torn last line."""


class ItemError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    """Exception class name, or ``"AnswerMismatch"`` for a response without a matching answer."""
    message: str
    status: int | None = None
    """HTTP status for Jev HTTP errors."""


class ItemRecord(BaseModel):
    """One item's outcome: exactly one of ``answer`` and ``error`` is set."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    question_type: QuestionType
    gold: Gold
    idk_key: str | None = None
    """The "I don't know" option's key for the IDK benchmarks' choice questions."""
    answer: Answer | None = None
    model: str | None = None
    """``SystemOneResponse.model``."""
    latency_ms: float | None = None
    wall_ms: float | None = None
    usage: Usage | None = None
    cost_usd: float | None = None
    """Jev and paid-API emulators: spend incurred by this call (0 for a cache hit)."""
    cached: bool | None = None
    """Jev and paid-API emulators: served from the response cache."""
    diagnostics: EmulatorDiagnostics | None = None
    """Emulator only (``include_diagnostics=True``): ``x_jevemu`` for the question."""
    error: ItemError | None = None

    @model_validator(mode="after")
    def _answer_or_error(self) -> ItemRecord:
        if (self.answer is None) == (self.error is None):
            raise ValueError("a record has exactly one of answer and error")
        return self

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class RunResult:
    """Outcome of one :func:`run_split` call; counts refer to the requested items."""

    system_id: str
    benchmark: str
    split: SplitName
    path: Path
    status: RunStatus
    n_items: int
    n_ok: int
    n_errors: int
    """Requested items whose latest attempt failed."""
    n_called: int
    """Items sent to the system by this call."""
    n_skipped: int
    """Items already answered by an earlier call."""
    spent_usd: float
    """Spend of this call's records."""
    elapsed_s: float
    stop_reason: str | None


def records_path(out_dir: str | Path, system_id: str, benchmark: str, split: SplitName) -> Path:
    return Path(out_dir) / system_id / f"{run_key(benchmark, split)}.jsonl"


def item_request(item: BenchItem) -> SystemOneRequest:
    return SystemOneRequest(state=item.state, questions={QUESTION_KEY: item.question})


def idk_key(item: BenchItem) -> str | None:
    """The key of the "I don't know" option of an IDK bank item, else ``None``."""
    if not (isinstance(item.question, ChoiceQuestion) and item.metadata.get("idk") is True):
        return None
    keys = [key for key, text in item.question.criteria.items() if text == IDK_TEXT]
    return keys[-1] if keys else None


def frozen_split(
    out_dir: str | Path,
    benchmark: str | BenchmarkSpec,
    split: SplitName,
    *,
    seed: int = 0,
    screen_size: int = DEFAULT_SCREEN_SIZE,
) -> FrozenSplit:
    """``out_dir/splits/<benchmark>.<split>.jsonl``, frozen from the dataset on first use.

    An existing file wins over the dataset (so later runs pair with earlier ones) but must
    have been frozen with the same ``seed`` (and ``screen_size`` for ``screen``).
    """
    spec = get_benchmark(benchmark) if isinstance(benchmark, str) else benchmark
    path = Path(out_dir) / SPLITS_DIR / f"{run_key(spec.name, split)}.jsonl"
    if path.exists():
        frozen = FrozenSplit.from_jsonl(path)
        expected = (spec.name, split, seed, screen_size if split == "screen" else None)
        meta = frozen.metadata
        found = (meta.benchmark, meta.split, meta.seed, meta.screen_size)
        if found != expected:
            raise ValueError(
                f"{path} holds (benchmark, split, seed, screen_size) {found}, "
                f"requested {expected}; use another out_dir"
            )
        return frozen
    frozen = freeze_split(spec, split, seed=seed, screen_size=screen_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # runs may share out_dir
    frozen.to_jsonl(tmp)
    os.replace(tmp, path)
    return frozen


def load_records(path: str | Path) -> dict[str, ItemRecord]:
    """Item id -> latest record, a success winning over any error; ``{}`` without a file.

    A torn last line (an interrupted write) is ignored; any other malformed line raises
    :class:`RecordFormatError`.
    """
    path = Path(path)
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").split("\n")
    records: dict[str, ItemRecord] = {}
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = ItemRecord.model_validate_json(line)
        except ValidationError as exc:
            if number == len(lines):
                logger.warning("%s:%d: ignoring a torn last line", path, number)
                continue
            raise RecordFormatError(f"{path}:{number}: {exc}") from exc
        previous = records.get(record.item_id)
        if previous is None or record.ok or not previous.ok:
            records[record.item_id] = record
    return records


def _write_records(path: Path, records: Iterable[ItemRecord]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(record.model_dump_json() + "\n")
    os.replace(tmp, path)


def _item_error(exc: Exception) -> ItemError:
    message = str(exc)
    if len(message) > _ERROR_MESSAGE_LIMIT:
        message = message[:_ERROR_MESSAGE_LIMIT] + "..."
    status = getattr(exc, "status_code", None)
    return ItemError(
        type=type(exc).__name__,
        message=message,
        status=status if isinstance(status, int) else None,
    )


def _billed(exc: BaseException) -> float | None:
    """What the paid-API calls behind a failed item were billed (``None`` if none was)."""
    seen: BaseException | None = exc
    while seen is not None:
        if isinstance(seen, ChatAPIResponseError):
            return seen.cost_usd or None
        seen = seen.__cause__
    return None


async def answer_item(system: SystemOneClient, item: BenchItem) -> ItemRecord:
    """Ask ``system`` one item. Budget and :data:`FATAL_ERRORS` propagate; any other failure
    becomes an error record."""
    request = item_request(item)
    fields = {
        "item_id": item.item_id,
        "question_type": item.question.type,
        "gold": item.gold,
        "idk_key": idk_key(item),
    }
    meta: JevCallRecord | None = None
    started = time.perf_counter()
    try:
        if isinstance(system, JevClient):
            response, meta = await system.system_one_with_meta(request)
        else:
            response = await system.system_one(request)
    except _STOP_ERRORS:
        raise
    except Exception as exc:
        cause = exc.__cause__ if isinstance(exc, QuestionError) else None
        if isinstance(cause, _STOP_ERRORS):
            raise cause from exc
        wall_ms = (time.perf_counter() - started) * 1000.0
        return ItemRecord(**fields, wall_ms=wall_ms, cost_usd=_billed(exc), error=_item_error(exc))
    wall_ms = (time.perf_counter() - started) * 1000.0
    answer = response.answers.get(QUESTION_KEY)
    if answer is None or answer.type != item.question.type:
        found = "none" if answer is None else f"a {answer.type} answer"
        return ItemRecord(
            **fields,
            model=response.model,
            wall_ms=wall_ms,
            error=ItemError(
                type="AnswerMismatch",
                message=f"expected a {item.question.type} answer under {QUESTION_KEY!r}, "
                f"got {found}",
            ),
        )
    diagnostics = (response.x_jevemu or {}).get(QUESTION_KEY)
    api = diagnostics.api if diagnostics is not None else None
    if meta is not None:
        latency_ms, cost_usd, cached = meta.latency_ms, meta.cost_usd, meta.cached
    elif api is not None:
        latency_ms, cost_usd, cached = (
            api.latency_ms,
            api.cost_usd,
            api.response_cache_hits == api.calls,
        )
    else:
        latency_ms, cost_usd, cached = wall_ms, None, None
    return ItemRecord(
        **fields,
        answer=answer,
        model=response.model,
        latency_ms=latency_ms,
        wall_ms=meta.wall_ms if meta else wall_ms,
        usage=response.usage,
        cost_usd=cost_usd,
        cached=cached,
        diagnostics=diagnostics,
    )


def _open_manifest(run_dir: Path, system_id: str, description: dict[str, Any]) -> RunManifest:
    manifest = RunManifest.load(run_dir)
    if manifest is None:
        return RunManifest(system_id=system_id, system=description, jevemu_version=jevemu_version())
    if manifest.system_id != system_id:
        raise ValueError(f"{run_dir} belongs to system_id {manifest.system_id!r}")
    differences = system_differences(manifest.system, description)
    if differences:
        raise ValueError(
            f"{run_dir} was run by a different system (fields {differences} differ): "
            f"recorded {manifest.system}, now {description}; use another system_id"
        )
    return manifest


def _counts(items: Iterable[BenchItem], records: Mapping[str, ItemRecord]) -> tuple[int, int]:
    """(ok, error) among ``items``."""
    ok = errors = 0
    for item in items:
        record = records.get(item.item_id)
        if record is not None:
            ok += record.ok
            errors += not record.ok
    return ok, errors


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def run_split(
    system: SystemOneClient,
    system_id: str,
    benchmark: str | BenchmarkSpec,
    split: SplitName,
    out_dir: str | Path,
    *,
    concurrency: int,
    seed: int = 0,
    screen_size: int = DEFAULT_SCREEN_SIZE,
    limit: int | None = None,
) -> RunResult:
    """Answer ``split`` of ``benchmark`` (its first ``limit`` items) with ``system``, at most
    ``concurrency`` items in flight, resuming ``out_dir/<system_id>`` (module docstring).

    Run one split of a run directory at a time: the manifest is read and rewritten by each call.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    if not system_id or system_id != Path(system_id).name or system_id == SPLITS_DIR:
        raise ValueError(f"system_id must be a plain directory name, got {system_id!r}")
    out = Path(out_dir)
    frozen = frozen_split(out, benchmark, split, seed=seed, screen_size=screen_size)
    name = frozen.metadata.benchmark
    items = list(frozen.items if limit is None else frozen.items[:limit])
    run_dir = out / system_id
    manifest = _open_manifest(run_dir, system_id, await describe_system(system))
    key = run_key(name, split)
    previous = manifest.runs.get(key)
    if previous is not None and previous.split_metadata != frozen.metadata:
        raise ValueError(
            f"{run_dir}: {key} was run on another split (items_sha256 "
            f"{previous.split_metadata.items_sha256} != {frozen.metadata.items_sha256})"
        )

    path = run_dir / f"{key}.jsonl"
    records = load_records(path)
    order = [item.item_id for item in frozen.items]
    foreign = set(records) - set(order)
    if foreign:
        raise ValueError(f"{path} holds {len(foreign)} records of items outside the split")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_records(path, (records[i] for i in order if i in records))

    todo = [item for item in items if item.item_id not in records or not records[item.item_id].ok]
    commit, dirty = git_state()
    invocation = Invocation(
        started_at=_now(), git_commit=commit, git_dirty=dirty, concurrency=concurrency, limit=limit
    )
    entry = SplitRun(
        benchmark=name,
        split=split,
        split_metadata=frozen.metadata,
        status="running",
        n_items=len(items),
        n_ok=0,
        n_errors=0,
        spent_usd=0.0,
        invocations=[*(previous.invocations if previous else []), invocation],
    )
    manifest.runs[key] = entry

    def save(status: RunStatus) -> None:
        invocation.status = entry.status = status
        entry.n_ok, entry.n_errors = _counts(items, records)
        entry.spent_usd = sum(r.cost_usd or 0.0 for r in records.values())
        manifest.save(run_dir)

    save("running")
    logger.info("%s %s: %d items, %d to answer", system_id, key, len(items), len(todo))

    started = time.perf_counter()
    stop: tuple[RunStatus, str] | None = None
    fatal: Exception | None = None
    pending = iter(todo)

    async def worker(sink: _Sink) -> None:
        nonlocal stop, fatal
        for item in pending:
            if stop is not None:
                return
            try:
                record = await answer_item(system, item)
            except BudgetExceeded as exc:
                if stop is None:
                    stop = ("stopped_budget", f"{type(exc).__name__}: {exc}")
                return
            except FATAL_ERRORS as exc:
                if stop is None:
                    stop, fatal = ("failed", f"{type(exc).__name__}: {exc}"), exc
                return
            own_usd = record.cost_usd or 0.0
            failed = records.get(item.item_id)
            if failed is not None and not failed.ok and failed.cost_usd:
                record = record.model_copy(update={"cost_usd": own_usd + failed.cost_usd})
            records[item.item_id] = record
            sink.write(record)
            invocation.n_called += 1
            invocation.n_errors += not record.ok
            invocation.n_cache_hits += bool(record.cached)
            invocation.spent_usd += own_usd
            if invocation.n_called % _PROGRESS_EVERY == 0:
                elapsed = time.perf_counter() - started
                logger.info(
                    "%s %s: %d/%d answered (%d errors), %.1f items/s, $%.6f",
                    system_id,
                    key,
                    invocation.n_called,
                    len(todo),
                    invocation.n_errors,
                    invocation.n_called / elapsed if elapsed > 0 else 0.0,
                    invocation.spent_usd,
                )

    status: RunStatus = "incomplete"
    try:
        with _Sink(path) as sink:
            await asyncio.gather(*(worker(sink) for _ in range(min(concurrency, len(todo)))))
        if stop is not None:
            status, invocation.stop_reason = stop
        else:
            n_ok, _ = _counts(items, records)
            status = "complete" if n_ok == len(items) else "incomplete"
    finally:
        _write_records(path, (records[i] for i in order if i in records))
        invocation.finished_at = _now()
        save(status)
    elapsed_s = time.perf_counter() - started
    logger.info(
        "%s %s: %s, %d/%d ok, %d errors, %d called, $%.6f in %.1f s",
        system_id,
        key,
        status,
        entry.n_ok,
        len(items),
        entry.n_errors,
        invocation.n_called,
        invocation.spent_usd,
        elapsed_s,
    )
    if fatal is not None:
        raise fatal
    return RunResult(
        system_id=system_id,
        benchmark=name,
        split=split,
        path=path,
        status=status,
        n_items=len(items),
        n_ok=entry.n_ok,
        n_errors=entry.n_errors,
        n_called=invocation.n_called,
        n_skipped=len(items) - len(todo),
        spent_usd=invocation.spent_usd,
        elapsed_s=elapsed_s,
        stop_reason=invocation.stop_reason,
    )


class _Sink:
    """Appends records to a file, flushing each so an interruption keeps finished items."""

    def __init__(self, path: Path) -> None:
        self._file = path.open("a", encoding="utf-8")

    def __enter__(self) -> _Sink:
        return self

    def __exit__(self, *exc: object) -> None:
        self._file.close()

    def write(self, record: ItemRecord) -> None:
        self._file.write(record.model_dump_json() + "\n")
        self._file.flush()
