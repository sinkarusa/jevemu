"""Plumbing shared by the built-in dataset adapters.

Each adapter reads one file from the Hugging Face Hub, pinned to a commit and checked against a
hash (:func:`jevemu.eval.bank.fetch_pinned`), and maps its rows to benchmark items in file
order. States are the dataset's text, verbatim. Every item's metadata records:

- ``question_id``: the dataset's own ID, else the 0-based row index in the pinned file;
- ``stratum``: the subject or gold class :mod:`jevemu.eval.splits` stratifies on;
- ``revision`` and ``source_sha256``: the pinned commit and the file's SHA-256.

The adapters take no params and ignore ``cfg.seed``; ``cfg.limit`` keeps the first items.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import ValidationError

from jevemu.bench.spec import BenchConfig, BenchItem, Gold
from jevemu.eval.bank import (
    OPTION_LETTERS,
    DatasetFormatError,
    DatasetSource,
    PinnedFile,
    fetch_pinned,
    instructions_for,
)
from jevemu.types import ChoiceQuestion, Description, JSONish, Question

Parse = Callable[[Path, DatasetSource], list[BenchItem]]
"""Maps a downloaded dataset file (and where it came from) to every benchmark item."""


def load_items(benchmark: str, pin: PinnedFile, parse: Parse, cfg: BenchConfig) -> list[BenchItem]:
    """Fetch ``pin``'s file, parse it, and keep the first ``cfg.limit`` items."""
    if cfg.params:
        raise ValueError(f"{benchmark} takes no params, got {sorted(cfg.params)}")
    if cfg.limit is not None and cfg.limit < 1:
        raise ValueError(f"limit must be >= 1 or None, got {cfg.limit}")
    path, sha256 = fetch_pinned(pin)
    items = parse(path, pin.source(sha256))
    return items if cfg.limit is None else items[: cfg.limit]


def parquet_records(path: Path, columns: Sequence[str]) -> list[dict[str, Any]]:
    """The given columns of every row, in file order."""
    names = pq.read_schema(path).names
    missing = [c for c in columns if c not in names]
    if missing:
        raise DatasetFormatError(f"{path}: missing columns {missing}")
    records: list[dict[str, Any]] = pq.read_table(path, columns=list(columns)).to_pylist()
    return records


def jsonl_records(path: Path, keys: Sequence[str]) -> list[dict[str, Any]]:
    """One JSON object per non-blank line, each holding at least ``keys``."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetFormatError(f"{path}:{number}: not JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise DatasetFormatError(f"{path}:{number}: not a JSON object")
            missing = [k for k in keys if k not in record]
            if missing:
                raise DatasetFormatError(f"{path}:{number}: missing keys {missing}")
            records.append(record)
    return records


def class_names(path: Path, column: str) -> tuple[str, ...]:
    """The ``ClassLabel`` names Hugging Face stores in a parquet file's schema metadata."""
    metadata = pq.read_schema(path).metadata or {}
    try:
        names = json.loads(metadata[b"huggingface"])["info"]["features"][column]["names"]
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetFormatError(f"{path}: no ClassLabel names for column {column!r}") from exc
    if not (
        isinstance(names, list)
        and names
        and all(isinstance(name, str) and name for name in names)
        and len(set(names)) == len(names)
    ):
        raise DatasetFormatError(f"{path}: malformed ClassLabel names for column {column!r}")
    return tuple(names)


def label_index(value: Any, n: int, where: str) -> int:
    """``value`` as a class index in ``0..n-1``."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < n:
        raise DatasetFormatError(f"{where}: label {value!r} is not in 0..{n - 1}")
    return value


def mcq_question(options: Any, where: str) -> ChoiceQuestion:
    """A multiple-choice question in the question banks' form (without IDK): options keyed
    ``A``, ``B``, ... in the given order, with the banks' instructions."""
    if not (isinstance(options, list) and all(isinstance(option, str) for option in options)):
        raise DatasetFormatError(f"{where}: options must be a list of strings")
    if not 2 <= len(options) <= len(OPTION_LETTERS):
        raise DatasetFormatError(f"{where}: {len(options)} options")
    return ChoiceQuestion(
        instructions=instructions_for(len(options), idk=False),
        criteria={OPTION_LETTERS[i]: option for i, option in enumerate(options)},
    )


def classification_items(
    benchmark: str,
    path: Path,
    source: DatasetSource,
    *,
    text_column: str,
    label_column: str,
    instructions: str,
    exclude: frozenset[str] = frozenset(),
) -> list[BenchItem]:
    """One Choice item per row: the text is the state, the file's class names are the option
    keys (no descriptions, in the file's label order) and the gold. Rows whose class is in
    ``exclude`` are dropped, and those classes are not options."""
    names = class_names(path, label_column)
    criteria: dict[str, Description] = {name: None for name in names if name not in exclude}
    question = ChoiceQuestion(instructions=instructions, criteria=criteria)
    items: list[BenchItem] = []
    for index, record in enumerate(parquet_records(path, (text_column, label_column))):
        where = f"{path}: row {index}"
        gold = names[label_index(record[label_column], len(names), where)]
        if gold in exclude:
            continue
        items.append(
            make_item(
                benchmark,
                str(index),
                state=nonblank(record[text_column], where),
                question=question,
                gold=gold,
                stratum=gold,
                source=source,
                where=where,
            )
        )
    return items


def nonblank(value: Any, where: str) -> str:
    """A non-blank dataset string, returned verbatim."""
    if not isinstance(value, str) or not value.strip():
        raise DatasetFormatError(f"{where}: expected non-empty text, got {value!r}")
    return value


def make_item(
    benchmark: str,
    question_id: str,
    *,
    state: JSONish,
    question: Question,
    gold: Gold,
    stratum: str,
    source: DatasetSource,
    where: str,
    **extra: Any,
) -> BenchItem:
    """``BenchItem`` ``"{benchmark}:{question_id}"`` with the shared metadata plus ``extra``."""
    try:
        return BenchItem(
            item_id=f"{benchmark}:{question_id}",
            state=state,
            question=question,
            gold=gold,
            metadata={
                "dataset": benchmark,
                "question_id": question_id,
                "stratum": stratum,
                "revision": source.revision,
                "source_sha256": source.sha256,
                **extra,
            },
        )
    except ValidationError as exc:
        raise DatasetFormatError(f"{where}: {exc}") from exc
