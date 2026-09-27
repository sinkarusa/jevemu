"""Your own labelled data as a benchmark: JSONL or CSV plus a column mapping.

The mapping names the columns holding the state, the gold label and (optionally) a row ID,
instructions, criteria and extra metadata. Instructions and criteria can instead be fixed for
the whole file, which suits classification data (one question, many states).

Gold per question type: Choice, a criteria key; Score, a level index (``0``..``K-1``); Noul,
a boolean (``true``/``false``, ``yes``/``no`` or ``1``/``0`` in CSV). In CSV every cell is text;
a criteria column holds JSON. Any malformed row raises :class:`CustomDataError` naming the file,
the line and the problem.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, model_validator

from jevemu.bench.spec import BenchConfig, BenchItem, BenchmarkSpec, QuestionType
from jevemu.types import Description, Question

CustomFormat = Literal["jsonl", "csv"]

_QUESTION = TypeAdapter[Question](Question)
_INT = re.compile(r"[+-]?\d+")
_NOUL_TEXT = {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False}


class CustomDataError(ValueError):
    """A custom benchmark file or one of its rows is malformed."""


class ColumnMapping(BaseModel):
    """Which columns hold what. ``instructions``/``criteria`` fix a value for every row;
    ``instructions_column``/``criteria_column`` read it per row (at most one of each pair)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_type: QuestionType
    state: str
    gold: str
    id: str | None = None
    """Column with a unique row ID; default is the 1-based row number."""
    instructions: Description = None
    instructions_column: str | None = None
    criteria: dict[str, Description] | list[Description] | None = None
    """Choice: ``{key: description}`` or a list of keys. Score: a list of levels. Noul:
    ``{"true": ..., "false": ...}`` or omitted."""
    criteria_column: str | None = None
    metadata: tuple[str, ...] = ()
    """Extra columns copied into ``BenchItem.metadata``."""

    @model_validator(mode="after")
    def _coherent(self) -> ColumnMapping:
        if self.instructions is not None and self.instructions_column is not None:
            raise ValueError("set instructions or instructions_column, not both")
        if self.criteria is not None and self.criteria_column is not None:
            raise ValueError("set criteria or criteria_column, not both")
        needs_criteria = self.question_type in ("choice", "score")
        if needs_criteria and self.criteria is None and self.criteria_column is None:
            raise ValueError(f"{self.question_type} questions need criteria or criteria_column")
        if self.criteria is not None:
            _build_question(self.question_type, self.instructions, self.criteria)
        return self

    def columns(self) -> list[str]:
        optional = (self.id, self.instructions_column, self.criteria_column)
        return [self.state, self.gold, *(c for c in optional if c is not None), *self.metadata]


def _choice_criteria(value: Any) -> Any:
    if isinstance(value, list) and all(isinstance(key, str) for key in value):
        if len(set(value)) != len(value):
            raise ValueError(f"duplicate option in {value}")
        return dict.fromkeys(value)
    return value


def _build_question(question_type: QuestionType, instructions: Any, criteria: Any) -> Question:
    payload: dict[str, Any] = {"type": question_type, "instructions": instructions}
    if criteria is not None:
        payload["criteria"] = _choice_criteria(criteria) if question_type == "choice" else criteria
    try:
        return _QUESTION.validate_python(payload)
    except ValidationError as exc:
        # Drop the union tag ("choice", ...) from each error location.
        raise ValueError(f"invalid {question_type} question: {_brief(exc, skip=1)}") from None


def _brief(exc: ValidationError, *, skip: int = 0) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc'][skip:]) or 'value'}: {err['msg']}"
        for err in exc.errors()
    )


def _gold(question_type: QuestionType, value: Any) -> str | int | bool:
    if question_type == "choice":
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
    elif question_type == "score":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and _INT.fullmatch(value.strip()):
            return int(value)
    else:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in _NOUL_TEXT:
            return _NOUL_TEXT[value.strip().lower()]
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
    expected = {"choice": "a criteria key", "score": "a level index", "noul": "a boolean"}
    raise ValueError(f"gold {value!r} is not {expected[question_type]}")


def _items(
    records: Iterable[tuple[str, Mapping[str, Any]]],
    mapping: ColumnMapping,
    *,
    name: str,
    path: Path,
    decode_criteria: bool,
    limit: int | None,
) -> list[BenchItem]:
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1 or None, got {limit}")
    items: list[BenchItem] = []
    first_seen: dict[str, str] = {}
    for number, (where, record) in enumerate(records, start=1):
        if limit is not None and len(items) >= limit:
            break
        try:
            item = _item(record, mapping, name=name, number=number, decode=decode_criteria)
        except ValueError as exc:
            raise CustomDataError(f"{where}: {exc}") from None
        if item.item_id in first_seen:
            raise CustomDataError(
                f"{where}: duplicate id {item.item_id!r} (first at {first_seen[item.item_id]})"
            )
        first_seen[item.item_id] = where
        items.append(item)
    if not items:
        raise CustomDataError(f"{path}: no rows")
    return items


def _item(
    record: Mapping[str, Any], mapping: ColumnMapping, *, name: str, number: int, decode: bool
) -> BenchItem:
    def cell(column: str) -> Any:
        if column not in record:
            raise ValueError(f"missing column {column!r}")
        return record[column]

    state = cell(mapping.state)
    if not isinstance(state, (str, dict, list)):
        raise ValueError(f"state must be text or JSON, got {state!r}")
    instructions = (
        mapping.instructions
        if mapping.instructions_column is None
        else cell(mapping.instructions_column)
    )
    criteria = mapping.criteria
    if mapping.criteria_column is not None:
        criteria = cell(mapping.criteria_column)
        if decode:
            try:
                criteria = json.loads(criteria)
            except json.JSONDecodeError as exc:
                raise ValueError(f"criteria column {mapping.criteria_column!r}: {exc}") from None
    question = _build_question(mapping.question_type, instructions, criteria)
    row_id: Any = str(number) if mapping.id is None else cell(mapping.id)
    if isinstance(row_id, int) and not isinstance(row_id, bool):
        row_id = str(row_id)
    if not isinstance(row_id, str) or not row_id:
        raise ValueError(f"id {row_id!r} must be a non-empty string or an integer")
    gold = _gold(mapping.question_type, cell(mapping.gold))
    try:
        return BenchItem(
            item_id=f"{name}:{row_id}",
            state=state,
            question=question,
            gold=gold,
            metadata={column: cell(column) for column in mapping.metadata},
        )
    except ValidationError as exc:
        raise ValueError(_brief(exc)) from None


def _jsonl_records(path: Path) -> Iterator[tuple[str, Mapping[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            where = f"{path}:{line_no}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CustomDataError(f"{where}: invalid JSON: {exc}") from None
            if not isinstance(record, dict):
                raise CustomDataError(f"{where}: expected a JSON object per line")
            yield where, record


def _csv_records(path: Path, mapping: ColumnMapping) -> Iterator[tuple[str, Mapping[str, Any]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [c for c in mapping.columns() if c not in header]
        if missing:
            raise CustomDataError(f"{path}: missing columns {missing}; header is {header}")
        for record in reader:
            where = f"{path}:{reader.line_num}"
            if None in record:
                raise CustomDataError(f"{where}: more fields than the header")
            if any(value is None for value in record.values()):
                raise CustomDataError(f"{where}: fewer fields than the header")
            yield where, record


def custom_jsonl(
    path: str | Path, mapping: ColumnMapping, *, name: str, limit: int | None = None
) -> list[BenchItem]:
    """Load a JSONL file (one object per line; blank lines skipped) as benchmark items."""
    path = Path(path)
    return _items(
        _jsonl_records(path), mapping, name=name, path=path, decode_criteria=False, limit=limit
    )


def custom_csv(
    path: str | Path, mapping: ColumnMapping, *, name: str, limit: int | None = None
) -> list[BenchItem]:
    """Load a CSV file with a header row as benchmark items."""
    path = Path(path)
    return _items(
        _csv_records(path, mapping),
        mapping,
        name=name,
        path=path,
        decode_criteria=True,
        limit=limit,
    )


def custom_benchmark(
    name: str,
    path: str | Path,
    mapping: ColumnMapping,
    *,
    license: str,
    format: CustomFormat | None = None,
    tags: Iterable[str] = (),
) -> BenchmarkSpec:
    """A spec for a custom file (format from the suffix unless given); not registered.

    Pass it to ``load_benchmark`` directly, or add it to a registry to list it.
    """
    path = Path(path)
    if format is None:
        suffix = path.suffix.lower()
        if suffix in (".jsonl", ".ndjson"):
            format = "jsonl"
        elif suffix == ".csv":
            format = "csv"
        else:
            raise ValueError(f"cannot infer the format of {path}; pass format='jsonl' or 'csv'")
    reader = custom_jsonl if format == "jsonl" else custom_csv

    def load(cfg: BenchConfig) -> list[BenchItem]:
        if cfg.params:
            raise ValueError(f"custom benchmark {name!r} takes no params, got {dict(cfg.params)}")
        return reader(path, mapping, name=name, limit=cfg.limit)

    return BenchmarkSpec(
        name=name,
        question_type=mapping.question_type,
        loader=load,
        splits={"test": "all"},
        license=license,
        tags=frozenset({"custom", *tags}),
    )
