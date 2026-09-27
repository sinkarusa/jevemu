"""Frozen ``select``/``holdout`` halves of every benchmark, and a small ``screen`` subset.

Model and configuration choices are made on ``select``; calibrators are fitted, and final numbers
reported, on ``holdout``, which selection never sees. ``screen`` is a cheap stratified subset of
``select`` for first-pass sweeps.

- **Halves.** Each benchmark is split on its own, so adding a benchmark never moves another's
  items. Items are grouped by stratum (``metadata["stratum"]`` from the built-in adapters, else
  the question banks' ``metadata["subject"]``), each stratum is ordered by
  ``sha256(f"{seed}\\x1f{benchmark}\\x1f{question_id}")``, and the strata, in name order,
  alternate ``select``, ``holdout``, ``select``, ... with the alternation carried from one
  stratum to the next. Each stratum and the whole benchmark are then halved exactly (the halves
  differ by at most one item), and an item's half does not depend on the order it was loaded in.
- **Screen.** At most ``screen_size`` items of ``select`` (all of it when smaller). Each stratum
  gets its share of ``screen_size`` by largest remainder (ties broken by a seeded hash of the
  stratum name) and contributes its first items in the hash order above.

Items are returned in the benchmark's own order. Benchmarks load with their default config (the
question banks: evaluate-idk option order with "I don't know"); ``seed`` only moves items between
halves. :func:`freeze_split` writes a split as a ``split.jsonl`` for paired runs: a
``{"metadata": ...}`` line (benchmark, split, seed, dataset revision and file hash,
``items_sha256``), then one :class:`BenchItem` per line.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from jevemu.bench import (
    BenchConfig,
    BenchItem,
    BenchmarkSpec,
    QuestionType,
    get_benchmark,
    load_benchmark,
)

SplitName = Literal["select", "holdout", "screen"]
SPLITS: tuple[SplitName, ...] = ("select", "holdout", "screen")

DATASETS: tuple[str, ...] = (
    "gpqa_diamond_idk",
    "lexam_en_idk",
    "mmlu_pro",
    "arc_challenge",
    "ag_news",
    "banking77",
    "clinc150",
    "boolq",
    "sst5",
)
"""Every built-in benchmark the select/holdout protocol covers."""

DROPPED_DATASETS: tuple[str, ...] = ("yelp_stars",)
"""Built-in benchmarks taken out of the protocol (2026-09-25, user decision): Yelp's levels
("1 star" .. "5 stars") are shown to the models as digits 0-4 and read ambiguously. Their
adapters stay registered and can be run and scored when named explicitly, but summaries,
comparisons, statistics and calibration studies skip their records otherwise."""

DEFAULT_SCREEN_SIZE = 300
SPLIT_VERSION = "1"
"""Bumped whenever the same inputs would put an item in a different split."""


class SplitFormatError(ValueError):
    """A ``split.jsonl`` file is malformed or was modified after it was written."""


def split_items(
    benchmark: str | BenchmarkSpec,
    split: SplitName,
    *,
    seed: int = 0,
    screen_size: int = DEFAULT_SCREEN_SIZE,
) -> list[BenchItem]:
    """The items of ``benchmark`` in ``split`` (see the module docstring)."""
    spec = get_benchmark(benchmark) if isinstance(benchmark, str) else benchmark
    return split_of(
        spec.name, load_benchmark(spec, BenchConfig()), split, seed=seed, screen_size=screen_size
    )


def split_of(
    benchmark: str,
    items: Sequence[BenchItem],
    split: SplitName,
    *,
    seed: int = 0,
    screen_size: int = DEFAULT_SCREEN_SIZE,
) -> list[BenchItem]:
    """``split`` of an already loaded benchmark: ``items`` must be all of its items."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if screen_size < 1:
        raise ValueError(f"screen_size must be >= 1, got {screen_size}")
    strata = _ranked_strata(benchmark, items, seed)
    select: dict[str, list[int]] = {}
    holdout: dict[str, list[int]] = {}
    turn = 0
    for name, ranked in strata.items():
        select[name] = ranked[turn::2]
        holdout[name] = ranked[1 - turn :: 2]
        turn = (turn + len(ranked)) % 2
    if split == "holdout":
        chosen = holdout
    elif split == "select":
        chosen = select
    else:
        chosen = _screen(benchmark, select, seed, screen_size)
    keep = {index for indices in chosen.values() for index in indices}
    return [item for index, item in enumerate(items) if index in keep]


def _ranked_strata(benchmark: str, items: Sequence[BenchItem], seed: int) -> dict[str, list[int]]:
    """Stratum name (sorted) -> item indices in seeded-hash order."""
    by_stratum: defaultdict[str, list[int]] = defaultdict(list)
    ranks: list[bytes] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        question_id = question_id_of(item)
        if question_id in seen:
            raise ValueError(f"{benchmark}: duplicate question_id {question_id!r}")
        seen.add(question_id)
        ranks.append(_hash(seed, benchmark, question_id))
        by_stratum[stratum_of(item)].append(index)
    return {name: sorted(by_stratum[name], key=ranks.__getitem__) for name in sorted(by_stratum)}


def _screen(
    benchmark: str, select: dict[str, list[int]], seed: int, size: int
) -> dict[str, list[int]]:
    total = sum(len(indices) for indices in select.values())
    if total <= size:
        return select
    quotas = {name: size * len(indices) // total for name, indices in select.items()}
    remainders = sorted(
        select,
        key=lambda name: (-(size * len(select[name]) % total), _hash(seed, benchmark, name)),
    )
    for name in remainders[: size - sum(quotas.values())]:
        quotas[name] += 1
    return {name: indices[: quotas[name]] for name, indices in select.items()}


def _hash(seed: int, benchmark: str, key: str) -> bytes:
    return hashlib.sha256(f"{seed}\x1f{benchmark}\x1f{key}".encode()).digest()


def question_id_of(item: BenchItem) -> str:
    """``metadata["question_id"]`` (the dataset's own id), else the item id."""
    question_id = item.metadata.get("question_id")
    return question_id if isinstance(question_id, str) else item.item_id


def stratum_of(item: BenchItem) -> str:
    """``metadata["stratum"]`` (built-in adapters), else ``metadata["subject"]`` (question
    banks), else ``""``."""
    stratum = item.metadata.get("stratum", item.metadata.get("subject"))
    return "" if stratum is None else str(stratum)


# --- frozen split files ----------------------------------------------------------------------


class SplitMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    split_version: str
    benchmark: str
    question_type: QuestionType
    license: str
    split: SplitName
    seed: int
    screen_size: int | None
    """Only for ``screen``."""
    revision: str
    """Commit of the dataset repo the items came from."""
    source_sha256: str
    """SHA-256 of the dataset file the items came from."""
    n_benchmark: int = Field(ge=0)
    """Items in the whole benchmark."""
    n_items: int = Field(ge=0)
    items_sha256: str
    """SHA-256 over the item lines; :meth:`FrozenSplit.from_jsonl` rejects a file that fails it."""


def _item_lines(items: Sequence[BenchItem]) -> list[str]:
    return [item.model_dump_json() for item in items]


def _items_sha256(lines: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class FrozenSplit(BaseModel):
    """One benchmark split, frozen to a file so every system answers the identical items."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: SplitMetadata
    items: tuple[BenchItem, ...]

    @model_validator(mode="after")
    def _consistent(self) -> FrozenSplit:
        if self.metadata.n_items != len(self.items):
            raise ValueError(f"metadata n_items {self.metadata.n_items} != {len(self.items)} items")
        ids = [item.item_id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate item_id in split")
        if _items_sha256(_item_lines(self.items)) != self.metadata.items_sha256:
            raise ValueError("items_sha256 does not match the items")
        return self

    def dumps(self) -> str:
        """The ``split.jsonl`` text: a ``{"metadata": ...}`` line, then one item per line."""
        header = '{"metadata":' + self.metadata.model_dump_json() + "}"
        return "".join(f"{line}\n" for line in [header, *_item_lines(self.items)])

    def to_jsonl(self, path: str | Path) -> None:
        Path(path).write_bytes(self.dumps().encode("utf-8"))

    @classmethod
    def from_jsonl(cls, path: str | Path) -> FrozenSplit:
        # Only "\n" ends a line: JSON escapes it inside strings but leaves U+0085 and U+2028
        # raw (MMLU-Pro has a U+0085), which str.splitlines would also split on.
        lines = Path(path).read_text(encoding="utf-8").split("\n")
        if lines[-1] == "":
            lines.pop()
        prefix = '{"metadata":'
        if not (lines and lines[0].startswith(prefix) and lines[0].endswith("}")):
            raise SplitFormatError(f'{path}:1: first line must be {{"metadata": {{...}}}}')
        try:
            metadata = SplitMetadata.model_validate_json(lines[0][len(prefix) : -1])
        except ValidationError as exc:
            raise SplitFormatError(f"{path}:1: bad metadata line: {exc}") from exc
        items: list[BenchItem] = []
        for number, line in enumerate(lines[1:], start=2):
            try:
                items.append(BenchItem.model_validate_json(line))
            except ValidationError as exc:
                raise SplitFormatError(f"{path}:{number}: bad item: {exc}") from exc
        try:
            return cls(metadata=metadata, items=tuple(items))
        except ValidationError as exc:
            raise SplitFormatError(f"{path}: {exc}") from exc


def freeze_split(
    benchmark: str | BenchmarkSpec,
    split: SplitName,
    *,
    seed: int = 0,
    screen_size: int = DEFAULT_SCREEN_SIZE,
) -> FrozenSplit:
    """:func:`split_items` with the metadata to write it out (``FrozenSplit.to_jsonl``)."""
    spec = get_benchmark(benchmark) if isinstance(benchmark, str) else benchmark
    everything = load_benchmark(spec, BenchConfig())
    items = split_of(spec.name, everything, split, seed=seed, screen_size=screen_size)
    return FrozenSplit(
        metadata=SplitMetadata(
            split_version=SPLIT_VERSION,
            benchmark=spec.name,
            question_type=spec.question_type,
            license=spec.license,
            split=split,
            seed=seed,
            screen_size=screen_size if split == "screen" else None,
            revision=_one_source_value(spec.name, everything, "revision"),
            source_sha256=_one_source_value(spec.name, everything, "source_sha256"),
            n_benchmark=len(everything),
            n_items=len(items),
            items_sha256=_items_sha256(_item_lines(items)),
        ),
        items=tuple(items),
    )


def _one_source_value(benchmark: str, items: Sequence[BenchItem], key: str) -> str:
    values = {item.metadata.get(key) for item in items}
    if len(values) != 1:
        raise ValueError(f"{benchmark}: items must share one metadata[{key!r}], got {values}")
    (value,) = values
    if not isinstance(value, str):
        raise ValueError(f"{benchmark}: metadata[{key!r}] must be a string, got {value!r}")
    return value
