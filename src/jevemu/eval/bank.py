"""Frozen multiple-choice question banks mapped to Jev Choice requests.

A bank is the unit of fairness in the paired comparison: Jev and the emulator answer the
identical ``SystemOneRequest`` objects it holds, in the same option order. Each dataset file is
downloaded from the Hugging Face Hub at a pinned commit and checked against a pinned hash, and
both are recorded in the bank. The same arguments always give a byte-identical ``bank.jsonl``.

Two option orders are supported:

- ``per_question`` (default): the options are shuffled by a private RNG seeded from a stable
  hash of ``(seed, question_id)``, so a question's order does not depend on its row position.
- ``evaluate_idk``: evaluate-idk's order, reproduced exactly
  (``docs/research/evaluate_idk_audit.md`` §6, §11): one ``random.Random(42)`` stream per
  dataset and one ``shuffle`` per row, in dataset order. ``seed`` is ignored.

``n_permutations > 1`` adds cyclic shifts of that order. "I don't know" is always the last
option when present and is never the gold answer.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import random
import string
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq  # type: ignore[import-untyped]
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from jevemu.types import DEFAULT_MODEL, ChoiceQuestion, SystemOneRequest

BUILDER_VERSION = "1"
"""Bumped whenever the same inputs would produce different bank bytes."""

QUESTION_KEY = "answer"
IDK_TEXT = "I don't know"
EVALUATE_IDK_SEED = 42
"""evaluate-idk seeds its option stream with ``random.seed(42)`` (``custom_tasks.py:33``)."""

OPTION_LETTERS = string.ascii_uppercase
MAX_OPTIONS = len(OPTION_LETTERS) - 1
"""One letter stays free for the "I don't know" option."""

BankOrder = Literal["per_question", "evaluate_idk"]
BANK_ORDERS: tuple[BankOrder, ...] = ("per_question", "evaluate_idk")


class DatasetAccessError(RuntimeError):
    """The dataset cannot be downloaded with the current credentials (e.g. gated terms)."""


class DatasetIntegrityError(RuntimeError):
    """A downloaded dataset file does not match its pinned hash."""


class DatasetFormatError(ValueError):
    """A dataset file lacks a required column or holds a malformed row."""


class BankFormatError(ValueError):
    """A ``bank.jsonl`` file is malformed or was modified after it was written."""


def instructions_for(n_options: int, *, idk: bool) -> str:
    """The pre-registered Choice instructions for a bank question with ``n_options`` options."""
    base = "Which option correctly answers the question?"
    if not idk:
        return base
    return (
        f"{base} A wrong answer costs 1 point, a correct answer earns 1 point, "
        f"and {OPTION_LETTERS[n_options]} ({IDK_TEXT}) earns 0."
    )


def render_options(criteria: Mapping[str, str]) -> str:
    """Options as evaluate-idk renders them: ``"{key}) {text}"`` lines, no trailing newline.

    For an IDK bank in ``evaluate_idk`` order this is byte-identical to the block between
    ``"Choices:\\n"`` and ``"\\n\\nBefore answering"`` in evaluate-idk's ``doc.query``.
    """
    return "\n".join(f"{key}) {text}" for key, text in criteria.items())


def options_sha256(criteria: Mapping[str, str]) -> str:
    return hashlib.sha256(render_options(criteria).encode("utf-8")).hexdigest()


# --- dataset rows and sources ----------------------------------------------------------------


@dataclass(frozen=True)
class McqRow:
    """One multiple-choice question in the dataset's base option order."""

    question_id: str
    """The dataset's own ID (LEXam ``id``, GPQA ``Record ID``), not a row position."""
    question: str
    options: tuple[str, ...]
    gold_index: int
    """Index into ``options`` of the correct answer."""
    subject: str | None = None

    def __post_init__(self) -> None:
        if not self.question_id:
            raise ValueError("question_id must be non-empty")
        if not 2 <= len(self.options) <= MAX_OPTIONS:
            raise ValueError(
                f"question {self.question_id!r}: {len(self.options)} options, "
                f"expected 2..{MAX_OPTIONS}"
            )
        if not 0 <= self.gold_index < len(self.options):
            raise ValueError(
                f"question {self.question_id!r}: gold_index {self.gold_index} out of range"
            )


class DatasetSource(BaseModel):
    """Where a bank's questions came from, recorded in the bank metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo_id: str
    config: str
    split: str
    filename: str
    revision: str
    """Full commit SHA of the dataset repo."""
    sha256: str
    """SHA-256 of the file the bank was built from."""
    row_filter: str | None
    license: str


@dataclass(frozen=True)
class PinnedFile:
    """A dataset file at a fixed Hub commit, with the hash its content must have."""

    repo_id: str
    config: str
    split: str
    filename: str
    revision: str
    license: str
    row_filter: str | None = None
    sha256: str | None = None
    """Expected SHA-256 (known for LFS files)."""
    git_blob_sha1: str | None = None
    """Expected git blob id (for non-LFS files, whose SHA-256 the Hub does not publish)."""

    def __post_init__(self) -> None:
        if (self.sha256 is None) == (self.git_blob_sha1 is None):
            raise ValueError("pin exactly one of sha256 or git_blob_sha1")

    def source(self, sha256: str) -> DatasetSource:
        return DatasetSource(
            repo_id=self.repo_id,
            config=self.config,
            split=self.split,
            filename=self.filename,
            revision=self.revision,
            sha256=sha256,
            row_filter=self.row_filter,
            license=self.license,
        )


def fetch_pinned(pin: PinnedFile) -> tuple[Path, str]:
    """Download (or reuse from the HF cache) a pinned file; return its path and SHA-256.

    Authentication is huggingface_hub's normal chain (``HF_TOKEN`` or the cached token).
    """
    try:
        path = Path(
            hf_hub_download(pin.repo_id, pin.filename, repo_type="dataset", revision=pin.revision)
        )
    except GatedRepoError as exc:
        raise DatasetAccessError(
            f"{pin.repo_id} is gated and the current Hugging Face token has no access. Accept "
            f"the dataset terms at https://huggingface.co/datasets/{pin.repo_id} with the "
            "account whose token is configured (HF_TOKEN or `hf auth login`), then retry."
        ) from exc
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    if pin.sha256 is not None and sha256 != pin.sha256:
        raise DatasetIntegrityError(
            f"{pin.repo_id}/{pin.filename}@{pin.revision}: sha256 {sha256} != pinned {pin.sha256}"
        )
    if pin.git_blob_sha1 is not None:
        blob_sha1 = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        if blob_sha1 != pin.git_blob_sha1:
            raise DatasetIntegrityError(
                f"{pin.repo_id}/{pin.filename}@{pin.revision}: git blob {blob_sha1} "
                f"!= pinned {pin.git_blob_sha1}"
            )
    return path, sha256


GPQA_COLUMNS = (
    "Record ID",
    "Question",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
    "Correct Answer",
    "High-level domain",
)


def parse_gpqa_csv(path: Path) -> list[McqRow]:
    """GPQA rows as evaluate-idk builds them (``custom_tasks.py:143-181``).

    Question and options are stripped; the base order is ``[Incorrect 1, Incorrect 2,
    Incorrect 3, Correct]`` with gold 3. The subject is ``High-level domain``.
    """
    rows: list[McqRow] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in GPQA_COLUMNS if c not in (reader.fieldnames or ())]
        if missing:
            raise DatasetFormatError(f"{path}: missing GPQA columns {missing}")
        for number, record in enumerate(reader, start=1):
            values = {c: (record[c] or "").strip() for c in GPQA_COLUMNS}
            empty = [c for c in GPQA_COLUMNS[:-1] if not values[c]]
            if empty:
                raise DatasetFormatError(f"{path}: data row {number}: empty {empty}")
            rows.append(
                McqRow(
                    question_id=values["Record ID"],
                    question=values["Question"],
                    options=(
                        values["Incorrect Answer 1"],
                        values["Incorrect Answer 2"],
                        values["Incorrect Answer 3"],
                        values["Correct Answer"],
                    ),
                    gold_index=3,
                    subject=values["High-level domain"] or None,
                )
            )
    return rows


LEXAM_COLUMNS = ("id", "question", "choices", "gold", "language", "area")


def parse_lexam_parquet(path: Path) -> list[McqRow]:
    """English LEXam rows as evaluate-idk builds them (``custom_tasks.py:98-116``).

    Keeps ``language == "en"`` in file order. The question is stripped; ``choices`` (a
    stringified Python list) is parsed with ``ast.literal_eval`` and used as stored. The
    subject is ``area``.
    """
    names = pq.read_schema(path).names
    missing = [c for c in LEXAM_COLUMNS if c not in names]
    if missing:
        raise DatasetFormatError(f"{path}: missing LEXam columns {missing}")
    rows: list[McqRow] = []
    table = pq.read_table(path, columns=list(LEXAM_COLUMNS))
    for number, record in enumerate(table.to_pylist(), start=1):
        if record["language"] != "en":
            continue
        where = f"{path}: row {number} (id {record['id']!r})"
        raw = record["choices"]
        try:
            choices = raw if isinstance(raw, list) else ast.literal_eval(raw)
        except (ValueError, SyntaxError) as exc:
            raise DatasetFormatError(f"{where}: choices is not a Python list: {exc}") from exc
        if not (isinstance(choices, list) and all(isinstance(c, str) for c in choices)):
            raise DatasetFormatError(f"{where}: choices must be a list of strings")
        try:
            rows.append(
                McqRow(
                    question_id=record["id"],
                    question=record["question"].strip(),
                    options=tuple(choices),
                    gold_index=record["gold"],
                    subject=record["area"],
                )
            )
        except ValueError as exc:
            raise DatasetFormatError(f"{where}: {exc}") from exc
    return rows


@dataclass(frozen=True)
class BankDataset:
    name: str
    pin: PinnedFile
    parse: Callable[[Path], list[McqRow]]


GPQA_LICENSE = (
    "CC-BY-4.0; gated: do not reveal examples in plain text or images online "
    "(https://huggingface.co/datasets/Idavidrein/gpqa)"
)
LEXAM_LICENSE = "CC-BY-4.0 (https://huggingface.co/datasets/LEXam-Benchmark/LEXam)"

BANK_DATASETS: Mapping[str, BankDataset] = {
    "gpqa_diamond": BankDataset(
        name="gpqa_diamond",
        pin=PinnedFile(
            repo_id="Idavidrein/gpqa",
            config="gpqa_diamond",
            split="train",
            filename="gpqa_diamond.csv",
            revision="83022cefff930aea54f654c0b282e74b9eeda5c6",
            license=GPQA_LICENSE,
            git_blob_sha1="7589e3e467d69a1dceb126a60c4108d6d4f1d166",
        ),
        parse=parse_gpqa_csv,
    ),
    "lexam_en": BankDataset(
        name="lexam_en",
        pin=PinnedFile(
            repo_id="LEXam-Benchmark/LEXam",
            config="mcq_4_choices",
            split="test",
            filename="mcq_4_choices/test-00000-of-00001.parquet",
            revision="31e57ae395f92ed7284a4c28d278c55da892b898",
            license=LEXAM_LICENSE,
            row_filter="language == 'en'",
            sha256="f4c10c4271ca00fc74ff28c88fc6939479635d9ebe7b67ff8d8a28ca46d957e9",
        ),
        parse=parse_lexam_parquet,
    ),
}


# --- bank items ------------------------------------------------------------------------------


def _choice(request: SystemOneRequest) -> ChoiceQuestion:
    if set(request.questions) != {QUESTION_KEY}:
        raise ValueError(f"request must hold exactly one question keyed {QUESTION_KEY!r}")
    question = request.questions[QUESTION_KEY]
    if not isinstance(question, ChoiceQuestion):
        raise ValueError("bank question must be a Choice question")
    return question


class BankItem(BaseModel):
    """One question in one option order, ready to send to Jev or the emulator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str
    """``f"{dataset}:{question_id}:{permutation_id}"``."""
    dataset: str
    question_id: str
    permutation_id: int = Field(ge=0)
    request: SystemOneRequest
    gold: str
    """Criteria key of the correct option (never the IDK key)."""
    option_order: tuple[int, ...]
    """``option_order[i]`` is the base-order index of the option shown at letter ``i``."""
    options_sha256: str
    """SHA-256 of :func:`render_options` over the request's criteria (IDK line included)."""
    subject: str | None
    idk: bool

    @model_validator(mode="after")
    def _consistent(self) -> BankItem:
        if self.item_id != f"{self.dataset}:{self.question_id}:{self.permutation_id}":
            raise ValueError(f"item_id {self.item_id!r} does not match its fields")
        criteria = _choice(self.request).criteria
        n = len(self.option_order)
        if not 2 <= n <= MAX_OPTIONS:
            raise ValueError(f"{n} options, expected 2..{MAX_OPTIONS}")
        letters = OPTION_LETTERS[:n]
        expected_keys = [*letters, OPTION_LETTERS[n]] if self.idk else list(letters)
        if list(criteria) != expected_keys:
            raise ValueError(f"criteria keys {list(criteria)} != {expected_keys}")
        if sorted(self.option_order) != list(range(n)):
            raise ValueError(f"option_order {self.option_order} is not a permutation")
        if self.idk and criteria[OPTION_LETTERS[n]] != IDK_TEXT:
            raise ValueError(f"last option must be {IDK_TEXT!r}")
        if self.gold not in letters:
            raise ValueError(f"gold {self.gold!r} must be one of {list(letters)}")
        texts: dict[str, str] = {}
        for key, text in criteria.items():
            if not isinstance(text, str):
                raise ValueError(f"option {key} must be plain text")
            texts[key] = text
        if options_sha256(texts) != self.options_sha256:
            raise ValueError("options_sha256 does not match the request's options")
        return self

    @property
    def question(self) -> ChoiceQuestion:
        return _choice(self.request)


def _make_item(
    dataset: str, row: McqRow, order: tuple[int, ...], permutation_id: int, *, idk: bool
) -> BankItem:
    n = len(order)
    criteria = {OPTION_LETTERS[i]: row.options[base] for i, base in enumerate(order)}
    if idk:
        criteria[OPTION_LETTERS[n]] = IDK_TEXT
    request = SystemOneRequest(
        state=row.question,
        model=DEFAULT_MODEL,
        questions={
            QUESTION_KEY: ChoiceQuestion(
                instructions=instructions_for(n, idk=idk), criteria=criteria
            )
        },
    )
    return BankItem(
        item_id=f"{dataset}:{row.question_id}:{permutation_id}",
        dataset=dataset,
        question_id=row.question_id,
        permutation_id=permutation_id,
        request=request,
        gold=OPTION_LETTERS[order.index(row.gold_index)],
        option_order=order,
        options_sha256=options_sha256(criteria),
        subject=row.subject,
        idk=idk,
    )


def _question_rng(seed: int, question_id: str) -> random.Random:
    """Private RNG seeded from a stable hash of ``(seed, question_id)`` (not ``hash()``)."""
    digest = hashlib.sha256(f"{seed}\x1f{question_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


# --- the bank --------------------------------------------------------------------------------


class BankMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    builder_version: str
    dataset: str
    source: DatasetSource
    order: BankOrder
    seed: int
    """Seeds ``per_question`` order; recorded but unused for ``evaluate_idk`` (always 42)."""
    idk: bool
    n_permutations: int = Field(ge=1)
    limit: int | None
    n_questions: int
    n_items: int
    items_sha256: str
    """SHA-256 over the serialized item lines; ``from_jsonl`` rejects a bank that fails it."""


def _item_lines(items: Sequence[BankItem]) -> list[str]:
    return [item.model_dump_json() for item in items]


def _items_sha256(lines: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class QuestionBank(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: BankMetadata
    items: tuple[BankItem, ...]

    @model_validator(mode="after")
    def _consistent(self) -> QuestionBank:
        meta = self.metadata
        if meta.n_items != len(self.items):
            raise ValueError(f"metadata n_items {meta.n_items} != {len(self.items)} items")
        ids = [item.item_id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate item_id in bank")
        if any(item.dataset != meta.dataset for item in self.items):
            raise ValueError(f"every item must belong to dataset {meta.dataset!r}")
        if _items_sha256(_item_lines(self.items)) != meta.items_sha256:
            raise ValueError("items_sha256 does not match the items")
        return self

    def dumps(self) -> str:
        """The ``bank.jsonl`` text: a ``{"metadata": ...}`` line, then one item per line."""
        header = '{"metadata":' + self.metadata.model_dump_json() + "}"
        return "".join(f"{line}\n" for line in [header, *_item_lines(self.items)])

    def to_jsonl(self, path: str | Path) -> None:
        Path(path).write_bytes(self.dumps().encode("utf-8"))

    @classmethod
    def from_jsonl(cls, path: str | Path) -> QuestionBank:
        # Only "\n" ends a line: JSON escapes it inside strings but leaves U+0085 and U+2028
        # raw, which str.splitlines would also split on.
        lines = Path(path).read_text(encoding="utf-8").split("\n")
        if lines[-1] == "":
            lines.pop()
        if not lines:
            raise BankFormatError(f"{path}: empty file")
        try:
            header = BankMetadata.model_validate_json(_unwrap_header(lines[0]))
        except (ValidationError, ValueError) as exc:
            raise BankFormatError(f"{path}:1: bad metadata line: {exc}") from exc
        items: list[BankItem] = []
        for number, line in enumerate(lines[1:], start=2):
            try:
                items.append(BankItem.model_validate_json(line))
            except ValidationError as exc:
                raise BankFormatError(f"{path}:{number}: bad item: {exc}") from exc
        try:
            return cls(metadata=header, items=tuple(items))
        except ValidationError as exc:
            raise BankFormatError(f"{path}: {exc}") from exc


def _unwrap_header(line: str) -> str:
    prefix = '{"metadata":'
    if not (line.startswith(prefix) and line.endswith("}")):
        raise ValueError('first line must be {"metadata": {...}}')
    return line[len(prefix) : -1]


def _check_options(order: str, n_permutations: int, limit: int | None) -> None:
    if order not in BANK_ORDERS:
        raise ValueError(f"order must be one of {BANK_ORDERS}, got {order!r}")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1 or None, got {limit}")


def bank_from_rows(
    dataset: str,
    rows: Sequence[McqRow],
    *,
    source: DatasetSource,
    seed: int,
    idk: bool,
    n_permutations: int = 1,
    order: BankOrder = "per_question",
    limit: int | None = None,
) -> QuestionBank:
    """Build a bank from parsed rows in dataset order (the first ``limit`` questions).

    ``evaluate_idk`` order consumes one shuffle per row from a single ``Random(42)`` stream, so
    it must be given every row of the filtered split from the first one; a limited bank is a
    prefix of the full bank.
    """
    _check_options(order, n_permutations, limit)
    seen: set[str] = set()
    stream = random.Random(EVALUATE_IDK_SEED) if order == "evaluate_idk" else None
    items: list[BankItem] = []
    kept = rows if limit is None else rows[:limit]
    for row in kept:
        if row.question_id in seen:
            raise DatasetFormatError(f"{dataset}: duplicate question_id {row.question_id!r}")
        seen.add(row.question_id)
        n = len(row.options)
        if n_permutations > n:
            raise ValueError(
                f"n_permutations {n_permutations} exceeds the {n} options of "
                f"question {row.question_id!r}"
            )
        base = list(range(n))
        rng = stream if stream is not None else _question_rng(seed, row.question_id)
        # Shuffling indices is the same permutation as shuffling the texts: Fisher-Yates
        # swaps depend only on the length.
        rng.shuffle(base)
        for shift in range(n_permutations):
            shifted = tuple(base[shift:] + base[:shift])
            items.append(_make_item(dataset, row, shifted, shift, idk=idk))
    lines = _item_lines(items)
    metadata = BankMetadata(
        builder_version=BUILDER_VERSION,
        dataset=dataset,
        source=source,
        order=order,
        seed=seed,
        idk=idk,
        n_permutations=n_permutations,
        limit=limit,
        n_questions=len(kept),
        n_items=len(items),
        items_sha256=_items_sha256(lines),
    )
    return QuestionBank(metadata=metadata, items=tuple(items))


def build_question_bank(
    dataset: str,
    *,
    seed: int,
    idk: bool,
    n_permutations: int = 1,
    order: BankOrder = "per_question",
    limit: int | None = None,
) -> QuestionBank:
    """Download the pinned dataset file and build its bank (see the module docstring).

    ``dataset`` is a key of :data:`BANK_DATASETS`: ``"gpqa_diamond"`` (198 questions, gated)
    or ``"lexam_en"`` (619).
    """
    spec = BANK_DATASETS.get(dataset)
    if spec is None:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(BANK_DATASETS)}")
    _check_options(order, n_permutations, limit)
    path, sha256 = fetch_pinned(spec.pin)
    return bank_from_rows(
        dataset,
        spec.parse(path),
        source=spec.pin.source(sha256),
        seed=seed,
        idk=idk,
        n_permutations=n_permutations,
        order=order,
        limit=limit,
    )
