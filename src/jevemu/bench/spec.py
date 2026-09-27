"""Benchmark contracts: a spec names a loader that yields labelled Jev questions.

Every benchmark item is a (state, question, gold) triple in Jev's own types, so benchmarks and
the paired Jev comparison share one scoring path. Gold depends on the question type: a
criteria key for Choice, a level index for Score, a boolean for Noul.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from jevemu.eval.bank import (
    BANK_ORDERS,
    QUESTION_KEY,
    QuestionBank,
    build_question_bank,
)
from jevemu.types import ChoiceQuestion, JSONish, Question, ScoreQuestion

QuestionType = Literal["choice", "score", "noul"]
QUESTION_TYPES: tuple[QuestionType, ...] = ("choice", "score", "noul")

Gold = StrictStr | StrictBool | StrictInt
"""Choice: criteria key (str). Score: level index (int). Noul: bool."""


class BenchItem(BaseModel):
    """One labelled question. ``item_id`` is unique within its benchmark."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str = Field(min_length=1)
    state: JSONish
    question: Question
    gold: Gold
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _gold_fits_question(self) -> BenchItem:
        question, gold = self.question, self.gold
        if isinstance(question, ChoiceQuestion):
            if not isinstance(gold, str) or gold not in question.criteria:
                raise ValueError(
                    f"choice gold {gold!r} is not a criteria key {list(question.criteria)}"
                )
        elif isinstance(question, ScoreQuestion):
            k = len(question.criteria)
            if isinstance(gold, bool) or not isinstance(gold, int) or not 0 <= gold < k:
                raise ValueError(f"score gold {gold!r} must be a level index in 0..{k - 1}")
        elif not isinstance(gold, bool):
            raise ValueError(f"noul gold {gold!r} must be true or false")
        return self


@dataclass(frozen=True)
class BenchConfig:
    """What a loader is asked for.

    ``split`` is a role (a key of :attr:`BenchmarkSpec.splits`, e.g. ``"test"``); ``limit``
    caps the number of questions (``None`` = all); ``params`` holds benchmark-specific options.
    """

    seed: int = 0
    split: str = "test"
    limit: int | None = None
    params: Mapping[str, Any] = field(default_factory=dict)


Loader = Callable[[BenchConfig], Iterable[BenchItem]]


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    question_type: QuestionType
    loader: Loader
    splits: Mapping[str, str]
    """Role (``"test"``, ``"calib"``) -> dataset split name."""
    license: str
    tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("benchmark name must be non-empty")
        if self.question_type not in QUESTION_TYPES:
            raise ValueError(f"question_type must be one of {QUESTION_TYPES}")
        if not self.splits:
            raise ValueError(f"benchmark {self.name!r} declares no splits")
        if not self.license:
            raise ValueError(f"benchmark {self.name!r} must record its license")


# --- question banks as benchmarks ------------------------------------------------------------


def bench_items_from_bank(bank: QuestionBank) -> list[BenchItem]:
    """The single mapping from bank items to benchmark items (same request, same gold)."""
    source = bank.metadata.source
    return [
        BenchItem(
            item_id=item.item_id,
            state=item.request.state,
            question=item.request.questions[QUESTION_KEY],
            gold=item.gold,
            metadata={
                "dataset": item.dataset,
                "question_id": item.question_id,
                "permutation_id": item.permutation_id,
                "option_order": list(item.option_order),
                "options_sha256": item.options_sha256,
                "subject": item.subject,
                "idk": item.idk,
                "revision": source.revision,
                "source_sha256": source.sha256,
            },
        )
        for item in bank.items
    ]


BANK_PARAMS = frozenset({"order", "idk", "n_permutations"})


def load_bank_benchmark(dataset: str, cfg: BenchConfig) -> list[BenchItem]:
    """Build ``dataset``'s question bank from ``cfg`` and map it to benchmark items.

    These are evaluate-idk parity benchmarks, so ``cfg.params["order"]`` defaults to
    ``"evaluate_idk"`` (where ``cfg.seed`` has no effect); ``idk`` defaults to true and
    ``n_permutations`` to 1. ``cfg.limit`` caps questions, not permutations.
    """
    unknown = set(cfg.params) - BANK_PARAMS
    if unknown:
        raise ValueError(
            f"unknown params {sorted(unknown)} for {dataset}; use {sorted(BANK_PARAMS)}"
        )
    order = cfg.params.get("order", "evaluate_idk")
    if order not in BANK_ORDERS:
        raise ValueError(f"order must be one of {BANK_ORDERS}, got {order!r}")
    idk = cfg.params.get("idk", True)
    n_permutations = cfg.params.get("n_permutations", 1)
    if not isinstance(idk, bool):
        raise ValueError(f"idk must be a boolean, got {idk!r}")
    if isinstance(n_permutations, bool) or not isinstance(n_permutations, int):
        raise ValueError(f"n_permutations must be an integer, got {n_permutations!r}")
    bank = build_question_bank(
        dataset,
        seed=cfg.seed,
        idk=idk,
        n_permutations=n_permutations,
        order=order,
        limit=cfg.limit,
    )
    return bench_items_from_bank(bank)
