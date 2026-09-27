from __future__ import annotations

from collections.abc import Iterable

import pytest
from pydantic import ValidationError

from jevemu.bench import (
    BenchConfig,
    BenchItem,
    BenchmarkRegistry,
    bench_items_from_bank,
    list_benchmarks,
    load_bank_benchmark,
    load_benchmark,
    register_benchmark,
)
from jevemu.eval.bank import DatasetSource, McqRow, bank_from_rows
from jevemu.types import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

CHOICE = ChoiceQuestion(instructions="Route it.", criteria={"billing": None, "tech": "Bugs"})
SCORE = ScoreQuestion(instructions="Rate it.", criteria=["bad", "ok", "good"])
NOUL = NoulQuestion(instructions="Is it spam?")


@pytest.mark.parametrize(
    ("question", "gold"),
    [(CHOICE, "tech"), (SCORE, 0), (SCORE, 2), (NOUL, True), (NOUL, False)],
)
def test_bench_item_round_trips_gold_with_its_type(question: Question, gold: object) -> None:
    item = BenchItem(item_id="x:1", state={"text": "hi"}, question=question, gold=gold)
    again = BenchItem.model_validate_json(item.model_dump_json())
    assert again == item
    assert type(again.gold) is type(gold)


@pytest.mark.parametrize(
    ("question", "gold", "message"),
    [
        (CHOICE, "refund", "not a criteria key"),
        (CHOICE, 1, "not a criteria key"),
        (SCORE, 3, r"level index in 0\.\.2"),
        (SCORE, -1, r"level index in 0\.\.2"),
        (SCORE, True, "level index"),
        (SCORE, "1", "level index"),
        (NOUL, 1, "true or false"),
        (NOUL, "true", "true or false"),
    ],
)
def test_bench_item_rejects_gold_that_does_not_fit_the_question(
    question: Question, gold: object, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        BenchItem(item_id="x:1", state="s", question=question, gold=gold)


# --- registry --------------------------------------------------------------------------------


def _items(*pairs: tuple[str, Question]) -> list[BenchItem]:
    return [
        BenchItem(
            item_id=item_id,
            state="s",
            question=q,
            gold="tech" if q.type == "choice" else (True if q.type == "noul" else 0),
        )
        for item_id, q in pairs
    ]


def test_registry_rejects_duplicates_and_names_known_benchmarks() -> None:
    registry = BenchmarkRegistry()

    @register_benchmark(
        "toy", question_type="choice", license="CC0-1.0", splits={"test": "t"}, registry=registry
    )
    def toy(cfg: BenchConfig) -> Iterable[BenchItem]:
        return _items(("toy:1", CHOICE))

    with pytest.raises(ValueError, match="'toy' is already registered"):
        register_benchmark(
            "toy", question_type="noul", license="MIT", splits={"test": "t"}, registry=registry
        )(toy)
    with pytest.raises(KeyError, match=r"unknown benchmark 'nope'; registered: \['toy'\]"):
        registry.get("nope")
    assert registry.get("toy").loader is toy


def test_builtin_benchmarks_are_listed_with_licenses() -> None:
    specs = {spec.name: spec for spec in list_benchmarks()}
    assert {"gpqa_diamond_idk", "lexam_en_idk"} <= set(specs)
    assert "do not reveal examples" in specs["gpqa_diamond_idk"].license
    assert specs["lexam_en_idk"].license.startswith("CC-BY-4.0")


@pytest.mark.parametrize(
    ("items", "cfg", "message"),
    [
        (_items(("t:1", CHOICE)), BenchConfig(split="calib"), "no split 'calib'"),
        (_items(("t:1", CHOICE), ("t:2", NOUL)), BenchConfig(), "holds a 'noul' question"),
        (_items(("t:1", CHOICE), ("t:1", CHOICE)), BenchConfig(), "duplicate item_id 't:1'"),
    ],
)
def test_load_benchmark_enforces_the_spec(
    items: list[BenchItem], cfg: BenchConfig, message: str
) -> None:
    registry = BenchmarkRegistry()
    register_benchmark(
        "t", question_type="choice", license="CC0-1.0", splits={"test": "t"}, registry=registry
    )(lambda cfg: items)
    with pytest.raises(ValueError, match=message):
        load_benchmark(registry.get("t"), cfg)


# --- question banks as benchmarks ------------------------------------------------------------

SOURCE = DatasetSource(
    repo_id="example/mcq",
    config="default",
    split="test",
    filename="mcq.parquet",
    revision="0" * 40,
    sha256="1" * 64,
    row_filter=None,
    license="CC0-1.0",
)


def test_bank_items_map_to_bench_items_with_the_same_request_and_gold() -> None:
    rows = [McqRow(f"q{i}", f"Q{i}?", ("w", "x", "y", "z"), i % 4, subject="s") for i in range(3)]
    bank = bank_from_rows("toy", rows, source=SOURCE, seed=1, idk=True, n_permutations=2)
    items = bench_items_from_bank(bank)
    assert [i.item_id for i in items] == [b.item_id for b in bank.items]
    for item, bank_item in zip(items, bank.items, strict=True):
        assert item.state == bank_item.request.state
        assert item.question == bank_item.request.questions["answer"]
        assert item.gold == bank_item.gold
        assert item.metadata["question_id"] == bank_item.question_id
        assert item.metadata["option_order"] == list(bank_item.option_order)
        assert item.metadata["revision"] == "0" * 40


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"shuffle": True}, r"unknown params \['shuffle'\]"),
        ({"order": "random"}, "order must be one of"),
        ({"idk": "yes"}, "idk must be a boolean"),
        ({"n_permutations": 2.0}, "n_permutations must be an integer"),
    ],
)
def test_bank_benchmark_params_are_validated(params: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_bank_benchmark("lexam_en", BenchConfig(params=params))
