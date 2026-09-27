from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from jevemu.bench import BenchConfig, load_benchmark
from jevemu.bench.datasets.arc import parse_arc
from jevemu.bench.datasets.boolq import parse_boolq
from jevemu.bench.datasets.clinc150 import parse_clinc150
from jevemu.bench.datasets.mmlu_pro import parse_mmlu_pro
from jevemu.bench.datasets.sst5 import parse_sst5
from jevemu.bench.datasets.yelp import sample_rows
from jevemu.eval.bank import DatasetFormatError, DatasetSource
from jevemu.types import ChoiceQuestion

SOURCE = DatasetSource(
    repo_id="org/data",
    config="default",
    split="test",
    filename="test.parquet",
    revision="0" * 40,
    sha256="1" * 64,
    row_filter=None,
    license="CC0",
)


def _parquet(path: Path, rows: list[dict[str, Any]], class_names: dict[str, list[str]]) -> Path:
    features = {
        column: {"names": names, "_type": "ClassLabel"} for column, names in class_names.items()
    }
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {"huggingface": json.dumps({"info": {"features": features}})}
    )
    pq.write_table(table, path)
    return path


def _arc_row(qid: str, labels: list[str], answer: str) -> dict[str, Any]:
    texts = [f"option {label}" for label in labels]
    return {
        "id": qid,
        "question": "Why?",
        "choices": {"text": texts, "label": labels},
        "answerKey": answer,
    }


def test_arc_digit_labels_become_letters_and_keep_the_gold_position(tmp_path: Path) -> None:
    path = _parquet(
        tmp_path / "arc.parquet",
        [_arc_row("q1", ["A", "B", "C", "D"], "B"), _arc_row("q2", ["1", "2", "3"], "3")],
        {},
    )
    first, second = parse_arc(path, SOURCE)
    assert isinstance(second.question, ChoiceQuestion)
    assert list(second.question.criteria) == ["A", "B", "C"]
    assert second.question.criteria["C"] == "option 3"
    assert (first.gold, second.gold) == ("B", "C")
    assert second.item_id == "arc_challenge:q2"


def test_mmlu_pro_keys_as_many_letters_as_the_question_has_options(tmp_path: Path) -> None:
    row = {
        "question_id": 7,
        "question": "2 + 2 = ?",
        "options": ["3", "4", "5", "22"],
        "answer": "B",
        "answer_index": 1,
        "category": "math",
        "src": "ori_mmlu-high_school_mathematics",
    }
    (item,) = parse_mmlu_pro(_parquet(tmp_path / "m.parquet", [row], {}), SOURCE)
    assert isinstance(item.question, ChoiceQuestion)
    assert item.question.criteria == {"A": "3", "B": "4", "C": "5", "D": "22"}
    assert (item.gold, item.metadata["stratum"], item.metadata["question_id"]) == ("B", "math", "7")


def test_clinc_drops_out_of_scope_rows_and_the_oos_option(tmp_path: Path) -> None:
    names = ["translate", "oos", "timer"]
    rows = [
        {"text": "say hi in french", "intent": 0},
        {"text": "who won the 1966 world cup", "intent": 1},
        {"text": "set a timer", "intent": 2},
    ]
    items = parse_clinc150(_parquet(tmp_path / "c.parquet", rows, {"intent": names}), SOURCE)
    assert [(item.item_id, item.gold) for item in items] == [
        ("clinc150:0", "translate"),
        ("clinc150:2", "timer"),
    ]
    assert isinstance(items[0].question, ChoiceQuestion)
    assert list(items[0].question.criteria) == ["translate", "timer"]


def test_yelp_sample_keeps_at_most_per_level_rows_of_each_label_in_file_order() -> None:
    labels = [i % 5 for i in range(100)] + [4] * 3
    kept = sample_rows(labels, per_level=10, seed=0)
    assert kept == sorted(kept)
    assert [sum(labels[row] == level for row in kept) for level in range(5)] == [10] * 5
    assert sample_rows(labels, per_level=10, seed=0) == kept
    assert sample_rows(labels, per_level=10, seed=1) != kept
    assert sample_rows([0, 0, 1], per_level=10, seed=0) == [0, 1, 2]


def _sst_file(tmp_path: Path, record: dict[str, Any]) -> Path:
    path = tmp_path / "sst.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("parse", "make", "message"),
    [
        (
            parse_sst5,
            lambda tmp: _sst_file(tmp, {"text": "fine .", "label": 3, "label_text": "neutral"}),
            "label_text",
        ),
        (
            parse_sst5,
            lambda tmp: _sst_file(tmp, {"text": "fine .", "label": 5, "label_text": "x"}),
            r"not in 0\.\.4",
        ),
        (
            parse_arc,
            lambda tmp: _parquet(tmp / "a.parquet", [_arc_row("q", ["A", "B"], "E")], {}),
            "answerKey",
        ),
        (
            parse_boolq,
            lambda tmp: _parquet(
                tmp / "b.parquet", [{"question": "q", "answer": "yes", "passage": "p"}], {}
            ),
            "not a boolean",
        ),
    ],
)
def test_malformed_rows_are_rejected_with_their_location(
    tmp_path: Path, parse: Any, make: Any, message: str
) -> None:
    with pytest.raises(DatasetFormatError, match=message) as info:
        parse(make(tmp_path), SOURCE)
    assert "row 0" in str(info.value)


def test_builtin_adapters_reject_params_meant_for_the_question_banks() -> None:
    with pytest.raises(ValueError, match="takes no params"):
        load_benchmark("boolq", BenchConfig(params={"order": "evaluate_idk"}))
