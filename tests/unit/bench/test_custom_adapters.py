from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from jevemu.bench import BenchConfig, load_benchmark
from jevemu.bench.datasets import (
    ColumnMapping,
    CustomDataError,
    custom_benchmark,
    custom_csv,
    custom_jsonl,
)
from jevemu.types import ChoiceQuestion, NoulQuestion, ScoreQuestion


def _jsonl(path: Path, rows: list[Any]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


CHOICE_FIXED = ColumnMapping(
    question_type="choice",
    state="text",
    gold="label",
    instructions="Route the ticket.",
    criteria=["billing", "tech", "other"],
    metadata=("lang",),
)
SCORE_FIXED = ColumnMapping(
    question_type="score",
    state="review",
    gold="stars",
    id="rid",
    instructions="How positive is the review?",
    criteria=["negative", "neutral", "positive"],
)
NOUL_ROWWISE = ColumnMapping(
    question_type="noul",
    state="text",
    gold="spam",
    instructions_column="prompt",
)
CHOICE_ROWWISE = ColumnMapping(
    question_type="choice",
    state="stem",
    gold="answer",
    id="qid",
    instructions="Pick one.",
    criteria_column="options",
)


def test_all_three_question_types_load_from_jsonl(tmp_path: Path) -> None:
    choice = custom_jsonl(
        _jsonl(
            tmp_path / "c.jsonl",
            [
                {"text": "card charged twice", "label": "billing", "lang": "en"},
                {"text": {"subject": "crash", "body": "app dies"}, "label": "tech", "lang": "de"},
            ],
        ),
        CHOICE_FIXED,
        name="tickets",
    )
    assert [(i.item_id, i.gold, i.metadata) for i in choice] == [
        ("tickets:1", "billing", {"lang": "en"}),
        ("tickets:2", "tech", {"lang": "de"}),
    ]
    assert choice[1].state == {"subject": "crash", "body": "app dies"}
    assert choice[0].question == ChoiceQuestion(
        instructions="Route the ticket.", criteria={"billing": None, "tech": None, "other": None}
    )

    score = custom_jsonl(
        _jsonl(tmp_path / "s.jsonl", [{"rid": 17, "review": "meh", "stars": 1}]),
        SCORE_FIXED,
        name="reviews",
    )
    assert [(i.item_id, i.gold) for i in score] == [("reviews:17", 1)]
    assert isinstance(score[0].question, ScoreQuestion)

    noul = custom_jsonl(
        _jsonl(
            tmp_path / "n.jsonl",
            [{"text": "WIN $$$", "spam": True, "prompt": "Spam?"}, {"text": "hi", "spam": False}],
        ),
        NOUL_ROWWISE.model_copy(update={"instructions_column": None}),
        name="mail",
    )
    assert [i.gold for i in noul] == [True, False]
    assert all(isinstance(i.question, NoulQuestion) for i in noul)


def test_all_three_question_types_load_from_csv(tmp_path: Path) -> None:
    choice = custom_csv(
        _csv(
            tmp_path / "c.csv",
            ["qid", "stem", "options", "answer"],
            [
                ["q1", "2+2?", json.dumps({"A": "3", "B": "4"}), "B"],
                ["q2", "Sky?", '["blue", "red"]', "blue"],
            ],
        ),
        CHOICE_ROWWISE,
        name="quiz",
    )
    assert [(i.item_id, i.gold) for i in choice] == [("quiz:q1", "B"), ("quiz:q2", "blue")]
    assert choice[0].question == ChoiceQuestion(
        instructions="Pick one.", criteria={"A": "3", "B": "4"}
    )
    assert choice[1].question == ChoiceQuestion(
        instructions="Pick one.", criteria={"blue": None, "red": None}
    )

    score = custom_csv(
        _csv(
            tmp_path / "s.csv",
            ["rid", "review", "stars"],
            [["r1", "great", " 2 "], ["r2", "awful", "0"]],
        ),
        SCORE_FIXED,
        name="reviews",
    )
    assert [(i.item_id, i.gold) for i in score] == [("reviews:r1", 2), ("reviews:r2", 0)]

    noul = custom_csv(
        _csv(
            tmp_path / "n.csv",
            ["text", "spam", "prompt"],
            [["a", "TRUE", "Spam?"], ["b", "no", "Spam?"], ["c", "1", "Is it spam?"]],
        ),
        NOUL_ROWWISE,
        name="mail",
    )
    assert [(i.gold, i.question.instructions) for i in noul] == [
        (True, "Spam?"),
        (False, "Spam?"),
        (True, "Is it spam?"),
    ]


@pytest.mark.parametrize(
    ("rows", "mapping", "message"),
    [
        ([{"text": "x"}], CHOICE_FIXED, r":1: missing column 'label'"),
        (
            [{"text": "x", "label": "sales", "lang": "en"}],
            CHOICE_FIXED,
            "'sales' is not a criteria key",
        ),
        ([{"rid": 1, "review": "x", "stars": 3}], SCORE_FIXED, r"level index in 0\.\.2"),
        ([{"rid": 1, "review": "x", "stars": 1.5}], SCORE_FIXED, "gold 1.5 is not a level index"),
        ([{"rid": 1, "review": "x", "stars": True}], SCORE_FIXED, "gold True is not a level index"),
        ([{"text": "x", "spam": "maybe", "prompt": "?"}], NOUL_ROWWISE, "'maybe' is not a boolean"),
        ([{"text": None, "spam": True, "prompt": "?"}], NOUL_ROWWISE, "state must be text or JSON"),
        (
            [{"stem": "s", "answer": "A", "qid": "q", "options": ["A"]}],
            CHOICE_ROWWISE,
            "invalid choice question: criteria",
        ),
        (
            [{"stem": "s", "answer": "A", "qid": "q", "options": ["A", "A"]}],
            CHOICE_ROWWISE,
            "duplicate option",
        ),
        (
            [
                {"rid": "a", "review": "x", "stars": 0},
                {"rid": "a", "review": "y", "stars": 1},
            ],
            SCORE_FIXED,
            r":2: duplicate id 'reviews:a' \(first at .*:1\)",
        ),
        ([["not", "an", "object"]], CHOICE_FIXED, r":1: expected a JSON object"),
    ],
)
def test_malformed_jsonl_rows_raise_located_errors(
    tmp_path: Path, rows: list[Any], mapping: ColumnMapping, message: str
) -> None:
    path = _jsonl(tmp_path / "bad.jsonl", rows)
    with pytest.raises(CustomDataError, match=message) as info:
        custom_jsonl(path, mapping, name="reviews")
    assert str(path) in str(info.value)


def test_jsonl_skips_blank_lines_but_reports_invalid_json_with_its_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('\n{"text": "a", "spam": true}\n{"text": "b", "spam": }\n', encoding="utf-8")
    mapping = NOUL_ROWWISE.model_copy(update={"instructions_column": None})
    with pytest.raises(CustomDataError, match=r"bad\.jsonl:3: invalid JSON"):
        custom_jsonl(path, mapping, name="mail")


@pytest.mark.parametrize(
    ("header", "rows", "message"),
    [
        (["qid", "stem", "answer"], [], r"missing columns \['options'\]"),
        (
            ["qid", "stem", "options", "answer"],
            [["q", "s", "{not json", "A"]],
            r":2: criteria column 'options'",
        ),
        (["qid", "stem", "options", "answer"], [["q", "s", '["A", "B"]']], r":2: fewer fields"),
        (
            ["qid", "stem", "options", "answer"],
            [["q", "s", '["A", "B"]', "A", "extra"]],
            r":2: more fields",
        ),
        (["qid", "stem", "options", "answer"], [["", "s", '["A", "B"]', "A"]], r"id '' must be"),
        (["qid", "stem", "options", "answer"], [], "no rows"),
    ],
)
def test_malformed_csv_raises_located_errors(
    tmp_path: Path, header: list[str], rows: list[list[str]], message: str
) -> None:
    path = _csv(tmp_path / "bad.csv", header, rows)
    with pytest.raises(CustomDataError, match=message):
        custom_csv(path, CHOICE_ROWWISE, name="quiz")


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"question_type": "choice"}, "need criteria or criteria_column"),
        ({"question_type": "score", "criteria": ["only one"]}, "invalid score question"),
        (
            {"question_type": "noul", "instructions": "x", "instructions_column": "y"},
            "not both",
        ),
        (
            {"question_type": "choice", "criteria": ["a", "b"], "criteria_column": "c"},
            "not both",
        ),
    ],
)
def test_incoherent_mappings_are_rejected(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ColumnMapping(state="s", gold="g", **fields)


def test_custom_benchmark_infers_format_and_honours_limit(tmp_path: Path) -> None:
    path = _csv(
        tmp_path / "reviews.csv",
        ["rid", "review", "stars"],
        [["r1", "a", "0"], ["r2", "b", "1"], ["r3", "c", "2"]],
    )
    spec = custom_benchmark("reviews", path, SCORE_FIXED, license="CC0-1.0")
    assert spec.question_type == "score"
    items = load_benchmark(spec, BenchConfig(limit=2))
    assert [i.item_id for i in items] == ["reviews:r1", "reviews:r2"]
    with pytest.raises(ValueError, match="takes no params"):
        load_benchmark(spec, BenchConfig(params={"order": "x"}))
    with pytest.raises(ValueError, match="cannot infer the format"):
        custom_benchmark("reviews", tmp_path / "reviews.txt", SCORE_FIXED, license="CC0-1.0")
