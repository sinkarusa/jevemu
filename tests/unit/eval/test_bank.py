from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub.errors import GatedRepoError
from hypothesis import given, settings
from hypothesis import strategies as st

from jevemu.eval import bank as bank_module
from jevemu.eval.bank import (
    BankFormatError,
    DatasetAccessError,
    DatasetFormatError,
    DatasetIntegrityError,
    DatasetSource,
    McqRow,
    PinnedFile,
    QuestionBank,
    bank_from_rows,
    build_question_bank,
    fetch_pinned,
    parse_gpqa_csv,
    parse_lexam_parquet,
)
from jevemu.types import ChoiceQuestion

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


def _rows(n: int = 12) -> list[McqRow]:
    return [
        McqRow(
            question_id=f"q{i:03d}",
            question=f"Question {i}?",
            options=tuple(f"q{i} option {j}" for j in range(4)),
            gold_index=i % 4,
            subject="even" if i % 2 == 0 else "odd",
        )
        for i in range(n)
    ]


def _bank(rows: list[McqRow], **kwargs: object) -> QuestionBank:
    options: dict[str, object] = {"seed": 0, "idk": True} | kwargs
    return bank_from_rows("toy", rows, source=SOURCE, **options)


def _criteria(item: bank_module.BankItem) -> dict[str, str]:
    question = item.request.questions["answer"]
    assert isinstance(question, ChoiceQuestion)
    return {key: str(text) for key, text in question.criteria.items()}


def _orders(bank: QuestionBank) -> dict[str, tuple[int, ...]]:
    return {item.question_id: item.option_order for item in bank.items}


# --- determinism -----------------------------------------------------------------------------


@pytest.mark.parametrize("order", ["per_question", "evaluate_idk"])
@pytest.mark.parametrize("n_permutations", [1, 4])
def test_same_seed_gives_byte_identical_bank_file(
    tmp_path: Path, order: str, n_permutations: int
) -> None:
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _bank(_rows(), seed=7, order=order, n_permutations=n_permutations).to_jsonl(first)
    _bank(_rows(), seed=7, order=order, n_permutations=n_permutations).to_jsonl(second)
    assert first.read_bytes() == second.read_bytes()

    reloaded = QuestionBank.from_jsonl(first)
    reloaded.to_jsonl(second)
    assert second.read_bytes() == first.read_bytes()
    assert reloaded == _bank(_rows(), seed=7, order=order, n_permutations=n_permutations)


def test_seed_changes_per_question_order_only() -> None:
    rows = _rows(30)
    assert _orders(_bank(rows, seed=0)) != _orders(_bank(rows, seed=1))
    assert _orders(_bank(rows, seed=0, order="evaluate_idk")) == _orders(
        _bank(rows, seed=1, order="evaluate_idk")
    )


def test_per_question_order_does_not_depend_on_row_position() -> None:
    rows = _rows(20)
    full = _orders(_bank(rows, seed=3))
    reordered = _orders(_bank([*reversed(rows[5:]), rows[0]], seed=3))
    assert all(full[qid] == order for qid, order in reordered.items())


def test_evaluate_idk_order_is_one_random42_stream_over_rows() -> None:
    # evaluate-idk's shuffle_choices: shuffle a copy of the option texts on a Random(42) stream
    # (one shuffle per row, in row order) and find the correct text with .index().
    rows = _rows(15)
    stream = random.Random(42)
    expected = []
    for row in rows:
        shuffled = list(row.options)
        stream.shuffle(shuffled)
        expected.append((shuffled, "ABCD"[shuffled.index(row.options[row.gold_index])]))

    bank = _bank(rows, order="evaluate_idk")
    got = [(list(_criteria(item).values())[:4], item.gold) for item in bank.items]
    assert got == expected

    # The stream is positional: dropping the first row changes the next rows' orders.
    assert _orders(_bank(rows[1:], order="evaluate_idk")) != {
        qid: order for qid, order in _orders(bank).items() if qid != rows[0].question_id
    }


@pytest.mark.parametrize("order", ["per_question", "evaluate_idk"])
def test_limit_keeps_a_prefix_of_the_full_bank(order: str) -> None:
    full = _bank(_rows(10), order=order, n_permutations=2)
    limited = _bank(_rows(10), order=order, n_permutations=2, limit=4)
    assert limited.items == full.items[:8]
    assert (limited.metadata.n_questions, limited.metadata.n_items) == (4, 8)


# --- mapping ---------------------------------------------------------------------------------

row_strategy = st.builds(
    lambda n, gold, qid, texts: McqRow(
        question_id=qid, question="Q?", options=tuple(texts[:n]), gold_index=gold % n
    ),
    st.integers(2, 6),
    st.integers(0, 5),
    st.text(min_size=1, max_size=8),
    st.lists(st.text(max_size=5), min_size=6, max_size=6),
)


@settings(max_examples=150, deadline=None)
@given(
    row=row_strategy,
    seed=st.integers(-(2**40), 2**40),
    order=st.sampled_from(["per_question", "evaluate_idk"]),
    idk=st.booleans(),
    data=st.data(),
)
def test_gold_follows_the_correct_option_through_every_shuffle(
    row: McqRow, seed: int, order: str, idk: bool, data: st.DataObject
) -> None:
    n = len(row.options)
    n_permutations = data.draw(st.integers(1, n))
    bank = _bank([row], seed=seed, order=order, idk=idk, n_permutations=n_permutations)
    assert len(bank.items) == n_permutations
    for item in bank.items:
        criteria = _criteria(item)
        letters = "ABCDEFG"[:n]
        assert list(criteria) == ([*letters, "ABCDEFG"[n]] if idk else list(letters))
        assert sorted(item.option_order) == list(range(n))
        assert [criteria[k] for k in letters] == [row.options[i] for i in item.option_order]
        assert item.option_order[letters.index(item.gold)] == row.gold_index
        if idk:
            assert list(criteria.items())[-1] == ("ABCDEFG"[n], "I don't know")
            assert item.gold != "ABCDEFG"[n]


def test_cyclic_shifts_put_every_option_at_every_letter_once() -> None:
    bank = _bank(_rows(3), n_permutations=4)
    assert len(bank.items) == 12
    for start in range(0, 12, 4):
        shifts = bank.items[start : start + 4]
        base = shifts[0].option_order
        assert [s.permutation_id for s in shifts] == [0, 1, 2, 3]
        assert [s.option_order for s in shifts] == [base[p:] + base[:p] for p in range(4)]
        for letter in range(4):
            assert sorted(s.option_order[letter] for s in shifts) == [0, 1, 2, 3]
        assert sorted(s.gold for s in shifts) == ["A", "B", "C", "D"]
        assert all(list(_criteria(s).items())[-1] == ("E", "I don't know") for s in shifts)
        assert len({s.item_id for s in shifts}) == 4


@pytest.mark.parametrize("n_permutations", [0, 5])
def test_n_permutations_outside_one_to_n_options_is_rejected(n_permutations: int) -> None:
    with pytest.raises(ValueError, match="n_permutations"):
        _bank(_rows(2), n_permutations=n_permutations)


def test_request_matches_the_design_choice_mapping() -> None:
    item = _bank(_rows(1), order="evaluate_idk").items[0]
    payload = json.loads(item.request.model_dump_json())
    options = list(_criteria(item).values())[:4]
    assert payload == {
        "state": "Question 0?",
        "model": "jev-1.13.0",
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": "Which option correctly answers the question? A wrong answer "
                "costs 1 point, a correct answer earns 1 point, and E (I don't know) earns 0.",
                "criteria": dict(zip("ABCDE", [*options, "I don't know"], strict=True)),
            }
        },
    }
    assert item.item_id == "toy:q000:0"

    plain = _bank(_rows(1), idk=False).items[0]
    question = plain.request.questions["answer"]
    assert isinstance(question, ChoiceQuestion)
    assert list(question.criteria) == ["A", "B", "C", "D"]
    assert question.instructions == "Which option correctly answers the question?"


def test_options_sha256_hashes_evaluate_idk_rendering_of_the_options() -> None:
    item = _bank(_rows(1)).items[0]
    block = "\n".join(f"{key}) {text}" for key, text in _criteria(item).items())
    assert block.endswith("\nE) I don't know")
    assert item.options_sha256 == hashlib.sha256(block.encode()).hexdigest()


def test_duplicate_question_ids_are_rejected() -> None:
    rows = _rows(2)
    with pytest.raises(DatasetFormatError, match="duplicate question_id 'q000'"):
        _bank([rows[0], rows[1], rows[0]])


# --- bank file integrity ---------------------------------------------------------------------


def _rewrite_item(path: Path, line: int, edit: dict[str, object]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[line])
    for dotted, value in edit.items():
        target = record
        *parents, leaf = dotted.split(".")
        for key in parents:
            target = target[key]
        target[leaf] = value
    lines[line] = json.dumps(record, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        ({"gold": "E"}, "gold"),
        ({"request.questions.answer.criteria.A": "tampered"}, "options_sha256"),
        ({"option_order": [0, 0, 1, 2]}, "permutation"),
        ({"permutation_id": 3}, "item_id"),
    ],
)
def test_from_jsonl_rejects_an_edited_item(
    tmp_path: Path, edit: dict[str, object], message: str
) -> None:
    path = tmp_path / "bank.jsonl"
    _bank(_rows(3)).to_jsonl(path)
    _rewrite_item(path, 2, edit)
    with pytest.raises(BankFormatError, match=rf"bank\.jsonl:3: bad item(.|\n)*{message}"):
        QuestionBank.from_jsonl(path)


def test_from_jsonl_rejects_consistent_but_different_items(tmp_path: Path) -> None:
    # A swapped item passes per-item checks; only the bank-level items_sha256 catches it.
    path = tmp_path / "bank.jsonl"
    _bank(_rows(3)).to_jsonl(path)
    other = _bank(_rows(3), seed=99).items[1].model_dump_json()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[2] != other
    lines[2] = other
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(BankFormatError, match="items_sha256"):
        QuestionBank.from_jsonl(path)


def test_from_jsonl_rejects_a_missing_item_and_a_bad_header(tmp_path: Path) -> None:
    path = tmp_path / "bank.jsonl"
    _bank(_rows(3)).to_jsonl(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(BankFormatError, match="n_items 3 != 2"):
        QuestionBank.from_jsonl(path)
    path.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(BankFormatError, match=r"bank\.jsonl:1: bad metadata"):
        QuestionBank.from_jsonl(path)


# --- dataset parsers -------------------------------------------------------------------------

GPQA_HEADER = [
    "Question",
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
    "Subdomain",
    "Record ID",
    "High-level domain",
]


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_parse_gpqa_csv_builds_evaluate_idk_base_order(tmp_path: Path) -> None:
    path = tmp_path / "gpqa.csv"
    _write_csv(
        path,
        GPQA_HEADER,
        [[" Stem?\n", " right ", "w1", "w2\nmore", " w3", "sub", "rec1", "Physics"]],
    )
    [row] = parse_gpqa_csv(path)
    assert row == McqRow(
        question_id="rec1",
        question="Stem?",
        options=("w1", "w2\nmore", "w3", "right"),
        gold_index=3,
        subject="Physics",
    )


def test_parse_gpqa_csv_names_missing_columns_and_empty_cells(tmp_path: Path) -> None:
    path = tmp_path / "gpqa.csv"
    _write_csv(path, GPQA_HEADER[:-2], [["Q", "a", "b", "c", "d", "s"]])
    with pytest.raises(DatasetFormatError, match=r"missing GPQA columns \['Record ID', 'High"):
        parse_gpqa_csv(path)
    _write_csv(path, GPQA_HEADER, [["Q", "  ", "b", "c", "d", "s", "rec1", "Physics"]])
    with pytest.raises(DatasetFormatError, match=r"data row 1: empty \['Correct Answer'\]"):
        parse_gpqa_csv(path)


def _write_lexam(path: Path, records: list[dict[str, object]]) -> None:
    columns = {key: [r[key] for r in records] for key in records[0]}
    pq.write_table(pa.table(columns), path)


def _lexam_record(qid: str, language: str, choices: str, gold: int = 1) -> dict[str, object]:
    return {
        "question": f" {qid} stem \n",
        "choices": choices,
        "gold": gold,
        "course": "Course",
        "language": language,
        "area": "Private",
        "id": qid,
    }


def test_parse_lexam_parquet_keeps_english_rows_in_file_order(tmp_path: Path) -> None:
    path = tmp_path / "lexam.parquet"
    _write_lexam(
        path,
        [
            _lexam_record("en-1", "en", "[' a', 'b ', 'c', 'd']", gold=2),
            _lexam_record("de-1", "de", "['w', 'x', 'y', 'z']"),
            _lexam_record("en-2", "en", '["it\'s", "e", "f", "g"]', gold=0),
        ],
    )
    assert parse_lexam_parquet(path) == [
        McqRow("en-1", "en-1 stem", (" a", "b ", "c", "d"), 2, "Private"),
        McqRow("en-2", "en-2 stem", ("it's", "e", "f", "g"), 0, "Private"),
    ]


@pytest.mark.parametrize(
    ("choices", "gold", "message"),
    [
        ("not a list", 0, "choices is not a Python list"),
        ("[1, 2, 3, 4]", 0, "choices must be a list of strings"),
        ("['a', 'b', 'c', 'd']", 4, "gold_index 4 out of range"),
    ],
)
def test_parse_lexam_parquet_rejects_malformed_rows(
    tmp_path: Path, choices: str, gold: int, message: str
) -> None:
    path = tmp_path / "lexam.parquet"
    _write_lexam(
        path,
        [_lexam_record("ok", "en", "['a','b','c','d']"), _lexam_record("bad", "en", choices, gold)],
    )
    with pytest.raises(DatasetFormatError, match=rf"row 2 \(id 'bad'\): .*{message}"):
        parse_lexam_parquet(path)


# --- pinned download -------------------------------------------------------------------------


def _pin(content: bytes, **overrides: str) -> PinnedFile:
    fields = {
        "repo_id": "example/gated",
        "config": "default",
        "split": "train",
        "filename": "data.csv",
        "revision": "a" * 40,
        "license": "CC0-1.0",
    }
    if "sha256" not in overrides and "git_blob_sha1" not in overrides:
        overrides["git_blob_sha1"] = hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()
    return PinnedFile(**(fields | overrides))


def _serve(monkeypatch: pytest.MonkeyPatch, path: Path) -> list[tuple[object, ...]]:
    calls: list[tuple[object, ...]] = []

    def fake_download(repo_id: str, filename: str, **kwargs: object) -> str:
        calls.append((repo_id, filename, kwargs["repo_type"], kwargs["revision"]))
        return str(path)

    monkeypatch.setattr(bank_module, "hf_hub_download", fake_download)
    return calls


def test_fetch_pinned_checks_the_pinned_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "data.csv"
    path.write_bytes(b"pinned content\n")
    calls = _serve(monkeypatch, path)

    assert fetch_pinned(_pin(b"pinned content\n")) == (
        path,
        hashlib.sha256(b"pinned content\n").hexdigest(),
    )
    assert calls == [("example/gated", "data.csv", "dataset", "a" * 40)]
    with pytest.raises(DatasetIntegrityError, match="git blob"):
        fetch_pinned(_pin(b"other content\n"))
    with pytest.raises(DatasetIntegrityError, match="sha256"):
        fetch_pinned(_pin(b"", sha256="0" * 64))


def test_fetch_pinned_explains_gated_access(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args: object, **kwargs: object) -> str:
        request = httpx.Request("GET", "https://huggingface.co/datasets/example/gated")
        raise GatedRepoError("403 Client Error", response=httpx.Response(403, request=request))

    monkeypatch.setattr(bank_module, "hf_hub_download", denied)
    with pytest.raises(DatasetAccessError, match=r"huggingface\.co/datasets/example/gated"):
        fetch_pinned(_pin(b"x"))


@pytest.mark.parametrize(
    ("dataset", "kwargs", "message"),
    [
        ("gpqa", {}, "unknown dataset 'gpqa'"),
        ("lexam_en", {"order": "random"}, "order must be one of"),
        ("lexam_en", {"limit": 0}, "limit must be >= 1"),
    ],
)
def test_build_question_bank_rejects_bad_arguments_before_downloading(
    monkeypatch: pytest.MonkeyPatch, dataset: str, kwargs: dict[str, object], message: str
) -> None:
    def no_download(*args: object, **kwargs: object) -> str:
        raise AssertionError("must not download")

    monkeypatch.setattr(bank_module, "hf_hub_download", no_download)
    with pytest.raises(ValueError, match=message):
        build_question_bank(dataset, seed=0, idk=True, **kwargs)
