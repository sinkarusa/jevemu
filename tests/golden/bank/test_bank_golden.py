"""Question-bank goldens: evaluate-idk option-order parity and byte-stable bank files.

``fixtures/lexam_en_first5.json`` holds the first five English LEXam rows at the pinned
revision (CC-BY-4.0) and evaluate-idk's stored prompts for them. GPQA's terms forbid
publishing examples, so ``fixtures/gpqa_diamond_synthetic.csv`` is synthetic, in GPQA's schema.
After an intended bank format change, bump ``BUILDER_VERSION`` and regenerate the snapshots with
``JEVEMU_UPDATE_SNAPSHOTS=1 uv run pytest tests/golden/bank``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub import try_to_load_from_cache

from jevemu.bench import BenchConfig, bench_items_from_bank, load_benchmark
from jevemu.eval import bank as bank_module
from jevemu.eval.bank import (
    BANK_DATASETS,
    QuestionBank,
    bank_from_rows,
    build_question_bank,
    parse_lexam_parquet,
    render_options,
)
from jevemu.types import ChoiceQuestion

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures"
SNAPSHOTS = HERE / "snapshots"
UPDATE = os.environ.get("JEVEMU_UPDATE_SNAPSHOTS") == "1"
REPO = HERE.parents[2]
EVALUATE_IDK_DETAILS = REPO / "third_party" / "evaluate-idk" / "results" / "details"

LEXAM = json.loads((FIXTURES / "lexam_en_first5.json").read_text(encoding="utf-8"))
GPQA_CSV = FIXTURES / "gpqa_diamond_synthetic.csv"


def _check_snapshot(bank: QuestionBank, name: str) -> None:
    path = SNAPSHOTS / name
    if UPDATE:
        SNAPSHOTS.mkdir(exist_ok=True)
        bank.to_jsonl(path)
    assert bank.dumps() == path.read_text(encoding="utf-8"), f"{name} changed"
    assert QuestionBank.from_jsonl(path) == bank


def _lexam_parquet(tmp_path: Path) -> Path:
    rows: list[dict[str, Any]] = list(LEXAM["rows"])
    # A German row between English ones must not consume the evaluate-idk stream.
    rows.insert(2, rows[0] | {"id": "de-row", "language": "de"})
    path = tmp_path / "lexam.parquet"
    pq.write_table(pa.table({key: [r[key] for r in rows] for key in rows[0]}), path)
    return path


def _lexam_bank(tmp_path: Path, **kwargs: Any) -> QuestionBank:
    pin = BANK_DATASETS["lexam_en"].pin
    assert pin.sha256 is not None
    rows = parse_lexam_parquet(_lexam_parquet(tmp_path))
    return bank_from_rows("lexam_en", rows, source=pin.source(pin.sha256), **kwargs)


def _criteria(question: object) -> dict[str, str]:
    assert isinstance(question, ChoiceQuestion)
    return {key: str(text) for key, text in question.criteria.items()}


def test_lexam_evaluate_idk_order_matches_stored_evaluate_idk_prompts(tmp_path: Path) -> None:
    bank = _lexam_bank(tmp_path, seed=0, idk=True, order="evaluate_idk")
    assert len(bank.items) == len(LEXAM["evaluate_idk_expected"]) == 5
    for item, expected in zip(bank.items, LEXAM["evaluate_idk_expected"], strict=True):
        assert item.request.state == expected["question"]
        assert render_options(_criteria(item.question)) == expected["choices_block"]
        assert item.gold == "ABCDE"[expected["gold_index"]]
        assert item.options_sha256 == hashlib.sha256(expected["choices_block"].encode()).hexdigest()


def test_lexam_per_question_bank_bytes_are_stable(tmp_path: Path) -> None:
    _check_snapshot(
        _lexam_bank(tmp_path, seed=0, idk=True), "lexam_en_first5.per_question.seed0.jsonl"
    )


@pytest.fixture
def synthetic_gpqa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the synthetic CSV in place of the gated, pinned GPQA file."""
    digest = hashlib.sha256(GPQA_CSV.read_bytes()).hexdigest()
    monkeypatch.setattr(bank_module, "fetch_pinned", lambda pin: (GPQA_CSV, digest))


def _evaluate_idk_gpqa_prompts() -> list[tuple[str, str, int]]:
    # evaluate-idk's gpqa_diamond_idk_prompt + shuffle_choices over a Random(42) stream.
    stream = random.Random(42)
    prompts = []
    with GPQA_CSV.open(newline="", encoding="utf-8") as handle:
        for line in csv.DictReader(handle):
            correct = line["Correct Answer"].strip()
            choices = [line[f"Incorrect Answer {i}"].strip() for i in (1, 2, 3)] + [correct]
            stream.shuffle(choices)
            block = "".join(f"{letter}) {c}\n" for letter, c in zip("ABCD", choices, strict=True))
            prompts.append(
                (line["Question"].strip(), block + "E) I don't know", choices.index(correct))
            )
    return prompts


@pytest.mark.usefixtures("synthetic_gpqa")
def test_gpqa_bank_from_the_pinned_schema_matches_evaluate_idk_and_is_stable() -> None:
    bank = build_question_bank(
        "gpqa_diamond", seed=0, idk=True, order="evaluate_idk", n_permutations=4
    )
    assert bank.metadata.source.revision == BANK_DATASETS["gpqa_diamond"].pin.revision
    first_shifts = [item for item in bank.items if item.permutation_id == 0]
    for item, (stem, block, gold) in zip(first_shifts, _evaluate_idk_gpqa_prompts(), strict=True):
        assert item.request.state == stem
        assert render_options(_criteria(item.question)) == block
        assert item.gold == "ABCD"[gold]
    _check_snapshot(bank, "gpqa_diamond_synthetic.evaluate_idk.n4.jsonl")

    benchmark = load_benchmark("gpqa_diamond_idk", BenchConfig(params={"n_permutations": 4}))
    assert benchmark == bench_items_from_bank(bank)


@pytest.mark.usefixtures("synthetic_gpqa")
def test_gpqa_benchmark_limit_and_order_params() -> None:
    items = load_benchmark(
        "gpqa_diamond_idk", BenchConfig(seed=5, limit=2, params={"order": "per_question"})
    )
    bank = build_question_bank("gpqa_diamond", seed=5, idk=True, limit=2)
    assert [i.item_id for i in items] == [
        "gpqa_diamond:recSYNTHETIC0001:0",
        "gpqa_diamond:recSYNTHETIC0002:0",
    ]
    assert items == bench_items_from_bank(bank)


def _cached_lexam() -> str | None:
    pin = BANK_DATASETS["lexam_en"].pin
    found = try_to_load_from_cache(
        pin.repo_id, pin.filename, revision=pin.revision, repo_type="dataset"
    )
    return found if isinstance(found, str) else None


@pytest.mark.skipif(
    _cached_lexam() is None or not EVALUATE_IDK_DETAILS.is_dir(),
    reason="needs the pinned LEXam parquet in the HF cache and the evaluate-idk submodule",
)
def test_full_lexam_bank_reproduces_every_stored_december_evaluate_idk_prompt() -> None:
    # The December 2025 runs used this parquet (sha256 f4c10c42…); see the audit, §4.
    bank = build_question_bank("lexam_en", seed=0, idk=True, order="evaluate_idk")
    assert len(bank.items) == 619
    runs = sorted(
        EVALUATE_IDK_DETAILS.glob("**/2025-12-*/details_community|lexam-en-idk|0_*.parquet")
    )
    assert len(runs) == 7
    for run in runs:
        for doc in pq.read_table(run, columns=["doc"]).column("doc").to_pylist():
            item = bank.items[int(doc["id"])]
            query = doc["query"]
            stem, rest = query.removeprefix("Question:\n").split("\n\nChoices:\n", 1)
            block = rest.split("\n\nBefore answering", 1)[0]
            assert (item.request.state, render_options(_criteria(item.question)), item.gold) == (
                stem,
                block,
                "ABCDE"[doc["gold_index"]],
            ), f"{run.parent.name} doc {doc['id']}"
