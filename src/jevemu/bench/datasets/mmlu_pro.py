"""MMLU-Pro test questions (up to 10 options) as Choice questions.

12,032 questions in 14 categories. Options keep the dataset's order, keyed ``A``..``J``; 2,051
questions have 3-9 options, so fewer letters. The state is the question text, verbatim; the
instructions are the question banks' ("Which option correctly answers the question?").
"""

from __future__ import annotations

from pathlib import Path

from jevemu.bench.datasets.hub import (
    label_index,
    load_items,
    make_item,
    mcq_question,
    nonblank,
    parquet_records,
)
from jevemu.bench.registry import register_benchmark
from jevemu.bench.spec import BenchConfig, BenchItem
from jevemu.eval.bank import OPTION_LETTERS, DatasetFormatError, DatasetSource, PinnedFile

NAME = "mmlu_pro"
PIN = PinnedFile(
    repo_id="TIGER-Lab/MMLU-Pro",
    config="default",
    split="test",
    filename="data/test-00000-of-00001.parquet",
    revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
    license="MIT (https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)",
    sha256="0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8",
)
COLUMNS = ("question_id", "question", "options", "answer", "answer_index", "category", "src")


def parse_mmlu_pro(path: Path, source: DatasetSource) -> list[BenchItem]:
    """Stratum and ``subject``: ``category``; ``src`` names the question's origin."""
    items: list[BenchItem] = []
    for index, record in enumerate(parquet_records(path, COLUMNS)):
        where = f"{path}: row {index} (question_id {record['question_id']!r})"
        question = mcq_question(record["options"], where)
        gold = OPTION_LETTERS[label_index(record["answer_index"], len(question.criteria), where)]
        if record["answer"] != gold:
            raise DatasetFormatError(f"{where}: answer {record['answer']!r} != {gold!r}")
        category = nonblank(record["category"], where)
        items.append(
            make_item(
                NAME,
                str(record["question_id"]),
                state=nonblank(record["question"], where),
                question=question,
                gold=gold,
                stratum=category,
                source=source,
                where=where,
                subject=category,
                src=record["src"],
            )
        )
    return items


@register_benchmark(
    NAME,
    question_type="choice",
    license=PIN.license,
    splits={"test": PIN.split},
    tags=("mcq", "10-options"),
)
def mmlu_pro(cfg: BenchConfig) -> list[BenchItem]:
    return load_items(NAME, PIN, parse_mmlu_pro, cfg)
