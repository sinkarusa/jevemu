"""ARC-Challenge test questions as Choice questions.

1,172 questions with 3-5 options in the dataset's order. The 22 questions whose options are
labelled with digits instead of letters are keyed ``A``, ``B``, ... like the rest. The state is
the question text, verbatim; the instructions are the question banks' ("Which option correctly
answers the question?"). ARC has no subject field, so the split stratum is the gold letter.
"""

from __future__ import annotations

from pathlib import Path

from jevemu.bench.datasets.hub import load_items, make_item, mcq_question, nonblank, parquet_records
from jevemu.bench.registry import register_benchmark
from jevemu.bench.spec import BenchConfig, BenchItem
from jevemu.eval.bank import OPTION_LETTERS, DatasetFormatError, DatasetSource, PinnedFile

NAME = "arc_challenge"
PIN = PinnedFile(
    repo_id="allenai/ai2_arc",
    config="ARC-Challenge",
    split="test",
    filename="ARC-Challenge/test-00000-of-00001.parquet",
    revision="210d026faf9955653af8916fad021475a3f00453",
    license="CC-BY-SA-4.0 (https://huggingface.co/datasets/allenai/ai2_arc)",
    sha256="62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9",
)
COLUMNS = ("id", "question", "choices", "answerKey")


def parse_arc(path: Path, source: DatasetSource) -> list[BenchItem]:
    items: list[BenchItem] = []
    for index, record in enumerate(parquet_records(path, COLUMNS)):
        where = f"{path}: row {index} (id {record['id']!r})"
        choices = record["choices"]
        if not isinstance(choices, dict):
            raise DatasetFormatError(f"{where}: choices must be a struct")
        question = mcq_question(choices.get("text"), where)
        n = len(question.criteria)
        labels = choices.get("label")
        if labels not in (list(OPTION_LETTERS[:n]), [str(i) for i in range(1, n + 1)]):
            raise DatasetFormatError(f"{where}: unexpected option labels {labels!r}")
        if record["answerKey"] not in labels:
            raise DatasetFormatError(f"{where}: answerKey {record['answerKey']!r} not in {labels}")
        gold = OPTION_LETTERS[labels.index(record["answerKey"])]
        items.append(
            make_item(
                NAME,
                nonblank(record["id"], where),
                state=nonblank(record["question"], where),
                question=question,
                gold=gold,
                stratum=gold,
                source=source,
                where=where,
            )
        )
    return items


@register_benchmark(
    NAME,
    question_type="choice",
    license=PIN.license,
    splits={"test": PIN.split},
    tags=("mcq",),
)
def arc_challenge(cfg: BenchConfig) -> list[BenchItem]:
    return load_items(NAME, PIN, parse_arc, cfg)
