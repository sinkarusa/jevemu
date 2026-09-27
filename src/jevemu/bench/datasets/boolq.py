"""BoolQ validation questions as Noul questions.

BoolQ has no labelled test split; its 3,270-question validation split is the standard test set
(2,033 yes, 1,237 no). The state is the Wikipedia passage, verbatim; the instructions are the
question with a ``?`` appended (BoolQ stores questions lowercase, without one). Gold is the
dataset's boolean answer.
"""

from __future__ import annotations

from pathlib import Path

from jevemu.bench.datasets.hub import load_items, make_item, nonblank, parquet_records
from jevemu.bench.registry import register_benchmark
from jevemu.bench.spec import BenchConfig, BenchItem
from jevemu.eval.bank import DatasetFormatError, DatasetSource, PinnedFile
from jevemu.types import NoulQuestion

NAME = "boolq"
PIN = PinnedFile(
    repo_id="google/boolq",
    config="default",
    split="validation",
    filename="data/validation-00000-of-00001.parquet",
    revision="35b264d03638db9f4ce671b711558bf7ff0f80d5",
    license="CC-BY-SA-3.0 (https://huggingface.co/datasets/google/boolq)",
    sha256="52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8",
)
COLUMNS = ("question", "answer", "passage")


def parse_boolq(path: Path, source: DatasetSource) -> list[BenchItem]:
    items: list[BenchItem] = []
    for index, record in enumerate(parquet_records(path, COLUMNS)):
        where = f"{path}: row {index}"
        answer = record["answer"]
        if not isinstance(answer, bool):
            raise DatasetFormatError(f"{where}: answer {answer!r} is not a boolean")
        items.append(
            make_item(
                NAME,
                str(index),
                state=nonblank(record["passage"], where),
                question=NoulQuestion(instructions=f"{nonblank(record['question'], where)}?"),
                gold=answer,
                stratum="yes" if answer else "no",
                source=source,
                where=where,
            )
        )
    return items


@register_benchmark(
    NAME,
    question_type="noul",
    license=PIN.license,
    splits={"test": PIN.split},
    tags=("yes-no", "reading-comprehension"),
)
def boolq(cfg: BenchConfig) -> list[BenchItem]:
    return load_items(NAME, PIN, parse_boolq, cfg)
