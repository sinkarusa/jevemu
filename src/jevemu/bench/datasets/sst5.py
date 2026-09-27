"""SST-5 (Stanford Sentiment Treebank, fine-grained) test sentences as 5-level Score questions.

2,210 sentences from movie reviews, from the sentence-level ``SetFit/sst5`` splits. The state
is the sentence, verbatim (SST's tokenized, lowercase text); the levels are the dataset's label
names, ``very negative`` (0) to ``very positive`` (4), and gold is the label.
"""

from __future__ import annotations

from pathlib import Path

from jevemu.bench.datasets.hub import jsonl_records, label_index, load_items, make_item, nonblank
from jevemu.bench.registry import register_benchmark
from jevemu.bench.spec import BenchConfig, BenchItem
from jevemu.eval.bank import DatasetFormatError, DatasetSource, PinnedFile
from jevemu.types import ScoreQuestion

NAME = "sst5"
PIN = PinnedFile(
    repo_id="SetFit/sst5",
    config="default",
    split="test",
    filename="test.jsonl",
    revision="e51bdcd8cd3a30da231967c1a249ba59361279a3",
    license=(
        "unknown: no license on the Hub cards of SetFit/sst5 or stanfordnlp/sst "
        "(https://huggingface.co/datasets/SetFit/sst5, https://nlp.stanford.edu/sentiment/)"
    ),
    git_blob_sha1="54eacda2affef6a368288d3fe7d3579f5b4a6c18",
)
LEVELS = ("very negative", "negative", "neutral", "positive", "very positive")
QUESTION = ScoreQuestion(
    instructions="What sentiment does this sentence from a movie review express?",
    criteria=list(LEVELS),
)


def parse_sst5(path: Path, source: DatasetSource) -> list[BenchItem]:
    items: list[BenchItem] = []
    for index, record in enumerate(jsonl_records(path, ("text", "label", "label_text"))):
        where = f"{path}: row {index}"
        level = label_index(record["label"], len(LEVELS), where)
        if record["label_text"] != LEVELS[level]:
            raise DatasetFormatError(f"{where}: label_text {record['label_text']!r} != {level}")
        items.append(
            make_item(
                NAME,
                str(index),
                state=nonblank(record["text"], where),
                question=QUESTION,
                gold=level,
                stratum=LEVELS[level],
                source=source,
                where=where,
            )
        )
    return items


@register_benchmark(
    NAME,
    question_type="score",
    license=PIN.license,
    splits={"test": PIN.split},
    tags=("ordinal", "sentiment"),
)
def sst5(cfg: BenchConfig) -> list[BenchItem]:
    return load_items(NAME, PIN, parse_sst5, cfg)
