"""Yelp review stars: a fixed 5,000-review sample of the test split as 5-level Score questions.

The test split has 50,000 reviews, 10,000 per star rating. The benchmark keeps 1,000 per rating
(:func:`sample_rows` with seed 0), in file order: a tenth of the Jev cost, and the standard error
of an accuracy is still at most 0.71 points. The state is the review text, verbatim; the levels
are ``1 star`` (0) to ``5 stars`` (4), the dataset's class names with its ``2 star`` typo fixed.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from jevemu.bench.datasets.hub import label_index, load_items, make_item, nonblank, parquet_records
from jevemu.bench.registry import register_benchmark
from jevemu.bench.spec import BenchConfig, BenchItem
from jevemu.eval.bank import DatasetSource, PinnedFile
from jevemu.types import ScoreQuestion

NAME = "yelp_stars"
SAMPLE_PER_LEVEL = 1_000
SAMPLE_SEED = 0
PIN = PinnedFile(
    repo_id="Yelp/yelp_review_full",
    config="yelp_review_full",
    split="test",
    filename="yelp_review_full/test-00000-of-00001.parquet",
    revision="c1f9ee939b7d05667af864ee1cb066393154bf85",
    license=(
        "Yelp Dataset Agreement (Hub license: other): non-commercial research use, no "
        "redistribution (https://huggingface.co/datasets/Yelp/yelp_review_full)"
    ),
    row_filter=f"{SAMPLE_PER_LEVEL} rows per label: smallest sha256('{SAMPLE_SEED}\\x1f<row>')",
    sha256="bf06d5969bff93ecd4a6d4b330643761ce108b42e9b8a894cf82c9df02a08540",
)
LEVELS = ("1 star", "2 stars", "3 stars", "4 stars", "5 stars")
QUESTION = ScoreQuestion(
    instructions="How many stars did the reviewer give?",
    criteria=list(LEVELS),
)


def sample_rows(labels: Sequence[int], *, per_level: int, seed: int) -> list[int]:
    """Row indices, ascending: for each label, the ``per_level`` rows (all, if fewer) with the
    smallest ``sha256(f"{seed}\\x1f{row}")``."""
    by_label: defaultdict[int, list[int]] = defaultdict(list)
    for row, label in enumerate(labels):
        by_label[label].append(row)
    kept: list[int] = []
    for rows in by_label.values():
        rows.sort(key=lambda row: hashlib.sha256(f"{seed}\x1f{row}".encode()).digest())
        kept.extend(rows[:per_level])
    return sorted(kept)


def parse_yelp(path: Path, source: DatasetSource) -> list[BenchItem]:
    records = parquet_records(path, ("label", "text"))
    levels = [
        label_index(record["label"], len(LEVELS), f"{path}: row {index}")
        for index, record in enumerate(records)
    ]
    items: list[BenchItem] = []
    for index in sample_rows(levels, per_level=SAMPLE_PER_LEVEL, seed=SAMPLE_SEED):
        where = f"{path}: row {index}"
        items.append(
            make_item(
                NAME,
                str(index),
                state=nonblank(records[index]["text"], where),
                question=QUESTION,
                gold=levels[index],
                stratum=LEVELS[levels[index]],
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
    tags=("ordinal", "sentiment", "sampled", "non-commercial"),
)
def yelp_stars(cfg: BenchConfig) -> list[BenchItem]:
    return load_items(NAME, PIN, parse_yelp, cfg)
