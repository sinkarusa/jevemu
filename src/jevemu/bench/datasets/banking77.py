"""Banking77 test queries as a 77-option intent Choice question.

3,080 queries, 40 per intent. The state is the customer's query, verbatim; the options are the
dataset's intent names (e.g. ``card_arrival``) without descriptions, in its label order. More
than 52 options, so the emulator scores them by name (echo or trie).

The file is the Hub's ``legacy-datasets/banking77`` parquet: ``PolyAI/banking77`` only holds a
loading script that downloads the unpinned CSVs from GitHub. The pinned file matches
``PolyAI-LDN/task-specific-datasets`` ``banking_data/test.csv`` row for row (checked
2026-09-24).
"""

from __future__ import annotations

from pathlib import Path

from jevemu.bench.datasets.hub import classification_items, load_items
from jevemu.bench.registry import register_benchmark
from jevemu.bench.spec import BenchConfig, BenchItem
from jevemu.eval.bank import DatasetSource, PinnedFile

NAME = "banking77"
PIN = PinnedFile(
    repo_id="legacy-datasets/banking77",
    config="default",
    split="test",
    filename="data/test-00000-of-00001.parquet",
    revision="f54121560de48f2852f90be299010d1d6dc612ec",
    license="CC-BY-4.0 (https://huggingface.co/datasets/PolyAI/banking77)",
    sha256="318da70fb77a0e01bcfaecc97ef6e3645313aab98c429f0f0630c9d48e703ecc",
)
INSTRUCTIONS = "Which banking intent does the customer's query express?"


def parse_banking77(path: Path, source: DatasetSource) -> list[BenchItem]:
    return classification_items(
        NAME, path, source, text_column="text", label_column="label", instructions=INSTRUCTIONS
    )


@register_benchmark(
    NAME,
    question_type="choice",
    license=PIN.license,
    splits={"test": PIN.split},
    tags=("classification", "many-options", "intent"),
)
def banking77(cfg: BenchConfig) -> list[BenchItem]:
    return load_items(NAME, PIN, parse_banking77, cfg)
