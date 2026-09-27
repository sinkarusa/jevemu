"""CLINC150 in-scope test queries as a 150-option intent Choice question.

The ``plus`` test split holds 4,500 in-scope queries (30 per intent) and 1,000 out-of-scope
ones labelled ``oos``. Out-of-scope queries are left out, and ``oos`` is not an option: it is
a rejection class with no meaning of its own, so it would test abstention rather than intent
recognition (the ``idk`` benchmarks cover abstention). The state is the user's request,
verbatim; the options are the 150 intent names (e.g. ``translate``) without descriptions, in the
dataset's label order. More than 52 options, so the emulator scores them by name (echo or
trie).
"""

from __future__ import annotations

from pathlib import Path

from jevemu.bench.datasets.hub import classification_items, load_items
from jevemu.bench.registry import register_benchmark
from jevemu.bench.spec import BenchConfig, BenchItem
from jevemu.eval.bank import DatasetSource, PinnedFile

NAME = "clinc150"
OUT_OF_SCOPE = "oos"
PIN = PinnedFile(
    repo_id="clinc/clinc_oos",
    config="plus",
    split="test",
    filename="plus/test-00000-of-00001.parquet",
    revision="155b9c710419136e17307b80d0a13e68cd46b4ec",
    license="CC-BY-3.0 (https://huggingface.co/datasets/clinc/clinc_oos)",
    row_filter=f"intent != '{OUT_OF_SCOPE}'",
    sha256="3e60e45b25bf86543aa5df8ba4fcc674114164e6184f0197690648c2908d0102",
)
INSTRUCTIONS = "Which intent does the user's request to the assistant express?"


def parse_clinc150(path: Path, source: DatasetSource) -> list[BenchItem]:
    return classification_items(
        NAME,
        path,
        source,
        text_column="text",
        label_column="intent",
        instructions=INSTRUCTIONS,
        exclude=frozenset({OUT_OF_SCOPE}),
    )


@register_benchmark(
    NAME,
    question_type="choice",
    license=PIN.license,
    splits={"test": PIN.split},
    tags=("classification", "many-options", "intent"),
)
def clinc150(cfg: BenchConfig) -> list[BenchItem]:
    return load_items(NAME, PIN, parse_clinc150, cfg)
