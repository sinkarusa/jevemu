"""AG News test articles as a 4-option topic Choice question.

7,600 articles, 1,900 per class. The state is the article text (title and description),
verbatim; the options are the dataset's class names (``World``, ``Sports``, ``Business``,
``Sci/Tech``) without descriptions.
"""

from __future__ import annotations

from pathlib import Path

from jevemu.bench.datasets.hub import classification_items, load_items
from jevemu.bench.registry import register_benchmark
from jevemu.bench.spec import BenchConfig, BenchItem
from jevemu.eval.bank import DatasetSource, PinnedFile

NAME = "ag_news"
PIN = PinnedFile(
    repo_id="fancyzhx/ag_news",
    config="default",
    split="test",
    filename="data/test-00000-of-00001.parquet",
    revision="eb185aade064a813bc0b7f42de02595523103ca4",
    license=(
        "unknown on the Hub card; the source AG's corpus is provided for research and other "
        "non-commercial use (https://huggingface.co/datasets/fancyzhx/ag_news)"
    ),
    sha256="71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb",
)
INSTRUCTIONS = "What is the topic of this news article?"


def parse_ag_news(path: Path, source: DatasetSource) -> list[BenchItem]:
    return classification_items(
        NAME, path, source, text_column="text", label_column="label", instructions=INSTRUCTIONS
    )


@register_benchmark(
    NAME,
    question_type="choice",
    license=PIN.license,
    splits={"test": PIN.split},
    tags=("classification", "non-commercial"),
)
def ag_news(cfg: BenchConfig) -> list[BenchItem]:
    return load_items(NAME, PIN, parse_ag_news, cfg)
