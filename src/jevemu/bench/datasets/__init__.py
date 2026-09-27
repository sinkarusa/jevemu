"""Built-in benchmarks (registered on import) and the custom-file adapters."""

from __future__ import annotations

from jevemu.bench.datasets import (
    ag_news,
    arc,
    banking77,
    boolq,
    clinc150,
    gpqa,
    lexam,
    mmlu_pro,
    sst5,
    yelp,
)
from jevemu.bench.datasets.custom import (
    ColumnMapping,
    CustomDataError,
    custom_benchmark,
    custom_csv,
    custom_jsonl,
)

__all__ = [
    "ColumnMapping",
    "CustomDataError",
    "ag_news",
    "arc",
    "banking77",
    "boolq",
    "clinc150",
    "custom_benchmark",
    "custom_csv",
    "custom_jsonl",
    "gpqa",
    "lexam",
    "mmlu_pro",
    "sst5",
    "yelp",
]
