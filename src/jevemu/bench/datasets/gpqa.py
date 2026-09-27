"""GPQA-Diamond with an "I don't know" option (evaluate-idk's ``gpqa-diamond-idk``).

Gated on the Hub: the account behind the Hugging Face token must accept the dataset terms.
"""

from __future__ import annotations

from jevemu.bench.registry import register_benchmark
from jevemu.bench.spec import BenchConfig, BenchItem, load_bank_benchmark
from jevemu.eval.bank import BANK_DATASETS

_PIN = BANK_DATASETS["gpqa_diamond"].pin


@register_benchmark(
    "gpqa_diamond_idk",
    question_type="choice",
    license=_PIN.license,
    splits={"test": _PIN.split},
    tags=("mcq", "idk", "evaluate-idk", "gated"),
)
def gpqa_diamond_idk(cfg: BenchConfig) -> list[BenchItem]:
    """198 questions; options in evaluate-idk order unless ``cfg.params["order"]`` says not."""
    return load_bank_benchmark("gpqa_diamond", cfg)
