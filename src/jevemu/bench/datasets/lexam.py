"""LEXam ``mcq_4_choices`` English test questions with an "I don't know" option
(evaluate-idk's ``lexam-en-idk``)."""

from __future__ import annotations

from jevemu.bench.registry import register_benchmark
from jevemu.bench.spec import BenchConfig, BenchItem, load_bank_benchmark
from jevemu.eval.bank import BANK_DATASETS

_PIN = BANK_DATASETS["lexam_en"].pin


@register_benchmark(
    "lexam_en_idk",
    question_type="choice",
    license=_PIN.license,
    splits={"test": _PIN.split},
    tags=("mcq", "idk", "evaluate-idk", "legal"),
)
def lexam_en_idk(cfg: BenchConfig) -> list[BenchItem]:
    """619 questions; options in evaluate-idk order unless ``cfg.params["order"]`` says not."""
    return load_bank_benchmark("lexam_en", cfg)
