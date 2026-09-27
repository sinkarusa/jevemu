"""Benchmark registry: ``@register_benchmark`` on a loader, then look it up by name.

Built-in benchmarks (``jevemu.bench.datasets``) register on import; the module-level lookups
import them on first use.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping

from jevemu.bench.spec import BenchConfig, BenchItem, BenchmarkSpec, Loader, QuestionType


class BenchmarkRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, BenchmarkSpec] = {}

    def add(self, spec: BenchmarkSpec) -> BenchmarkSpec:
        if spec.name in self._specs:
            raise ValueError(f"benchmark {spec.name!r} is already registered")
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> BenchmarkSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(
                f"unknown benchmark {name!r}; registered: {sorted(self._specs)}"
            ) from None

    def specs(self) -> list[BenchmarkSpec]:
        """All specs, sorted by name."""
        return [self._specs[name] for name in sorted(self._specs)]


REGISTRY = BenchmarkRegistry()
"""The default registry; built-ins land here."""


def register_benchmark(
    name: str,
    *,
    question_type: QuestionType,
    license: str,
    splits: Mapping[str, str],
    tags: Iterable[str] = (),
    registry: BenchmarkRegistry | None = None,
) -> Callable[[Loader], Loader]:
    """Decorator registering ``loader`` as benchmark ``name``; returns the loader unchanged."""

    def decorate(loader: Loader) -> Loader:
        (registry or REGISTRY).add(
            BenchmarkSpec(
                name=name,
                question_type=question_type,
                loader=loader,
                splits=dict(splits),
                license=license,
                tags=frozenset(tags),
            )
        )
        return loader

    return decorate


def _load_builtins() -> None:
    importlib.import_module("jevemu.bench.datasets")


def get_benchmark(name: str) -> BenchmarkSpec:
    _load_builtins()
    return REGISTRY.get(name)


def list_benchmarks() -> list[BenchmarkSpec]:
    _load_builtins()
    return REGISTRY.specs()


def load_benchmark(benchmark: str | BenchmarkSpec, cfg: BenchConfig) -> list[BenchItem]:
    """Run a benchmark's loader and check its items against the spec.

    Raises ``ValueError`` if ``cfg.split`` is not one of the spec's split roles, if an item's
    question type differs from the spec's, or if two items share an ``item_id``.
    """
    spec = get_benchmark(benchmark) if isinstance(benchmark, str) else benchmark
    if cfg.split not in spec.splits:
        raise ValueError(
            f"benchmark {spec.name!r} has no split {cfg.split!r}; available: {sorted(spec.splits)}"
        )
    items = list(spec.loader(cfg))
    seen: set[str] = set()
    for item in items:
        if item.question.type != spec.question_type:
            raise ValueError(
                f"benchmark {spec.name!r} is {spec.question_type!r} but item {item.item_id!r} "
                f"holds a {item.question.type!r} question"
            )
        if item.item_id in seen:
            raise ValueError(f"benchmark {spec.name!r}: duplicate item_id {item.item_id!r}")
        seen.add(item.item_id)
    return items
