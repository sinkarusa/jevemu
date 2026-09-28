"""Summaries of run directories and paired comparisons of two.

A run directory is ``<out>/<system_id>/`` as written by :func:`~jevemu.bench.runner.run_split`.
:func:`summarize` scores every ``<benchmark>.<split>.jsonl`` in it
(:mod:`jevemu.eval.item_scores`): per benchmark the accuracy with its 95% bootstrap interval,
the IDK rate, mean p(gold), NLL, Brier, ECE, MAE and within-one accuracy for score questions,
latency p50/p95 and spend; then the macro average over benchmarks (each weighs the same) of
accuracy, NLL and Brier with stratified bootstrap intervals. Failed items are counted, not
scored.

:func:`compare` pairs two run directories on the item ids both answered successfully, per
benchmark present in both: the difference ``a - b`` of accuracy (paired bootstrap interval and
exact McNemar test), NLL and Brier, and the macro average of those differences with a
stratified bootstrap interval. It refuses runs of different splits (``items_sha256``) or items
whose gold differs.

:func:`stats` gives a run directory's token and cost statistics under a
:class:`~jevemu.eval.costs.PriceBook` (per-item costs: :mod:`jevemu.eval.costs`): per benchmark
and pooled, the items, input/output tokens (mean per item and total), prompt-cache hits,
backend calls per item, latency p50/p95, throughput, cost in total, per 1,000 items and per
1,000 correct answers, and GPU-hours for local GPU systems (emulator, CLM). :func:`stats_table`
lays several runs side by side, one row per system, with the macro accuracy next to the cost.

Throughput (q/s) is measured from the run manifest: the items the timed invocations sent to the
system (called, minus response-cache hits), divided by their summed start-to-end wall time (per
benchmark, and pooled as total items over total seconds). A cache hit costs almost no wall time,
so counting it would inflate q/s. An invocation that never finished, called nothing, or served
every item from the response cache (a pure replay) is left out; a benchmark with only such
invocations has no throughput. Wall time excludes vLLM server startup. Jev's throughput is set
by our client's rate limiter, not by Jev (:data:`JEV_THROUGHPUT_NOTE`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from jevemu.bench.manifest import Invocation, RunManifest, SplitRun, run_key
from jevemu.bench.runner import ItemRecord, load_records
from jevemu.bench.spec import QuestionType
from jevemu.eval.costs import (
    GpuTimePrice,
    ItemCost,
    Price,
    PriceBook,
    invocation_seconds,
    item_costs,
)
from jevemu.eval.item_scores import (
    Interval,
    ScoreSummary,
    ScoreTable,
    macro_intervals,
    score_answer,
    summarize_scores,
)
from jevemu.eval.metrics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    BootstrapCI,
    McNemarResult,
    mcnemar_exact,
    paired_bootstrap_ci,
)
from jevemu.eval.splits import DATASETS, DROPPED_DATASETS, SplitName

__all__ = [
    "JEV_THROUGHPUT_NOTE",
    "BenchmarkRun",
    "BenchmarkSummary",
    "Comparison",
    "PairedBenchmark",
    "RunStats",
    "RunSummary",
    "UsageStats",
    "compare",
    "load_run",
    "stats",
    "stats_table",
    "summarize",
    "throughput_note",
]

FloatArray = NDArray[np.float64]

JEV_THROUGHPUT_NOTE = (
    "Jev's q/s is set by our client's rate limit. The runs sent one question per request, and "
    "JevClient caps request starts at max_rpm = 1,200/min (20 requests/s), which is Jev's "
    "documented limit (1,200 requests/min and 250,000 tokens/s). Jev evaluates the questions of "
    "one request in parallel, so with N questions per request its ceiling is about 20·N q/s, up "
    "to 250,000 tokens/s (about 500 q/s at 500 tokens per question). This ceiling comes from the "
    "docs; we did not measure it."
)
"""Footnote wherever Jev's throughput is shown."""


def throughput_note(system: Mapping[str, Any]) -> str | None:
    """Why a system's measured q/s is not its capacity (``None``: no caveat known).

    Only Jev is known to be bounded by our client's rate limiter.
    """
    return JEV_THROUGHPUT_NOTE if system.get("kind") == "jev" else None


@dataclass(frozen=True)
class BenchmarkRun:
    """One records file of a run directory, loaded and scored."""

    benchmark: str
    split: SplitName
    entry: SplitRun | None
    """The manifest's entry (``None`` without a manifest)."""
    records: dict[str, ItemRecord]
    """Item id -> latest record, in file order."""
    table: ScoreTable
    """Scores of the successful records, in file order."""

    @property
    def ok_records(self) -> list[ItemRecord]:
        return [record for record in self.records.values() if record.ok]


def _benchmark_order(names: Iterable[str]) -> list[str]:
    known = [name for name in DATASETS if name in names]
    return known + sorted(set(names) - set(known))


def _table(records: Mapping[str, ItemRecord]) -> ScoreTable:
    ok = [record for record in records.values() if record.ok]
    types = {record.question_type for record in records.values()}
    if len(types) > 1:
        raise ValueError(f"records mix question types {sorted(types)}")
    question_type: QuestionType = types.pop() if types else "choice"
    scores = []
    for record in ok:
        assert record.answer is not None
        scores.append(score_answer(record.answer, record.gold, idk_key=record.idk_key))
    return ScoreTable.from_scores(
        question_type,
        [record.item_id for record in ok],
        scores,
        idk=any(record.idk_key is not None for record in records.values()),
    )


def load_run(
    run_dir: str | Path,
    *,
    split: SplitName = "select",
    benchmarks: Iterable[str] | None = None,
) -> dict[str, BenchmarkRun]:
    """Benchmark -> its loaded records file, for every ``<benchmark>.<split>.jsonl`` (only
    ``benchmarks`` if given; otherwise every one but :data:`DROPPED_DATASETS`)."""
    path = Path(run_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"no run directory {path}")
    manifest = RunManifest.load(path)
    suffix = f".{split}.jsonl"
    wanted = None if benchmarks is None else set(benchmarks)
    runs: dict[str, BenchmarkRun] = {}
    for file in sorted(path.glob(f"*{suffix}")):
        benchmark = file.name[: -len(suffix)]
        skip = (benchmark in DROPPED_DATASETS) if wanted is None else (benchmark not in wanted)
        if skip:
            continue
        records = load_records(file)
        entry = manifest.runs.get(run_key(benchmark, split)) if manifest else None
        runs[benchmark] = BenchmarkRun(benchmark, split, entry, records, _table(records))
    return {name: runs[name] for name in _benchmark_order(runs)}


# --- summaries -------------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkSummary:
    benchmark: str
    question_type: QuestionType
    n_items: int
    """Items the latest run requested (records in the file without a manifest)."""
    n_errors: int
    """Items whose latest attempt failed."""
    scores: ScoreSummary
    """Over the successful items (``scores.n`` of them)."""
    latency_p50_ms: float
    latency_p95_ms: float
    spent_usd: float
    n_cached: int
    """Jev answers served from the response cache."""


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    system_id: str
    system: dict[str, Any]
    split: SplitName
    benchmarks: dict[str, BenchmarkSummary]
    macro_accuracy: Interval
    macro_nll: Interval
    macro_brier: Interval
    n_errors: int
    spent_usd: float
    """Sum of the recorded items' cost."""

    def to_markdown(self) -> str:
        """The per-benchmark table plus the macro row (aggregates only, no item text)."""
        header = [
            "Benchmark",
            "Type",
            "n",
            "Errors",
            "Accuracy [95% CI]",
            "IDK rate",
            "NLL",
            "Brier",
            "ECE",
            "MAE",
            "Within-1",
            "p50 ms",
            "p95 ms",
            "Spend $",
        ]
        rows = []
        for b in self.benchmarks.values():
            s = b.scores
            rows.append(
                [
                    b.benchmark,
                    b.question_type,
                    f"{s.n}/{b.n_items}",
                    str(b.n_errors),
                    _interval(s.accuracy),
                    _opt(s.idk_rate),
                    f"{s.nll:.3f}",
                    f"{s.brier:.3f}",
                    f"{s.ece:.3f}",
                    _opt(s.mae),
                    _opt(s.within_one),
                    f"{b.latency_p50_ms:.0f}",
                    f"{b.latency_p95_ms:.0f}",
                    f"{b.spent_usd:.4f}",
                ]
            )
        n_scored = sum(b.scores.n for b in self.benchmarks.values())
        n_items = sum(b.n_items for b in self.benchmarks.values())
        rows.append(
            [
                f"**macro ({len(self.benchmarks)})**",
                "",
                f"{n_scored}/{n_items}",
                str(self.n_errors),
                _interval(self.macro_accuracy),
                "",
                _interval(self.macro_nll, 3),
                _interval(self.macro_brier, 3),
                "",
                "",
                "",
                "",
                "",
                f"{self.spent_usd:.4f}",
            ]
        )
        return _markdown_table(header, rows)


def summarize(
    run_dir: str | Path,
    *,
    split: SplitName = "select",
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> RunSummary:
    """Score every benchmark of ``split`` in ``run_dir`` (module docstring)."""
    path = Path(run_dir)
    runs = load_run(path, split=split)
    if not runs:
        raise ValueError(f"{path} has no {split} records")
    manifest = RunManifest.load(path)
    benchmarks: dict[str, BenchmarkSummary] = {}
    for name, run in runs.items():
        ok = run.ok_records
        p50, p95 = _latency_percentiles(ok)
        benchmarks[name] = BenchmarkSummary(
            benchmark=name,
            question_type=run.table.question_type,
            n_items=run.entry.n_items if run.entry else len(run.records),
            n_errors=sum(not r.ok for r in run.records.values()),
            scores=summarize_scores(run.table, n_resamples=n_resamples, seed=seed),
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            spent_usd=math.fsum(r.cost_usd or 0.0 for r in run.records.values()),
            n_cached=sum(bool(r.cached) for r in ok),
        )
    scored = {name: run.table for name, run in runs.items() if len(run.table)}
    macro = (
        macro_intervals(
            {
                name: np.column_stack([t.correct.astype(np.float64), t.nll, t.brier])
                for name, t in scored.items()
            },
            n_resamples=n_resamples,
            seed=seed,
        )
        if scored
        else [Interval(math.nan, math.nan, math.nan)] * 3
    )
    return RunSummary(
        run_dir=path,
        system_id=manifest.system_id if manifest else path.name,
        system=manifest.system if manifest else {},
        split=split,
        benchmarks=benchmarks,
        macro_accuracy=macro[0],
        macro_nll=macro[1],
        macro_brier=macro[2],
        n_errors=sum(b.n_errors for b in benchmarks.values()),
        spent_usd=math.fsum(b.spent_usd for b in benchmarks.values()),
    )


def _latency_percentiles(records: Iterable[ItemRecord]) -> tuple[float, float]:
    """(p50, p95) of the recorded latencies in ms; NaN without any."""
    latencies = np.array(
        [r.latency_ms for r in records if r.latency_ms is not None], dtype=np.float64
    )
    if not len(latencies):
        return math.nan, math.nan
    p50, p95 = np.percentile(latencies, [50, 95])
    return float(p50), float(p95)


# --- cost and token statistics ---------------------------------------------------------------


@dataclass(frozen=True)
class UsageStats:
    """Tokens, latency and cost of one benchmark's items, or of a whole run's."""

    n_items: int
    """Records, answered or failed; cost per 1k items divides by this."""
    n_ok: int
    n_correct: int
    accuracy: float
    """One benchmark: ``n_correct / n_ok``. A whole run: the macro average over benchmarks."""
    n_usage: int
    """Items that reported usage; the token means are over these."""
    input_tokens: int
    output_tokens: int
    cached_tokens: int | None
    """Prompt-cache hits summed over the :attr:`n_cached_reported` items that report them
    (``None`` when none does)."""
    n_cached_reported: int
    backend_calls: int | None
    """Emulator: backend calls summed over the :attr:`n_diagnostics` items with diagnostics."""
    n_diagnostics: int
    latency_p50_ms: float
    latency_p95_ms: float
    cost_usd: float | None
    """Sum of the items' costs (:mod:`jevemu.eval.costs`); ``None`` if any item is unpriced."""
    gpu_seconds: float | None
    """Time-priced systems: GPU wall time spent on these items."""
    spent_usd: float
    """Recorded spend (Jev: billed calls only; a response-cache hit records 0)."""
    n_billed: int
    billed_table_usd: float
    """The price table's cost of the billed items, to check :attr:`spent_usd` against."""
    n_price_mismatch: int
    n_response_cache_hits: int
    throughput_items: int
    """Items the timed invocations sent to the system, response-cache hits excluded (module
    docstring); q/s divides these by :attr:`wall_seconds`."""
    wall_seconds: float | None
    """Summed wall time of the timed invocations; ``None`` without any."""
    concurrency: tuple[int, ...]
    """Distinct in-flight limits of the timed invocations, ascending."""

    @property
    def items_per_second(self) -> float | None:
        """Measured throughput (q/s); ``None`` without a timed invocation."""
        if not self.wall_seconds or not self.throughput_items:
            return None
        return self.throughput_items / self.wall_seconds

    @property
    def mean_input_tokens(self) -> float:
        return self.input_tokens / self.n_usage if self.n_usage else math.nan

    @property
    def mean_output_tokens(self) -> float:
        return self.output_tokens / self.n_usage if self.n_usage else math.nan

    @property
    def mean_cached_tokens(self) -> float | None:
        if self.cached_tokens is None:
            return None
        return self.cached_tokens / self.n_cached_reported

    @property
    def calls_per_item(self) -> float | None:
        if self.backend_calls is None:
            return None
        return self.backend_calls / self.n_diagnostics

    @property
    def gpu_hours(self) -> float | None:
        return None if self.gpu_seconds is None else self.gpu_seconds / 3600.0

    @property
    def cost_per_1k_items(self) -> float | None:
        if self.cost_usd is None or not self.n_items:
            return None
        return self.cost_usd / self.n_items * 1000.0

    @property
    def cost_per_correct(self) -> float | None:
        """``None`` when unpriced or when no answer is correct."""
        if self.cost_usd is None or not self.n_correct:
            return None
        return self.cost_usd / self.n_correct


def _timed(invocations: Iterable[Invocation]) -> list[Invocation]:
    """Invocations whose wall time measures the system: finished, called something, and not a
    pure response-cache replay."""
    return [
        i
        for i in invocations
        if i.finished_at is not None and i.n_called and i.n_cache_hits < i.n_called
    ]


def _usage_stats(
    records: Sequence[ItemRecord],
    costs: Sequence[ItemCost],
    n_correct: int,
    accuracy: float,
    invocations: Sequence[Invocation],
) -> UsageStats:
    timed = _timed(invocations)
    usages = [c.usage for c in costs if c.usage is not None]
    cached = [u.cached_input_tokens for u in usages if u.cached_input_tokens is not None]
    calls = [r.diagnostics.n_backend_calls for r in records if r.diagnostics is not None]
    item_costs_usd = [c.cost_usd for c in costs]
    gpu = [c.gpu_seconds for c in costs if c.gpu_seconds is not None]
    billed = [c for c in costs if c.billed]
    p50, p95 = _latency_percentiles(r for r in records if r.ok)
    return UsageStats(
        n_items=len(records),
        n_ok=sum(r.ok for r in records),
        n_correct=n_correct,
        accuracy=accuracy,
        n_usage=len(usages),
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        cached_tokens=sum(cached) if cached else None,
        n_cached_reported=len(cached),
        backend_calls=sum(calls) if calls else None,
        n_diagnostics=len(calls),
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        cost_usd=(
            None
            if any(c is None for c in item_costs_usd)
            else math.fsum(c for c in item_costs_usd if c is not None)
        ),
        gpu_seconds=math.fsum(gpu) if gpu else None,
        spent_usd=math.fsum(r.cost_usd or 0.0 for r in records),
        n_billed=len(billed),
        billed_table_usd=math.fsum(c.table_usd or 0.0 for c in billed),
        n_price_mismatch=sum(c.price_mismatch for c in costs),
        n_response_cache_hits=sum(bool(r.cached) for r in records),
        throughput_items=sum(i.n_called - i.n_cache_hits for i in timed),
        wall_seconds=invocation_seconds(timed),
        concurrency=tuple(sorted({i.concurrency for i in timed})),
    )


@dataclass(frozen=True)
class RunStats:
    run_dir: Path
    system_id: str
    system: dict[str, Any]
    split: SplitName
    price: Price | None
    """``None``: the price book has no entry for this system; costs stay empty."""
    benchmarks: dict[str, UsageStats]
    total: UsageStats
    """Every benchmark pooled; its accuracy is the macro average over benchmarks."""

    @property
    def price_id(self) -> str:
        return "none" if self.price is None else self.price.price_id

    def notes(self) -> list[str]:
        """Where the costs come from, and any disagreement with recorded spend."""
        t = self.total
        notes = [
            "price: no entry for this system; costs left empty"
            if self.price is None
            else f"price: {self.price.describe()}"
        ]
        if isinstance(self.price, GpuTimePrice):
            hours = (
                "unknown (no finished invocation)" if t.gpu_hours is None else f"{t.gpu_hours:.3f}"
            )
            notes.append(
                f"GPU time: {hours} h, the summed start-to-end wall time of the invocations that "
                "wrote the records (vLLM server startup excluded), split over items by input + "
                "output tokens"
            )
        if t.n_billed:
            notes.append(
                f"recorded spend ${t.spent_usd:.4f} on {t.n_billed:,} billed items; the price "
                f"table gives ${t.billed_table_usd:.4f} for them, {t.n_price_mismatch} items "
                "differ by more than 1%"
            )
        if t.n_response_cache_hits and t.cost_usd is not None:
            notes.append(
                f"{t.n_response_cache_hits:,} response-cache hits (recorded $0) are costed at the "
                f"table price: ${t.cost_usd - t.spent_usd:.4f}"
            )
        notes.append(_throughput_line(t))
        if note := self.throughput_note:
            notes.append(f"{_DAGGER} {note}")
        return notes

    @property
    def throughput_note(self) -> str | None:
        """The system's throughput caveat (:func:`throughput_note`), if it has a measured q/s."""
        if self.total.items_per_second is None:
            return None
        return throughput_note(self.system)

    def to_markdown(self) -> str:
        """Per-benchmark detail plus the pooled row (aggregates only); a throughput caveat
        (:func:`throughput_note`) follows as a footnote."""
        header = [
            "Benchmark",
            "Items",
            "Acc.",
            "In tok/item",
            "Out tok/item",
            "In tok total",
            "Out tok total",
            "Cached tok/item",
            "Calls/item",
            "p50 ms",
            "p95 ms",
            "q/s",
            "In flight",
            "GPU-h",
            "Cost $",
            "$ / 1k items",
            "$ / 1k correct",
        ]
        note = self.throughput_note
        marked = note is not None
        rows = [_detail_row(name, s, marked) for name, s in self.benchmarks.items()]
        rows.append(
            _detail_row(f"**all ({len(self.benchmarks)}), macro acc.**", self.total, marked)
        )
        return _markdown_table(header, rows) + _footnotes([note] if note else [])


_DAGGER = "†"


def _throughput_line(s: UsageStats) -> str:
    qps = s.items_per_second
    if qps is None or s.wall_seconds is None:
        return "throughput: none measured (no finished invocation that called the system)"
    return (
        f"throughput: {qps:.2f} q/s, {s.throughput_items:,} items in {s.wall_seconds:,.0f} s of "
        f"summed invocation wall time at {_in_flight(s)} in flight (response-cache hits "
        "excluded)"
    )


def _qps(s: UsageStats, marked: bool) -> str:
    """``—`` when unmeasured (e.g. a pure response-cache replay); ``†`` marks a caveat."""
    qps = s.items_per_second
    if qps is None:
        return "—"
    return f"{qps:.2f} {_DAGGER}" if marked else f"{qps:.2f}"


def _in_flight(s: UsageStats) -> str:
    return "/".join(str(c) for c in s.concurrency) or "—"


def _footnotes(notes: Sequence[str]) -> str:
    return "".join(f"\n{_DAGGER} {note}\n" for note in notes)


def _detail_row(label: str, s: UsageStats, marked: bool = False) -> list[str]:
    return [
        label,
        f"{s.n_items:,}",
        f"{s.accuracy:.4f}",
        _count(s.mean_input_tokens),
        _count(s.mean_output_tokens),
        f"{s.input_tokens:,}",
        f"{s.output_tokens:,}",
        _count(s.mean_cached_tokens),
        _opt(s.calls_per_item, 2),
        f"{s.latency_p50_ms:.0f}",
        f"{s.latency_p95_ms:.0f}",
        _qps(s, marked),
        _in_flight(s),
        _opt(s.gpu_hours),
        _opt(s.cost_usd, 4),
        _opt(s.cost_per_1k_items, 4),
        _per_1k(s.cost_per_correct),
    ]


def stats(run_dir: str | Path, prices: PriceBook, *, split: SplitName = "select") -> RunStats:
    """Token and cost statistics of every benchmark of ``split`` in ``run_dir`` under
    ``prices`` (module docstring)."""
    path = Path(run_dir)
    runs = load_run(path, split=split)
    if not runs:
        raise ValueError(f"{path} has no {split} records")
    manifest = RunManifest.load(path)
    system = manifest.system if manifest else {}
    price = prices.resolve(system)
    benchmarks: dict[str, UsageStats] = {}
    all_records: list[ItemRecord] = []
    all_costs: list[ItemCost] = []
    accuracies: list[float] = []
    n_correct = 0
    all_invocations: list[Invocation] = []
    for name, run in runs.items():
        records = list(run.records.values())
        invocations = run.entry.invocations if run.entry else []
        costs = item_costs(records, price, invocations=invocations)
        correct = int(run.table.correct.sum())
        accuracy = correct / len(run.table) if len(run.table) else math.nan
        benchmarks[name] = _usage_stats(records, costs, correct, accuracy, invocations)
        all_invocations += invocations
        all_records += records
        all_costs += costs
        n_correct += correct
        if len(run.table):
            accuracies.append(accuracy)
    macro = math.fsum(accuracies) / len(accuracies) if accuracies else math.nan
    return RunStats(
        run_dir=path,
        system_id=manifest.system_id if manifest else path.name,
        system=system,
        split=split,
        price=price,
        benchmarks=benchmarks,
        total=_usage_stats(all_records, all_costs, n_correct, macro, all_invocations),
    )


def stats_table(runs: Sequence[RunStats]) -> str:
    """One row per run: macro accuracy next to tokens, latency, throughput and cost (all
    benchmarks); throughput caveats (:func:`throughput_note`) follow as footnotes."""
    header = [
        "System",
        "Price",
        "Items",
        "Macro acc.",
        "In tok/item",
        "Out tok/item",
        "Cached tok/item",
        "Calls/item",
        "p50 ms",
        "p95 ms",
        "q/s",
        "In flight",
        "GPU-h",
        "Cost $",
        "$ / 1k items",
        "$ / 1k correct",
    ]
    rows = []
    notes: list[str] = []
    for run in runs:
        t = run.total
        note = run.throughput_note
        if note and note not in notes:
            notes.append(note)
        rows.append(
            [
                f"`{run.system_id}`",
                run.price_id,
                f"{t.n_items:,}",
                f"{t.accuracy:.4f}",
                _count(t.mean_input_tokens),
                _count(t.mean_output_tokens),
                _count(t.mean_cached_tokens),
                _opt(t.calls_per_item, 2),
                f"{t.latency_p50_ms:.0f}",
                f"{t.latency_p95_ms:.0f}",
                _qps(t, note is not None),
                _in_flight(t),
                _opt(t.gpu_hours),
                _opt(t.cost_usd, 4),
                _opt(t.cost_per_1k_items, 4),
                _per_1k(t.cost_per_correct),
            ]
        )
    return _markdown_table(header, rows) + _footnotes(notes)


# --- paired comparison -----------------------------------------------------------------------


@dataclass(frozen=True)
class PairedBenchmark:
    benchmark: str
    n: int
    """Items both runs answered."""
    accuracy_a: float
    accuracy_b: float
    delta_accuracy: BootstrapCI
    mcnemar: McNemarResult
    delta_nll: BootstrapCI
    delta_brier: BootstrapCI


@dataclass(frozen=True)
class Comparison:
    """``a - b`` on shared items: positive Δ accuracy and negative Δ NLL/Brier favour ``a``."""

    run_a: Path
    run_b: Path
    split: SplitName
    benchmarks: dict[str, PairedBenchmark]
    macro_accuracy_a: float
    macro_accuracy_b: float
    macro_delta_accuracy: Interval
    macro_delta_nll: Interval
    macro_delta_brier: Interval

    def to_markdown(self) -> str:
        header = [
            "Benchmark",
            "n",
            "Acc a",
            "Acc b",
            "Δ acc [95% CI]",
            "McNemar a/b, p",
            "Δ NLL [95% CI]",
            "Δ Brier [95% CI]",
        ]
        rows = [
            [
                p.benchmark,
                str(p.n),
                f"{p.accuracy_a:.4f}",
                f"{p.accuracy_b:.4f}",
                _ci(p.delta_accuracy),
                f"{p.mcnemar.only_a}/{p.mcnemar.only_b}, {_pvalue(p.mcnemar.p_value)}",
                _ci(p.delta_nll, 3),
                _ci(p.delta_brier, 3),
            ]
            for p in self.benchmarks.values()
        ]
        rows.append(
            [
                f"**macro ({len(self.benchmarks)})**",
                str(sum(p.n for p in self.benchmarks.values())),
                f"{self.macro_accuracy_a:.4f}",
                f"{self.macro_accuracy_b:.4f}",
                _interval(self.macro_delta_accuracy, signed=True),
                "",
                _interval(self.macro_delta_nll, 3, signed=True),
                _interval(self.macro_delta_brier, 3, signed=True),
            ]
        )
        return _markdown_table(header, rows)


def compare(
    run_dir_a: str | Path,
    run_dir_b: str | Path,
    *,
    split: SplitName = "select",
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 0,
    benchmarks: Iterable[str] | None = None,
) -> Comparison:
    """Pair two run directories on shared successful item ids (module docstring); with
    ``benchmarks``, only those (a sensitivity analysis leaving some out)."""
    runs_a = load_run(run_dir_a, split=split, benchmarks=benchmarks)
    runs_b = load_run(run_dir_b, split=split, benchmarks=benchmarks)
    paired: dict[str, PairedBenchmark] = {}
    diffs: dict[str, FloatArray] = {}
    means: dict[str, tuple[float, float]] = {}
    shared_names = set(runs_a) & set(runs_b)
    if benchmarks is not None:
        shared_names &= set(benchmarks)
    for name in _benchmark_order(shared_names):
        a, b = runs_a[name], runs_b[name]
        if a.entry and b.entry and a.entry.split_metadata != b.entry.split_metadata:
            raise ValueError(f"{name}: the runs answered different {split} splits")
        answered_b = set(b.table.item_ids)
        shared = [i for i in a.table.item_ids if i in answered_b]
        for item_id in shared:
            if a.records[item_id].gold != b.records[item_id].gold:
                raise ValueError(f"{name}: item {item_id} has different gold in the two runs")
        if not shared:
            continue
        ta, tb = a.table.take(shared), b.table.take(shared)
        ca, cb = ta.correct.astype(np.float64), tb.correct.astype(np.float64)

        def ci(x: FloatArray, y: FloatArray) -> BootstrapCI:
            return paired_bootstrap_ci(x, y, n_resamples=n_resamples, seed=seed)

        paired[name] = PairedBenchmark(
            benchmark=name,
            n=len(shared),
            accuracy_a=float(ca.mean()),
            accuracy_b=float(cb.mean()),
            delta_accuracy=ci(ca, cb),
            mcnemar=mcnemar_exact(ta.correct, tb.correct),
            delta_nll=ci(ta.nll, tb.nll),
            delta_brier=ci(ta.brier, tb.brier),
        )
        diffs[name] = np.column_stack([ca - cb, ta.nll - tb.nll, ta.brier - tb.brier])
        means[name] = (float(ca.mean()), float(cb.mean()))
    if not paired:
        raise ValueError(f"{run_dir_a} and {run_dir_b} share no answered {split} items")
    macro = macro_intervals(diffs, n_resamples=n_resamples, seed=seed)
    return Comparison(
        run_a=Path(run_dir_a),
        run_b=Path(run_dir_b),
        split=split,
        benchmarks=paired,
        macro_accuracy_a=math.fsum(m[0] for m in means.values()) / len(means),
        macro_accuracy_b=math.fsum(m[1] for m in means.values()) / len(means),
        macro_delta_accuracy=macro[0],
        macro_delta_nll=macro[1],
        macro_delta_brier=macro[2],
    )


# --- formatting ------------------------------------------------------------------------------


def _markdown_table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _opt(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _count(value: float | None) -> str:
    return "" if value is None or math.isnan(value) else f"{value:,.1f}"


def _per_1k(value: float | None) -> str:
    return _opt(None if value is None else value * 1000.0, 4)


def _num(value: float, digits: int, signed: bool) -> str:
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _interval(value: Interval, digits: int = 4, *, signed: bool = False) -> str:
    return (
        f"{_num(value.estimate, digits, signed)} "
        f"[{_num(value.low, digits, signed)}, {_num(value.high, digits, signed)}]"
    )


def _ci(value: BootstrapCI, digits: int = 4) -> str:
    return _interval(Interval(value.estimate, value.low, value.high), digits, signed=True)


def _pvalue(p: float) -> str:
    return "1" if p >= 0.9995 else f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"
