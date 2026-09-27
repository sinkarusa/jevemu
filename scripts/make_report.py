"""Regenerate the Jev vs Qwen report, ``reports/jev_vs_qwen/``, from the recorded runs.

    uv run --extra plot python scripts/make_report.py

Compares Jev with the dense emulator ("Qwen 27B", Qwen3.6-27B, preset
``qwen3.6-27b-int4-quanttrio``) and the fast MoE emulator ("Qwen 35B-A3B", Qwen3.6-35B-A3B, preset
``qwen3.6-35b-a3b-fast``). The constants that name the runs, the study and the items in flight
sit together below the imports.

Writes ``README.md`` (GitHub markdown: tables and relative image links), ``metrics.json`` (every
number in the tables and figures) and ``figures/*.png`` (seaborn on pandas frames, matplotlib's
Agg backend, fixed theme, 150 dpi).
The output depends only on the inputs below: a rerun on the same runs rewrites the same files.

:data:`EXCLUDED_BENCHMARKS` (Yelp) is left out of everything the report shows: per-benchmark
numbers are the study's, and every macro and pooled number is recomputed over the other nine
(macro intervals from the study's bootstrap re-run on its per-item probabilities with its seeds
and resamples, :func:`replicates`; checked against the study's per-benchmark intervals).

Inputs (``runs/`` is gitignored; a missing input stops the script with its path):

- The holdout calibration study :data:`STUDY_DIR` (docs/research/calibration.md),
  ``metrics.json`` and ``predictions.<system>.jsonl``::

      uv run python scripts/calibrate.py crossfit jev=runs/select/jev \\
          quanttrio=runs/select/qwen3.6-27b-int4-quanttrio.auto_single.state_first \\
          fast=runs/select/qwen3.6-35b-a3b-fast.auto_single.state_first \\
          gptq=runs/select/qwen3.6-35b-a3b-int4-palmfuture.auto_single.state_first \\
          gemma=runs/select/gemma-4-26b-a4b-int4-cyankiwi.auto_single.state_first \\
          --split holdout --reference jev --debias none --out runs/calibration/holdout_single

  Accuracy, NLL, Brier and ECE per benchmark, with paired bootstrap intervals against Jev and
  reliability bins. The report reads systems ``jev``, ``quanttrio`` and ``fast`` (other systems
  of the study are ignored), arms ``raw`` and ``temperature@signature`` (cross-fitted,
  "calibrated"); each must name the run below (checked against the study's config). The per-item
  probabilities give the mean top probability, the pooled reliability diagram, the
  top-probability histograms under the reliability diagrams, the macro intervals and the paired
  Qwen 35B-A3B minus Qwen 27B intervals. Only aggregates are written; the files must pair on the
  same items and agree with ``metrics.json`` (checked).
- One run directory per system, :data:`RUN`: ``runs/select/jev``,
  ``runs/select/qwen3.6-27b-int4-quanttrio.auto_single.state_first`` and
  ``runs/select/qwen3.6-35b-a3b-fast.auto_single.state_first``. Split ``select`` is the cost
  source: tokens, latency, throughput and cost (``jevemu.bench.compare.stats``, the numbers
  ``scripts/run_split.py stats`` prints, pooled over the reported benchmarks). Split ``holdout``
  is the study's run, a cross-check of those costs shown in the report. Every system answered
  every study item on both splits; every emulator record of both splits was answered in exactly
  one model call (diagnostics ``n_backend_calls == 1``), with no debiaser (checked).

Jev and Qwen 27B must have measured their throughput at :data:`IN_FLIGHT` items in flight; the
fast preset runs at its tuned client concurrency, which every timed invocation of its ``select``
and ``holdout`` runs must share (checked; the report states it).

Prices: Jev's list price (``jevemu.eval.costs``: $0.042 per 1M input tokens, output free); the
local GPU's electricity, ``DEFAULT_GPU_WATTS`` x ``DEFAULT_USD_PER_WH`` (410 W x $0.00027/Wh =
$0.1107 per GPU-hour; no hardware amortization), fixed here so the environment cannot change it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray

from jevemu.bench.compare import (
    JEV_THROUGHPUT_NOTE,
    BenchmarkRun,
    RunStats,
    UsageStats,
    load_run,
    stats,
)
from jevemu.eval.costs import DEFAULT_GPU_WATTS, DEFAULT_USD_PER_WH, GpuTimePrice, PriceBook
from jevemu.eval.metrics import NLL_EPS, ece
from jevemu.eval.metrics_calib import ItemMetrics, bootstrap_replicates, interval, reliability
from jevemu.eval.splits import DATASETS, SplitName
from jevemu.scoring.single_call import LETTER_LIMIT

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd  # type: ignore[import-untyped]
    import seaborn as sns  # type: ignore[import-untyped]
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullFormatter
    from matplotlib.transforms import Bbox
except ImportError:  # pragma: no cover - depends on the environment
    sys.exit("make_report: matplotlib/pandas/seaborn missing; run with `uv run --extra plot`")

FloatArray = NDArray[np.float64]

MINUS, TIMES = "\u2212", "\u00d7"
"""Typographic minus and multiplication signs for the report text."""

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = Path("reports/jev_vs_qwen")
STUDY_SPLIT: SplitName = "holdout"
COST_SPLIT: SplitName = "select"

# --- the compared systems: runs, studies and concurrency (module docstring) --------------------

JEV, QWEN, FAST = "jev", "qwen", "fast"
SYSTEMS = (JEV, QWEN, FAST)
"""Every system the report can compare, in table order (:func:`compared` picks them)."""
NAME = {JEV: "Jev", QWEN: "Qwen 27B", FAST: "Qwen 35B-A3B"}
"""Short names in the tables, the text and the legends."""
COLOR = {JEV: "#2b6cb0", QWEN: "#dd6b20", FAST: "#2f855a"}
LIGHT = {JEV: "#c3d7ee", QWEN: "#f7d6bd", FAST: "#c6f6d5"}
"""Figure colors per system; :data:`LIGHT` marks the raw series next to the calibrated one."""
STUDY_DIR = Path("runs/calibration/holdout_single")
"""The crossfit study of the holdout runs (module docstring); it holds every compared system."""
STUDY_SYSTEM = {JEV: "jev", QWEN: "quanttrio", FAST: "fast"}
"""Each system's name in the study: the name of its holdout run (``config.runs``), whose
probabilities the study scores as recorded (no debiaser, checked)."""
JEV_RUN = Path("runs/select/jev")
QWEN_RUN = Path("runs/select/qwen3.6-27b-int4-quanttrio.auto_single.state_first")
FAST_RUN = Path("runs/select/qwen3.6-35b-a3b-fast.auto_single.state_first")
RUN = {JEV: JEV_RUN, QWEN: QWEN_RUN, FAST: FAST_RUN}
"""Each system's run directory: its cost source on ``select`` and the study's run on
``holdout``."""
IN_FLIGHT = {JEV: 16, QWEN: 16}
"""Items in flight of every timed invocation of these systems' runs (checked). The fast
emulator runs at its tuned concurrency, read from its runs (:func:`load_costs`)."""
EXCLUDED_BENCHMARKS: tuple[str, ...] = ("yelp_stars",)
"""Left out of this report (user decision): Yelp's labels ("1 star" .. "5 stars") read ambiguously.
Every table, macro average, pooled cost, figure and metrics.json entry uses the rest; a study may
hold it or not (its rows are skipped), and docs/research keep the 10-benchmark numbers."""
BENCHMARKS: tuple[str, ...] = tuple(b for b in DATASETS if b not in EXCLUDED_BENCHMARKS)

RAW, CAL = "raw", "calibrated"
ARM = {RAW: "raw", CAL: "temperature@signature"}
MACRO = "macro"
DELTAS = ((QWEN, JEV), (FAST, JEV), (FAST, QWEN))
"""The paired differences reported among the compared systems: each emulator minus Jev (per
benchmark, the study's ``delta_vs_reference``) and the fast emulator minus the dense one."""
JEV_ROUND_TO = 0.01
"""Jev returns probabilities rounded to this step: a correct answer it rates below half a step
comes back as exactly 0, which raw NLL clips to ``NLL_EPS`` (:func:`rounding_check`)."""
ROUNDING_FLOOR = JEV_ROUND_TO / 2
METRICS = ("accuracy", "nll", "brier", "ece")
INTERVAL_METRICS = ("accuracy", "nll", "brier")
"""Metrics with bootstrap intervals; ECE and mean top probability are point estimates."""
GPU = "one RTX 3090 (24 GB)"
CALL_NOTE = (
    f"Each question takes one model call. Questions with more than {LETTER_LIMIT} options are "
    "labelled with two-capital single-token codes (AA, AB, …) instead of letters."
)
"""How the emulators score (``jevemu.scoring.single_call``; one call per item, checked)."""
DAGGER = "†"
"""Marks Jev's throughput, which is our client's rate limit (:data:`JEV_THROUGHPUT_NOTE`)."""
JEV_MAX_RPS = 1200 / 60
"""JevClient's default ``max_rpm`` (Jev's documented limit), per second."""
PRESET_ENV = "JEVEMU_VLLM_PRESET"
"""The ``launch_env`` entry of an emulator manifest that names its preset."""

# Predictions store probabilities at 6 decimals, which can break near-ties and floor tiny
# probabilities; their metrics must still agree with metrics.json to these tolerances, and the
# intervals re-run on them (:func:`replicates`) with the study's to the looser ones.
AGREEMENT = {"accuracy": 5e-3, "nll": 5e-4, "brier": 1e-5, "ece": 5e-3}
INTERVAL_AGREEMENT = {"accuracy": 1e-2, "nll": 2e-3, "brier": 1e-4}
HIST_EDGES = np.round(np.linspace(0.0, 1.0, 21), 2)
"""Equal-width 0.05 bins of the top-probability histograms under the reliability diagrams."""

LABEL = {
    "gpqa_diamond_idk": "GPQA-Diamond",
    "lexam_en_idk": "LEXam-en",
    "mmlu_pro": "MMLU-Pro",
    "arc_challenge": "ARC-Challenge",
    "ag_news": "AG News",
    "banking77": "Banking77",
    "clinc150": "CLINC150",
    "boolq": "BoolQ",
    "sst5": "SST-5",
    "yelp_stars": "Yelp",
}
STYLE: dict[str, Any] = {
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
}
"""Overrides of seaborn's ``paper`` context and ``whitegrid`` style."""
DPI = 150


class ReportError(RuntimeError):
    """An input is missing or disagrees with another."""


def fail(message: str) -> NoReturn:
    raise ReportError(message)


def require(path: Path) -> Path:
    full = REPO / path
    if not full.exists():
        fail(f"missing input {path} (inputs: the docstring of scripts/make_report.py)")
    return full


def label(benchmark: str) -> str:
    return "Macro" if benchmark == MACRO else LABEL.get(benchmark, benchmark)


def delta_key(a: str, b: str) -> str:
    """The metrics.json key of the paired difference ``a - b``."""
    return f"{a}_minus_{b}"


@dataclass(frozen=True)
class Compared:
    """The systems this report compares and the study that holds all of them."""

    systems: tuple[str, ...]
    study_dir: Path

    @property
    def emulators(self) -> tuple[str, ...]:
        return tuple(s for s in self.systems if s != JEV)

    @property
    def deltas(self) -> tuple[tuple[str, str], ...]:
        return tuple((a, b) for a, b in DELTAS if a in self.systems and b in self.systems)


def compared() -> Compared:
    """Every system of :data:`SYSTEMS`, from :data:`STUDY_DIR`."""
    return Compared(SYSTEMS, STUDY_DIR)


# --- inputs ------------------------------------------------------------------------------------


def load_study(c: Compared) -> dict[str, Any]:
    where = c.study_dir / "metrics.json"
    study: dict[str, Any] = json.loads(require(where).read_text("utf-8"))
    if study["split"] != STUDY_SPLIT:
        fail(f"{where} is split {study['split']!r}, expected {STUDY_SPLIT!r}")
    for system in c.systems:
        if STUDY_SYSTEM[system] not in study["systems"]:
            fail(f"{where} has no system {STUDY_SYSTEM[system]!r}")
    for arm in ARM.values():
        if arm not in study["arms"]:
            fail(f"{where} has no arm {arm!r}")
    if study["reference"] != STUDY_SYSTEM[JEV]:
        fail(f"{where} has reference {study['reference']!r}, expected 'jev'")
    reported = tuple(b for b in study["benchmarks"] if b not in EXCLUDED_BENCHMARKS)
    if reported != BENCHMARKS:
        fail(f"{where} has benchmarks {study['benchmarks']}, expected {list(BENCHMARKS)}")
    for system in c.systems:
        name = STUDY_SYSTEM[system]
        run = study["config"]["runs"].get(name)
        if run != str(RUN[system]):
            fail(f"{where} ran {name}={run}, expected {RUN[system]}")
    return study


@dataclass(frozen=True)
class Predictions:
    """One system's per-item probabilities from the study's predictions file."""

    items: dict[str, dict[str, ItemMetrics]]
    """Arm key -> benchmark -> per-item metrics."""
    item_ids: dict[str, list[str]]
    """Benchmark -> item ids, in the file's order."""
    clusters: dict[str, list[str]]
    """Benchmark -> each item's question id, the study's bootstrap cluster."""


def load_predictions(study_dir: Path, system: str) -> Predictions:
    """The rows of :data:`BENCHMARKS`; those of :data:`EXCLUDED_BENCHMARKS` are skipped."""
    path = require(study_dir / f"predictions.{system}.jsonl")
    probs: dict[str, dict[str, list[FloatArray]]] = {k: {b: [] for b in BENCHMARKS} for k in ARM}
    gold: dict[str, list[int]] = {b: [] for b in BENCHMARKS}
    item_ids: dict[str, list[str]] = {b: [] for b in BENCHMARKS}
    clusters: dict[str, list[str]] = {b: [] for b in BENCHMARKS}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            benchmark = row["benchmark"]
            if benchmark in EXCLUDED_BENCHMARKS:
                continue
            if benchmark not in gold:
                fail(f"{path.name}: benchmark {benchmark!r} is not in metrics.json")
            gold[benchmark].append(int(row["gold"]))
            item_ids[benchmark].append(row["item_id"])
            clusters[benchmark].append(row["question_id"])
            for key, arm in ARM.items():
                probs[key][benchmark].append(np.asarray(row["arms"][arm], dtype=np.float64))
    items = {
        key: {b: ItemMetrics.from_probs(probs[key][b], gold[b]) for b in BENCHMARKS} for key in ARM
    }
    return Predictions(items=items, item_ids=item_ids, clusters=clusters)


def check_items(
    study: Mapping[str, Any], system: str, items: Mapping[str, Mapping[str, ItemMetrics]]
) -> None:
    for key, arm in ARM.items():
        for benchmark, metrics in items[key].items():
            if len(metrics) != study["n_items"][benchmark]:
                fail(
                    f"predictions.{system}.jsonl has {len(metrics)} {benchmark} items, "
                    f"metrics.json {study['n_items'][benchmark]}"
                )
            recorded = study["series"][system][arm][benchmark]
            for metric, value in metrics.estimate(n_bins=study["config"]["n_bins"]).items():
                if abs(value - recorded[metric][0]) > AGREEMENT[metric]:
                    fail(
                        f"predictions.{system}.jsonl disagrees with metrics.json on {arm} "
                        f"{benchmark} {metric}: {value} against {recorded[metric][0]}"
                    )


def check_paired(predictions: Mapping[str, Predictions]) -> None:
    """Every predictions file holds the same items in the same order, so their rows pair."""
    for system, p in predictions.items():
        if p.item_ids != predictions[JEV].item_ids or p.clusters != predictions[JEV].clusters:
            fail(
                f"predictions.{STUDY_SYSTEM[system]}.jsonl and "
                f"predictions.{STUDY_SYSTEM[JEV]}.jsonl hold different items"
            )


Series = tuple[str, str]
"""(system, arm key): one system's probabilities in one arm."""
Replicates = dict[str, dict[Series, dict[str, FloatArray]]]
"""Benchmark -> series -> metric -> bootstrap replicates."""


def replicates(study: Mapping[str, Any], predictions: Mapping[str, Predictions]) -> Replicates:
    """The study's paired cluster bootstrap, re-run on the predictions files: per benchmark of
    :data:`BENCHMARKS`, its resamples (question clusters, seed ``config.seed`` plus the
    benchmark's index in the study) shared by every series, so series and their differences
    get the replicates the study drew (:func:`check_intervals`)."""
    config = study["config"]
    systems = list(predictions)
    out: Replicates = {}
    for benchmark in BENCHMARKS:
        drawn = bootstrap_replicates(
            predictions[JEV].clusters[benchmark],
            {f"{s}/{k}": predictions[s].items[k][benchmark] for s in systems for k in ARM},
            n_resamples=int(config["n_resamples"]),
            seed=int(config["seed"]) + list(study["benchmarks"]).index(benchmark),
            n_bins=int(config["n_bins"]),
        )
        out[benchmark] = {(s, k): drawn[f"{s}/{k}"] for s in systems for k in ARM}
    return out


def estimate(study: Mapping[str, Any], series: Series, benchmark: str, metric: str) -> float:
    system, key = series
    return float(study["series"][STUDY_SYSTEM[system]][ARM[key]][benchmark][metric][0])


def combined(
    study: Mapping[str, Any],
    reps: Replicates,
    benchmarks: Sequence[str],
    a: Series,
    b: Series | None = None,
) -> dict[str, Any]:
    """Series ``a``, or the paired difference ``a - b``, averaged over ``benchmarks`` (one, or
    all for the macro, as the study averages): each metric's estimate from the study's
    per-benchmark estimates; accuracy, NLL and Brier with the 95% percentile interval of the
    averaged replicates, ECE as a point estimate."""
    out: dict[str, Any] = {}
    for metric in METRICS:
        value = float(np.mean([estimate(study, a, x, metric) for x in benchmarks]))
        if b is not None:
            value -= float(np.mean([estimate(study, b, x, metric) for x in benchmarks]))
        if metric not in INTERVAL_METRICS:
            out[metric] = value
            continue
        spread = np.mean(
            [reps[x][a][metric] - (0.0 if b is None else reps[x][b][metric]) for x in benchmarks],
            axis=0,
        )
        out[metric] = interval(value, spread).to_json()
    return out


def check_intervals(study: Mapping[str, Any], reps: Replicates) -> None:
    """The re-run bootstrap reproduces the study's per-benchmark intervals: its values, the
    calibrated-minus-raw and the minus-Jev differences (up to :data:`INTERVAL_AGREEMENT`)."""
    for benchmark, by_series in reps.items():
        for system in dict.fromkeys(s for s, _ in by_series):
            for key, arm in ARM.items():
                entry = study["series"][STUDY_SYSTEM[system]][arm][benchmark]
                pairs: list[tuple[str, Mapping[str, Any], Series | None]] = [("value", entry, None)]
                if key == CAL:
                    pairs.append(("delta_vs_raw", entry["delta_vs_raw"], (system, RAW)))
                if system != JEV:
                    pairs.append(("delta_vs_reference", entry["delta_vs_reference"], (JEV, key)))
                for what, recorded, base in pairs:
                    ours = combined(study, reps, [benchmark], (system, key), base)
                    for metric in INTERVAL_METRICS:
                        pairs_ = zip(ours[metric], recorded[metric], strict=True)
                        gap = max(abs(x - y) for x, y in pairs_)
                        if gap > INTERVAL_AGREEMENT[metric]:
                            fail(
                                f"the re-run bootstrap misses the study's {benchmark} {metric} "
                                f"{what} of {STUDY_SYSTEM[system]} ({arm}) by {gap:.2g}: "
                                f"{ours[metric]} against {recorded[metric]}"
                            )


def pool(parts: Sequence[UsageStats], latencies: FloatArray) -> UsageStats:
    """``parts`` (one benchmark's stats each) pooled as ``jevemu.bench.compare.stats`` pools a
    run's benchmarks: sums, ``latencies`` (ms, every answered item) for the percentiles, and the
    macro accuracy over the parts."""
    accuracies = [p.accuracy for p in parts if not math.isnan(p.accuracy)]
    cached = [p.cached_tokens for p in parts if p.cached_tokens is not None]
    calls = [p.backend_calls for p in parts if p.backend_calls is not None]
    costs = [p.cost_usd for p in parts]
    gpu = [p.gpu_seconds for p in parts if p.gpu_seconds is not None]
    walls = [p.wall_seconds for p in parts if p.wall_seconds is not None]
    p50, p95 = np.percentile(latencies, [50, 95]) if len(latencies) else (math.nan, math.nan)
    return UsageStats(
        n_items=sum(p.n_items for p in parts),
        n_ok=sum(p.n_ok for p in parts),
        n_correct=sum(p.n_correct for p in parts),
        accuracy=math.fsum(accuracies) / len(accuracies) if accuracies else math.nan,
        n_usage=sum(p.n_usage for p in parts),
        input_tokens=sum(p.input_tokens for p in parts),
        output_tokens=sum(p.output_tokens for p in parts),
        cached_tokens=sum(cached) if cached else None,
        n_cached_reported=sum(p.n_cached_reported for p in parts),
        backend_calls=sum(calls) if calls else None,
        n_diagnostics=sum(p.n_diagnostics for p in parts),
        latency_p50_ms=float(p50),
        latency_p95_ms=float(p95),
        cost_usd=None if any(c is None for c in costs) else math.fsum(c or 0.0 for c in costs),
        gpu_seconds=math.fsum(gpu) if gpu else None,
        spent_usd=math.fsum(p.spent_usd for p in parts),
        n_billed=sum(p.n_billed for p in parts),
        billed_table_usd=math.fsum(p.billed_table_usd for p in parts),
        n_price_mismatch=sum(p.n_price_mismatch for p in parts),
        n_response_cache_hits=sum(p.n_response_cache_hits for p in parts),
        throughput_items=sum(p.throughput_items for p in parts),
        wall_seconds=math.fsum(walls) if walls else None,
        concurrency=tuple(sorted({c for p in parts for c in p.concurrency})),
    )


def restricted(
    path: Path, prices: PriceBook, split: SplitName
) -> tuple[RunStats, dict[str, BenchmarkRun]]:
    """``stats`` of ``path`` on ``split`` restricted to :data:`BENCHMARKS` (their rows and a
    total pooled over them alone), with those benchmarks' loaded records."""
    full = require(path / "manifest.json").parent
    if not any(full.glob(f"*.{split}.jsonl")):
        fail(f"{path} has no {split} records")
    run = stats(full, prices, split=split)
    missing = [b for b in BENCHMARKS if b not in run.benchmarks]
    if missing:
        fail(f"{path} has no {split} records for {', '.join(missing)}")
    loaded = load_run(full, split=split)
    records = {b: loaded[b] for b in BENCHMARKS}
    latencies = np.array(
        [
            r.latency_ms
            for b in BENCHMARKS
            for r in records[b].ok_records
            if r.latency_ms is not None
        ],
        dtype=np.float64,
    )
    total = pool([run.benchmarks[b] for b in BENCHMARKS], latencies)
    return replace(run, benchmarks={b: run.benchmarks[b] for b in BENCHMARKS}, total=total), records


@dataclass(frozen=True)
class Costs:
    select: dict[str, RunStats]
    """System -> its cost source on :data:`COST_SPLIT`."""
    holdout: dict[str, RunStats]
    """System -> its study run on :data:`STUDY_SPLIT`, the cost cross-check."""
    in_flight: dict[str, int]
    """System -> items in flight of every timed invocation of both its runs."""


def check_one_call(path: Path, split: SplitName, records: Mapping[str, BenchmarkRun]) -> None:
    """Every record of ``records`` (one run's, on ``split``) was answered in exactly one model
    call (its diagnostics' ``n_backend_calls``)."""
    for benchmark, run in records.items():
        for item_id, record in run.records.items():
            calls = None if record.diagnostics is None else record.diagnostics.n_backend_calls
            if calls != 1:
                fail(f"{path} ({split}): {benchmark} item {item_id} took {calls} model calls")


def load_costs(c: Compared, study: Mapping[str, Any]) -> Costs:
    """The runs of the compared systems on both splits, restricted to :data:`BENCHMARKS`, with
    their consistency checks (module docstring)."""
    prices = PriceBook(gpu=GpuTimePrice.from_power(DEFAULT_GPU_WATTS, DEFAULT_USD_PER_WH))
    select, holdout = {}, {}
    for system in c.systems:
        select[system], select_records = restricted(RUN[system], prices, COST_SPLIT)
        holdout[system], holdout_records = restricted(RUN[system], prices, STUDY_SPLIT)
        if system != JEV:
            check_one_call(RUN[system], COST_SPLIT, select_records)
            check_one_call(RUN[system], STUDY_SPLIT, holdout_records)
    in_flight: dict[str, int] = {}
    for system in c.systems:
        runs = (select[system], holdout[system])
        levels = [s.total.concurrency for s in runs]
        expected = IN_FLIGHT.get(system)
        if (
            any(s.total.items_per_second is None for s in runs)
            or any(len(level) != 1 for level in levels)
            or levels[0] != levels[1]
            or (expected is not None and levels[0] != (expected,))
        ):
            wanted = f"{expected}" if expected is not None else "one tuned value"
            fail(
                f"{RUN[system]} ({COST_SPLIT} and {STUDY_SPLIT}): every timed invocation must "
                f"run {wanted} items in flight, got "
                + " and ".join(str(level or "no timed invocation") for level in levels)
            )
        in_flight[system] = levels[0][0]
        for s in runs:
            if (s.throughput_note is not None) != (system == JEV):
                fail(f"{s.run_dir}: unexpected throughput caveat {s.throughput_note!r}")
    for system in c.systems:
        answered = {b: u.n_items for b, u in select[system].benchmarks.items()}
        if answered != {b: u.n_items for b, u in select[JEV].benchmarks.items()}:
            fail(f"{RUN[system]} and {RUN[JEV]} ({COST_SPLIT}) differ in items")
        for b, u in holdout[system].benchmarks.items():
            if u.n_items != u.n_ok or u.n_items != study["n_items"][b]:
                fail(
                    f"{RUN[system]} ({STUDY_SPLIT}) answered {u.n_ok} of {u.n_items} {b} "
                    f"items; the study holds {study['n_items'][b]}"
                )
    for system in c.emulators:
        manifest = select[system].system
        if manifest.get("debiaser") is not None:
            fail(f"{RUN[system]} has debiaser {manifest['debiaser']!r}; it must have none")
        if not (manifest.get("launch_env") or {}).get(PRESET_ENV):
            fail(f"{RUN[system]}/manifest.json records no launch_env.{PRESET_ENV}")
    return Costs(select, holdout, in_flight)


# --- metrics.json ------------------------------------------------------------------------------


def quality(study: Mapping[str, Any], system: str, arm: str, benchmark: str) -> dict[str, Any]:
    entry = study["series"][system][arm][benchmark]
    return {
        "accuracy": entry["accuracy"],
        "nll": entry["nll"],
        "brier": entry["brier"],
        "ece": entry["ece"][0],
    }


def delta(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Intervals of accuracy, NLL and Brier; ECE as its point estimate."""
    return {
        "accuracy": entry["accuracy"],
        "nll": entry["nll"],
        "brier": entry["brier"],
        "ece": entry["ece"][0],
    }


def mean_top(items: Mapping[str, ItemMetrics], benchmark: str) -> float:
    if benchmark == MACRO:
        return math.fsum(float(m.confidence.mean()) for m in items.values()) / len(items)
    return float(items[benchmark].confidence.mean())


def bins_json(bins: Sequence[Any]) -> list[dict[str, float]]:
    return [
        {
            "n": b.n,
            "confidence": b.confidence,
            "accuracy": b.accuracy,
            "low": b.confidence_low,
            "high": b.confidence_high,
        }
        for b in bins
    ]


def pooled(items: Mapping[str, ItemMetrics], n_bins: int) -> dict[str, Any]:
    confidence = np.concatenate([m.confidence for m in items.values()])
    correct = np.concatenate([m.correct for m in items.values()])
    return {
        "n": int(confidence.shape[0]),
        "ece": ece(confidence, correct, n_bins=n_bins),
        "bins": bins_json(reliability(confidence, correct, n_bins=n_bins)),
    }


def shares(confidence: FloatArray) -> list[float]:
    """Share of ``confidence`` in each :data:`HIST_EDGES` bin (the last bin includes 1)."""
    counts, _ = np.histogram(confidence, bins=HIST_EDGES)
    return [float(c) / confidence.shape[0] for c in counts]


def usage(s: UsageStats) -> dict[str, Any]:
    return {
        "n_items": s.n_items,
        "n_correct": s.n_correct,
        "accuracy": s.accuracy,
        "input_tokens": s.input_tokens,
        "output_tokens": s.output_tokens,
        "total_tokens": s.input_tokens + s.output_tokens,
        "input_tokens_per_item": s.mean_input_tokens,
        "output_tokens_per_item": s.mean_output_tokens,
        "backend_calls": s.backend_calls,
        "calls_per_item": s.calls_per_item,
        "latency_p50_ms": s.latency_p50_ms,
        "latency_p95_ms": s.latency_p95_ms,
        "items_per_second": s.items_per_second,
        "throughput_items": s.throughput_items,
        "wall_seconds": s.wall_seconds,
        "in_flight": list(s.concurrency),
        "gpu_hours": s.gpu_hours,
        "cost_usd": s.cost_usd,
        "usd_per_1k_items": s.cost_per_1k_items,
        "usd_per_1k_correct": None if s.cost_per_correct is None else s.cost_per_correct * 1e3,
    }


def run_json(run: RunStats, path: Path) -> dict[str, Any]:
    return {
        "run_dir": str(path),
        "system_id": run.system_id,
        "split": run.split,
        "price": None if run.price is None else run.price.describe(),
        "total": usage(run.total),
        "benchmarks": {name: usage(s) for name, s in run.benchmarks.items()},
    }


def system_json(system: str, run: RunStats, in_flight: int) -> dict[str, Any]:
    """A system's manifest entry (the cost source's) plus what the report adds: its study name,
    items in flight and, for an emulator, its preset, the report's calibrator (the run itself
    has none) and its deployed registry (``None`` if not in the repository)."""
    out: dict[str, Any] = {k: v for k, v in run.system.items() if k != "confidence_fn"}
    out |= {"study_system": STUDY_SYSTEM[system], "in_flight": in_flight}
    if system == JEV:
        return out
    preset = run.system["launch_env"][PRESET_ENV]
    registry = Path("calibration") / preset / "registry.json"
    return out | {
        "preset": preset,
        "calibrator": ARM[CAL],
        "registry": str(registry) if (REPO / registry).is_file() else None,
    }


def rounding_check(c: Compared, predictions: Mapping[str, Predictions]) -> dict[str, Any]:
    """How much Jev's rounding (:data:`JEV_ROUND_TO`) inflates its raw NLL: its items whose
    correct answer got exactly 0 (clipped to ``NLL_EPS``), and the macro raw NLL of every system
    and of the paired differences with every p(gold) floored at :data:`ROUNDING_FLOOR` instead."""
    cap = -np.log(ROUNDING_FLOOR)
    macro = {
        s: float(
            np.mean([np.minimum(predictions[s].items[RAW][b].nll, cap).mean() for b in BENCHMARKS])
        )
        for s in c.systems
    }
    jev = predictions[JEV].items[RAW]
    return {
        "floor": ROUNDING_FLOOR,
        "jev_items_gold_zero": sum(int((jev[b].nll >= -np.log(NLL_EPS)).sum()) for b in BENCHMARKS),
        "n": sum(len(jev[b]) for b in BENCHMARKS),
        "macro_nll_raw_floored": macro,
        "deltas": {delta_key(a, b): macro[a] - macro[b] for a, b in c.deltas},
    }


def build_metrics(
    c: Compared,
    study: Mapping[str, Any],
    predictions: Mapping[str, Predictions],
    reps: Replicates,
    costs: Costs,
) -> dict[str, Any]:
    """Per benchmark the study's numbers (the Qwen 35B-A3B minus Qwen 27B intervals from the
    re-run bootstrap); the macro over :data:`BENCHMARKS` recomputed (:func:`combined`)."""
    n_bins = int(study["config"]["n_bins"])
    per_benchmark: dict[str, Any] = {}
    for benchmark in [*BENCHMARKS, MACRO]:
        macro = benchmark == MACRO
        over = list(BENCHMARKS) if macro else [benchmark]
        entry: dict[str, Any] = {"n": sum(study["n_items"][b] for b in over)}
        for system in c.systems:
            name = STUDY_SYSTEM[system]
            entry[system] = {}
            for key, arm in ARM.items():
                values = (
                    combined(study, reps, over, (system, key))
                    if macro
                    else quality(study, name, arm, benchmark)
                )
                values["mean_top_prob"] = mean_top(predictions[system].items[key], benchmark)
                values["top_prob_minus_accuracy"] = values["mean_top_prob"] - values["accuracy"][0]
                entry[system][key] = values
            entry[system]["calibrated_minus_raw"] = (
                combined(study, reps, over, (system, CAL), (system, RAW))
                if macro
                else delta(study["series"][name][ARM[CAL]][benchmark]["delta_vs_raw"])
            )
        for a, b in c.deltas:
            entry[delta_key(a, b)] = {}
            for key, arm in ARM.items():
                if b == JEV and not macro:
                    series = study["series"][STUDY_SYSTEM[a]][arm][benchmark]
                    values = delta(series["delta_vs_reference"])
                else:
                    values = combined(study, reps, over, (a, key), (b, key))
                values["mean_top_prob"] = (
                    entry[a][key]["mean_top_prob"] - entry[b][key]["mean_top_prob"]
                )
                entry[delta_key(a, b)][key] = values
        per_benchmark[benchmark] = entry

    reliability_bins: dict[str, Any] = {"pooled": {}, "benchmarks": {}}
    histogram: dict[str, Any] = {
        "definition": "share of items per equal-width top-probability bin",
        "edges": [float(e) for e in HIST_EDGES],
        "pooled": {},
        "benchmarks": {b: {} for b in BENCHMARKS},
    }
    for system in c.systems:
        name = STUDY_SYSTEM[system]
        items = predictions[system].items
        reliability_bins["pooled"][system] = {key: pooled(items[key], n_bins) for key in ARM}
        histogram["pooled"][system] = {
            key: shares(np.concatenate([m.confidence for m in items[key].values()])) for key in ARM
        }
        for benchmark in BENCHMARKS:
            reliability_bins["benchmarks"].setdefault(benchmark, {})[system] = {
                key: [
                    {k: b[k] for k in ("n", "confidence", "accuracy")}
                    for b in study["series"][name][arm][benchmark]["reliability"]
                ]
                for key, arm in ARM.items()
            }
            histogram["benchmarks"][benchmark][system] = {
                key: shares(items[key][benchmark].confidence) for key in ARM
            }
    reliability_bins["histogram"] = histogram

    select, holdout = costs.select, costs.holdout
    jev = select[JEV].total
    if jev.cost_usd is None:
        fail(f"{JEV_RUN} is unpriced")
    derived: dict[str, Any] = {}
    for system in c.emulators:
        emu = select[system].total
        if emu.gpu_hours is None or emu.cost_usd is None:
            fail(f"{RUN[system]} is unpriced")
        by_gpu = sorted(select[system].benchmarks.items(), key=lambda kv: -(kv[1].gpu_hours or 0.0))
        by_p95 = sorted(select[system].benchmarks.items(), key=lambda kv: -kv[1].latency_p95_ms)
        derived[system] = {
            "over_jev_usd_per_item": (emu.cost_usd / emu.n_items) / (jev.cost_usd / jev.n_items),
            "over_jev_usd_per_correct": (emu.cost_usd / emu.n_correct)
            / (jev.cost_usd / jev.n_correct),
            "breakeven_usd_per_gpu_hour_per_item": (jev.cost_usd / jev.n_items)
            / (emu.gpu_hours / emu.n_items),
            "breakeven_usd_per_gpu_hour_per_correct": (jev.cost_usd / jev.n_correct)
            / (emu.gpu_hours / emu.n_correct),
            "top_gpu_benchmarks": [name for name, _ in by_gpu[:2]],
            "top_gpu_hours": math.fsum(s.gpu_hours or 0.0 for _, s in by_gpu[:2]),
            "top_p95_benchmarks": [name for name, _ in by_p95[:2]],
        }
    rate = GpuTimePrice.from_power(DEFAULT_GPU_WATTS, DEFAULT_USD_PER_WH)
    crosscheck: dict[str, Any] = {}
    for system in c.systems:
        run = holdout[system]
        crosscheck[system] = {
            "run_dir": str(RUN[system]),
            "system_id": run.system_id,
            "total": usage(run.total),
        }
    return {
        "generated_by": "scripts/make_report.py",
        "excluded_benchmarks": list(EXCLUDED_BENCHMARKS),
        "inputs": {
            "quality": str(c.study_dir / "metrics.json"),
            "per_item_probabilities": [
                str(c.study_dir / f"predictions.{STUDY_SYSTEM[s]}.jsonl") for s in c.systems
            ],
            "cost": {**{s: str(RUN[s]) for s in c.systems}, "split": COST_SPLIT},
            "cost_crosscheck": {**{s: str(RUN[s]) for s in c.systems}, "split": STUDY_SPLIT},
        },
        "protocol": {"model_calls_per_emulator_item": 1, "labels": CALL_NOTE},
        "systems": {s: system_json(s, select[s], costs.in_flight[s]) for s in c.systems},
        "quality": {
            "split": STUDY_SPLIT,
            "arms": ARM,
            "intervals": "[estimate, 95% percentile low, high]; ECE and mean top probability are "
            "point estimates",
            "n_folds": study["config"]["n_folds"],
            "n_resamples": study["config"]["n_resamples"],
            "n_bins": n_bins,
            "macro": f"mean over the {len(BENCHMARKS)} benchmarks of the study's per-benchmark "
            "estimates; intervals from the mean of their bootstrap replicates, re-run on the "
            "study's predictions with its resamples",
            "deltas": {
                delta_key(a, b): "per benchmark the study's delta_vs_reference"
                if b == JEV
                else "the study's paired cluster bootstrap on its predictions and resamples"
                for a, b in c.deltas
            },
            "benchmarks": per_benchmark,
            "raw_nll_rounding": rounding_check(c, predictions),
        },
        "reliability": reliability_bins,
        "cost": {
            "split": COST_SPLIT,
            "gpu": GPU,
            "gpu_price": {
                "watts": DEFAULT_GPU_WATTS,
                "usd_per_wh": DEFAULT_USD_PER_WH,
                "usd_per_gpu_hour": rate.usd_per_gpu_hour,
                "basis": (
                    "electricity estimate (410 W observed, $0.27/kWh); no hardware amortization"
                ),
            },
            **{s: run_json(select[s], RUN[s]) for s in c.systems},
            "derived": derived,
            "throughput": {
                "definition": "items sent by the timed invocations (response-cache hits "
                "excluded) / their summed wall time (vLLM startup excluded)",
                "jev_client_max_requests_per_second": JEV_MAX_RPS,
                "jev_note": JEV_THROUGHPUT_NOTE,
            },
            "crosscheck": crosscheck,
        },
    }


# --- formatting --------------------------------------------------------------------------------


def num(value: float, digits: int, *, signed: bool = False) -> str:
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def ci(value: Sequence[float], digits: int, *, signed: bool = False) -> str:
    est, low, high = value
    return (
        f"{num(est, digits, signed=signed)} "
        f"[{num(low, digits, signed=signed)}, {num(high, digits, signed=signed)}]"
    )


def arrow(raw: float, cal: float, digits: int = 3) -> str:
    return f"{raw:.{digits}f} → {cal:.{digits}f}"


def count(value: float | None, digits: int = 1) -> str:
    return "" if value is None or math.isnan(value) else f"{value:,.{digits}f}"


def usd(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def slash(values: Sequence[str]) -> str:
    return " / ".join(values)


def qps(s: Mapping[str, Any], *, jev: bool = True) -> str:
    """Throughput cell; Jev's carries :data:`DAGGER` (its rate-limit footnote)."""
    value = s["items_per_second"]
    if value is None:
        return "—"
    return f"{value:.2f} {DAGGER}" if jev else f"{value:.2f}"


def jev_footnote(m: Mapping[str, Any]) -> str:
    return f"{DAGGER} {m['cost']['throughput']['jev_note']}"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def join(parts: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``; ``none`` when empty."""
    if len(parts) < 2:
        return parts[0] if parts else "none"
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def names(benchmarks: Sequence[str]) -> str:
    return join([label(b) for b in benchmarks])


def vs(a: str, b: str) -> str:
    return f"{NAME[a]} {MINUS} {NAME[b]}"


def row_label(benchmark: str) -> str:
    return "**macro**" if benchmark == MACRO else benchmark


def systems_of(m: Mapping[str, Any]) -> list[str]:
    """The compared systems, in table order."""
    return list(m["systems"])


def emulators_of(m: Mapping[str, Any]) -> list[str]:
    return [s for s in m["systems"] if s != JEV]


def deltas_of(m: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [(a, b) for a, b in DELTAS if a in m["systems"] and b in m["systems"]]


def in_flight(m: Mapping[str, Any]) -> str:
    """Each system's items in flight, e.g. ``Jev 16, Qwen 27B 16 and Qwen 35B-A3B 64``."""
    return join([f"{NAME[s]} {m['systems'][s]['in_flight']}" for s in systems_of(m)])


def percent(value: float) -> str:
    return f"{value:.0%}"


# --- README ------------------------------------------------------------------------------------


def headline_quality(m: Mapping[str, Any]) -> str:
    q = m["quality"]["benchmarks"][MACRO]
    rows = []
    for s in systems_of(m):
        for k in (RAW, CAL):
            v = q[s][k]
            rows.append(
                [
                    f"{NAME[s]}, {k}",
                    ci(v["accuracy"], 4),
                    ci(v["nll"], 3),
                    ci(v["brier"], 3),
                    num(v["ece"], 3),
                    num(v["mean_top_prob"], 3),
                    num(v["top_prob_minus_accuracy"], 3, signed=True),
                ]
            )
    for a, b in deltas_of(m):
        for k in (RAW, CAL):
            d = q[delta_key(a, b)][k]
            rows.append(
                [
                    f"Δ {vs(a, b)}, {k}",
                    ci(d["accuracy"], 4, signed=True),
                    ci(d["nll"], 3, signed=True),
                    ci(d["brier"], 3, signed=True),
                    num(d["ece"], 3, signed=True),
                    num(d["mean_top_prob"], 3, signed=True),
                    "",
                ]
            )
    header = [
        f"Macro, holdout (n = {q['n']:,})",
        "Accuracy [95% CI]",
        "NLL [95% CI]",
        "Brier [95% CI]",
        "ECE (10 equal-mass bins)",
        "Mean top probability",
        f"Mean top probability {MINUS} accuracy",
    ]
    return table(header, rows)


def headline_cost(m: Mapping[str, Any]) -> str:
    c = m["cost"]
    t = {s: c[s]["total"] for s in systems_of(m)}
    j = t[JEV]
    blank = [""] * len(emulators_of(m))

    def ratios(key: str) -> list[str]:
        return [
            f"{t[s][key] / j[key]:.2f}" if j[key] and t[s][key] is not None else ""
            for s in emulators_of(m)
        ]

    def each(key: str, fmt: Callable[[Any], str]) -> list[str]:
        return [fmt(t[s][key]) for s in systems_of(m)]

    rows = [
        [
            "Input tokens / item",
            *each("input_tokens_per_item", count),
            *ratios("input_tokens_per_item"),
        ],
        [
            "Output tokens / item",
            f"{count(j['output_tokens_per_item'])} (not billed)",
            *[count(t[s]["output_tokens_per_item"]) for s in emulators_of(m)],
            *ratios("output_tokens_per_item"),
        ],
        [
            "Model calls / item",
            "1 API request",
            *[count(t[s]["calls_per_item"], 2) for s in emulators_of(m)],
            *blank,
        ],
        [
            "Latency p50 (ms)",
            *each("latency_p50_ms", lambda v: count(v, 0)),
            *ratios("latency_p50_ms"),
        ],
        [
            "Latency p95 (ms)",
            *each("latency_p95_ms", lambda v: count(v, 0)),
            *ratios("latency_p95_ms"),
        ],
        ["Items in flight", *[str(m["systems"][s]["in_flight"]) for s in systems_of(m)], *blank],
        [
            "Throughput (q/s)",
            qps(j),
            *[qps(t[s], jev=False) for s in emulators_of(m)],
            *ratios("items_per_second"),
        ],
        ["Input tokens, total", *each("input_tokens", "{:,}".format), *ratios("input_tokens")],
        ["Output tokens, total", *each("output_tokens", "{:,}".format), *ratios("output_tokens")],
        ["Tokens, total", *each("total_tokens", "{:,}".format), *ratios("total_tokens")],
        ["GPU-hours", "", *[count(t[s]["gpu_hours"], 3) for s in emulators_of(m)], *blank],
        [
            "Cost, total ($)",
            f"{usd(j['cost_usd'])} (list price)",
            *[f"{usd(t[s]['cost_usd'])} (electricity estimate)" for s in emulators_of(m)],
            *ratios("cost_usd"),
        ],
        ["$ / 1k items", *each("usd_per_1k_items", usd), *ratios("usd_per_1k_items")],
        ["$ / 1k correct", *each("usd_per_1k_correct", usd), *ratios("usd_per_1k_correct")],
        ["Correct answers", *each("n_correct", "{:,}".format), *ratios("n_correct")],
    ]
    header = [
        f"{COST_SPLIT}, all benchmarks pooled (n = {j['n_items']:,})",
        *[NAME[s] for s in systems_of(m)],
        *[f"{NAME[s]} / Jev" for s in emulators_of(m)],
    ]
    return table(header, rows)


def accuracy_table(m: Mapping[str, Any]) -> str:
    rows = [
        [
            row_label(benchmark),
            f"{e['n']:,}",
            *[ci(e[s][RAW]["accuracy"], 4) for s in systems_of(m)],
            *[ci(e[delta_key(a, b)][RAW]["accuracy"], 4, signed=True) for a, b in deltas_of(m)],
        ]
        for benchmark, e in m["quality"]["benchmarks"].items()
    ]
    header = [
        "Benchmark",
        "n",
        *[NAME[s] for s in systems_of(m)],
        *[f"Δ {vs(a, b)}" for a, b in deltas_of(m)],
    ]
    return table(header, rows)


def calibrated_table(m: Mapping[str, Any], metric: str) -> str:
    """NLL or Brier per dataset: each system raw → calibrated, then the paired Δs, calibrated."""
    rows = [
        [
            row_label(benchmark),
            *[arrow(e[s][RAW][metric][0], e[s][CAL][metric][0]) for s in systems_of(m)],
            *[ci(e[delta_key(a, b)][CAL][metric], 3, signed=True) for a, b in deltas_of(m)],
        ]
        for benchmark, e in m["quality"]["benchmarks"].items()
    ]
    header = [
        "Benchmark",
        *[f"{NAME[s]}, raw → cal." for s in systems_of(m)],
        *[f"Δ {vs(a, b)}, cal." for a, b in deltas_of(m)],
    ]
    return table(header, rows)


def ece_table(m: Mapping[str, Any]) -> str:
    rows = [
        [
            row_label(benchmark),
            f"{e['n']:,}",
            *[arrow(e[s][RAW]["ece"], e[s][CAL]["ece"]) for s in systems_of(m)],
            *[
                slash([num(e[delta_key(a, b)][k]["ece"], 3, signed=True) for k in (RAW, CAL)])
                for a, b in deltas_of(m)
            ],
        ]
        for benchmark, e in m["quality"]["benchmarks"].items()
    ]
    header = [
        "Benchmark",
        "n",
        *[f"{NAME[s]}, raw → cal." for s in systems_of(m)],
        *[f"Δ {vs(a, b)}, raw / cal." for a, b in deltas_of(m)],
    ]
    return table(header, rows)


def confidence_table(m: Mapping[str, Any]) -> str:
    rows = [
        [
            row_label(benchmark),
            *[arrow(e[s][RAW]["mean_top_prob"], e[s][CAL]["mean_top_prob"]) for s in systems_of(m)],
        ]
        for benchmark, e in m["quality"]["benchmarks"].items()
    ]
    return table(["Benchmark", *[f"{NAME[s]}, raw → cal." for s in systems_of(m)]], rows)


def cost_rows(m: Mapping[str, Any]) -> list[tuple[str, dict[str, dict[str, Any]]]]:
    c = m["cost"]
    rows = [(b, {s: c[s]["benchmarks"][b] for s in systems_of(m)}) for b in c[JEV]["benchmarks"]]
    rows.append(("**all (pooled)**", {s: c[s]["total"] for s in systems_of(m)}))
    return rows


def usage_table(m: Mapping[str, Any]) -> str:
    rows = [
        [
            name,
            f"{u[JEV]['n_items']:,}",
            slash([count(u[s]["input_tokens_per_item"]) for s in systems_of(m)]),
            slash([count(u[s]["output_tokens_per_item"]) for s in systems_of(m)]),
            slash([count(u[s]["latency_p50_ms"], 0) for s in systems_of(m)]),
            slash([count(u[s]["latency_p95_ms"], 0) for s in systems_of(m)]),
            slash([qps(u[s], jev=s == JEV) for s in systems_of(m)]),
        ]
        for name, u in cost_rows(m)
    ]
    header = [
        "Benchmark",
        "Items",
        "In tok/item",
        "Out tok/item",
        "p50 ms",
        "p95 ms",
        "q/s",
    ]
    return table(header, rows)


def cost_table(m: Mapping[str, Any]) -> str:
    rows = [
        [
            name,
            f"{u[JEV]['n_items']:,}",
            slash([usd(u[s]["cost_usd"]) for s in systems_of(m)]),
            slash([count(u[s]["gpu_hours"], 3) for s in emulators_of(m)]),
            slash([usd(u[s]["usd_per_1k_items"]) for s in systems_of(m)]),
            slash([usd(u[s]["usd_per_1k_correct"]) for s in systems_of(m)]),
            *[
                f"{u[s]['usd_per_1k_items'] / u[JEV]['usd_per_1k_items']:.2f}"
                for s in emulators_of(m)
            ],
        ]
        for name, u in cost_rows(m)
    ]
    header = [
        "Benchmark",
        "Items",
        "Cost $",
        "GPU-h",
        "$ / 1k items",
        "$ / 1k correct",
        *[f"{NAME[s]} / Jev per item" for s in emulators_of(m)],
    ]
    return table(header, rows)


def crosscheck_table(m: Mapping[str, Any]) -> str:
    c = m["cost"]
    runs: list[tuple[str, Mapping[str, Any], bool]] = []
    for s in systems_of(m):
        runs.append((f"{NAME[s]}, `{COST_SPLIT}` (used)", c[s]["total"], s == JEV))
        runs.append((f"{NAME[s]}, `{STUDY_SPLIT}`", c["crosscheck"][s]["total"], s == JEV))
    rows = [
        [
            title,
            f"{s['n_items']:,}",
            "/".join(str(n) for n in s["in_flight"]),
            count(s["input_tokens_per_item"]),
            count(s["latency_p50_ms"], 0),
            count(s["latency_p95_ms"], 0),
            qps(s, jev=jev),
            count(s["gpu_hours"], 3),
            usd(s["cost_usd"]),
            usd(s["usd_per_1k_items"]),
        ]
        for title, s, jev in runs
    ]
    header = [
        "Run",
        "Items",
        "In flight",
        "In tok/item",
        "p50 ms",
        "p95 ms",
        "q/s",
        "GPU-h",
        "Cost $",
        "$ / 1k items",
    ]
    return table(header, rows)


def crosscheck_text(m: Mapping[str, Any]) -> str:
    return (
        f"This table puts the `{COST_SPLIT}` runs used above next to the `{STUDY_SPLIT}` runs of "
        "the same systems. Each system's two rows come from one run directory, so they measure "
        "one configuration on disjoint items."
    )


def accuracy_reading(qb: Mapping[str, Any], system: str) -> str:
    """One emulator's accuracy gap to Jev: macro, the large per-benchmark gaps, the rest."""
    benchmarks = [b for b in qb if b != MACRO]
    key = delta_key(system, JEV)
    d = qb[MACRO][key][RAW]["accuracy"]
    acc = {b: qb[b][key][RAW]["accuracy"] for b in benchmarks}
    large = sorted((b for b in benchmarks if acc[b][0] <= -0.05), key=lambda b: acc[b][0])
    rest = [b for b in benchmarks if b not in large]
    null = [b for b in benchmarks if acc[b][1] <= 0.0 <= acc[b][2]]
    text = f"{NAME[system]} {'trails' if d[0] < 0 else 'leads'} Jev: {ci(d, 4, signed=True)}. "
    if large:
        text += (
            "The largest gaps are "
            + join([f"{label(b)} ({num(acc[b][0], 3, signed=True)})" for b in large])
            + (f". On the other {len(rest)} benchmarks" if rest else ".")
        )
    else:
        text += f"On all {len(rest)} benchmarks"
    if rest:
        text += f" the gap is at most {max(abs(acc[b][0]) for b in rest):.3f}."
    if not null:
        return text + " Per benchmark, the 95% interval excludes zero everywhere."
    return text + f" Per benchmark, the 95% interval includes zero on {names(null)}."


def between_emulators(m: Mapping[str, Any]) -> dict[str, str]:
    """The Qwen 35B-A3B minus Qwen 27B sentences of :func:`reading` (empty without it)."""
    if FAST not in m["systems"]:
        return {"accuracy": "", "probability": "", "throughput": ""}
    qb = m["quality"]["benchmarks"]
    fq = qb[MACRO][delta_key(FAST, QWEN)]
    acc = {b: qb[b][delta_key(FAST, QWEN)][RAW]["accuracy"] for b in qb if b != MACRO}
    ahead = [b for b in acc if acc[b][1] > 0.0]
    behind = [b for b in acc if acc[b][2] < 0.0]
    excluded = [f"above zero on {names(ahead)}"] if ahead else []
    excluded += [f"below zero on {names(behind)}"] if behind else []
    t = {s: m["cost"][s]["total"] for s in (QWEN, FAST)}
    speedup = t[FAST]["items_per_second"] / t[QWEN]["items_per_second"]
    return {
        "accuracy": f"Between the emulators, {vs(FAST, QWEN)} is "
        f"{ci(fq[RAW]['accuracy'], 4, signed=True)}. Per benchmark, the 95% "
        + (
            "interval lies " + " and ".join(excluded) + " and includes zero elsewhere."
            if excluded
            else "interval includes zero everywhere."
        ),
        "probability": f"Between the emulators, {vs(FAST, QWEN)} is "
        f"{ci(fq[CAL]['nll'], 3, signed=True)} in NLL and "
        f"{ci(fq[CAL]['brier'], 3, signed=True)} in Brier.",
        "throughput": f"Pooled, {NAME[FAST]} is {speedup:.2f}{TIMES} as fast as {NAME[QWEN]}.",
    }


def reading(m: Mapping[str, Any]) -> list[str]:
    qb = m["quality"]["benchmarks"]
    q = qb[MACRO]
    benchmarks = [b for b in qb if b != MACRO]
    systems, emulators = systems_of(m), emulators_of(m)
    c = m["cost"]
    j = c[JEV]["total"]
    t = {s: c[s]["total"] for s in emulators}
    per = {s: c[s]["benchmarks"] for s in emulators}
    dv = c["derived"]
    rate = c["gpu_price"]["usd_per_gpu_hour"]
    between = between_emulators(m)
    unchanged = all(abs(q[s]["calibrated_minus_raw"]["accuracy"][0]) < 5e-5 for s in systems)
    gap = {
        s: {k: num(q[s][k]["top_prob_minus_accuracy"], 3, signed=True) for k in ARM}
        for s in systems
    }

    probability = []
    for s in emulators:
        d = q[delta_key(s, JEV)]
        nll = {b: qb[b][delta_key(s, JEV)][CAL]["nll"][0] for b in benchmarks}
        top = sorted(benchmarks, key=lambda b: -nll[b])[:3]
        probability.append(
            f"{vs(s, JEV)} is {ci(d[CAL]['nll'], 3, signed=True)} in NLL and "
            f"{ci(d[CAL]['brier'], 3, signed=True)} in Brier (raw: "
            f"{ci(d[RAW]['nll'], 3, signed=True)} and {ci(d[RAW]['brier'], 3, signed=True)}). "
            "The largest NLL gaps are "
            + join([f"{label(b)} ({num(nll[b], 3, signed=True)})" for b in top])
            + "."
        )
    r = m["quality"]["raw_nll_rounding"]
    probability.append(
        f"Raw NLL is biased against Jev. Jev rounds its probabilities to {JEV_ROUND_TO:g}, so on "
        f"{r['jev_items_gold_zero']:,} of {r['n']:,} items its probability of the correct answer "
        f"is exactly 0, clipped to 10<sup>{MINUS}{-round(math.log10(NLL_EPS))}</sup> "
        f"({-math.log(NLL_EPS):.1f} nats each). This inflates Jev's raw NLL, so the raw-NLL gaps "
        "above understate its lead. With every system's probability of the correct answer "
        f"floored at {r['floor']:g} (half Jev's rounding step), "
        + join(
            [
                f"{vs(s, JEV)} is {num(r['deltas'][delta_key(s, JEV)], 3, signed=True)}"
                for s in emulators
            ]
        )
        + ". Calibrated NLL is barely affected."
    )

    cost = []
    for s in emulators:
        cost.append(
            f"{NAME[s]} costs ${t[s]['usd_per_1k_items']:.4f} per 1k items "
            f"({percent(dv[s]['over_jev_usd_per_item'])} of Jev's) and "
            f"${t[s]['usd_per_1k_correct']:.4f} per 1k correct answers "
            f"({percent(dv[s]['over_jev_usd_per_correct'])}). It would match Jev's price per "
            f"item at ${dv[s]['breakeven_usd_per_gpu_hour_per_item']:.2f} per GPU-hour, and per "
            f"correct answer at ${dv[s]['breakeven_usd_per_gpu_hour_per_correct']:.2f} "
            f"({dv[s]['breakeven_usd_per_gpu_hour_per_item'] / rate:.1f}{TIMES} and "
            f"{dv[s]['breakeven_usd_per_gpu_hour_per_correct'] / rate:.1f}{TIMES} the "
            "electricity-estimate rate)."
        )

    gpu = []
    for s in emulators:
        top = dv[s]["top_gpu_benchmarks"]
        gpu.append(
            f"{names(top)} take {dv[s]['top_gpu_hours']:.3f} of {NAME[s]}'s "
            f"{t[s]['gpu_hours']:.3f} GPU-hours "
            f"({percent(dv[s]['top_gpu_hours'] / t[s]['gpu_hours'])}) for "
            f"{percent(sum(per[s][b]['n_items'] for b in top) / t[s]['n_items'])} of the items."
        )

    one_per_call = {s: t[s]["output_tokens"] == t[s]["backend_calls"] for s in emulators}
    latency = []
    for s in emulators:
        slow = dv[s]["top_p95_benchmarks"]
        others = max(per[s][b]["latency_p95_ms"] for b in per[s] if b not in slow)
        latency.append(
            f"{NAME[s]}: median {t[s]['latency_p50_ms']:,.0f} ms, p95 "
            f"{t[s]['latency_p95_ms']:,.0f} ms. Its slowest benchmarks are "
            + " and ".join(f"{label(b)} (p95 {per[s][b]['latency_p95_ms']:,.0f} ms)" for b in slow)
            + f"; every other benchmark's p95 is under {math.ceil(others / 100) / 10:.1f} s."
        )

    throughput = []
    for s in emulators:
        slowest = min(per[s], key=lambda b: per[s][b]["items_per_second"])
        fastest = max(per[s], key=lambda b: per[s][b]["items_per_second"])
        throughput.append(
            f"{NAME[s]} answers {t[s]['items_per_second']:.2f} q/s pooled with "
            f"{m['systems'][s]['in_flight']} items in flight: from "
            f"{per[s][slowest]['items_per_second']:.2f} ({label(slowest)}) to "
            f"{per[s][fastest]['items_per_second']:.2f} ({label(fastest)}) by dataset."
        )

    def sub(items: Sequence[str]) -> str:
        """``items`` as a nested list under the bullet (empty items skipped)."""
        return "".join(f"\n  - {x}" for x in items if x)

    return [
        "- Accuracy (macro, paired differences):"
        + sub([*(accuracy_reading(qb, s) for s in emulators), between["accuracy"]]),
        "- Raw macro ECE is "
        + join([f"{q[s][RAW]['ece']:.3f} ({NAME[s]})" for s in systems])
        + ". Temperature scaling per signature lowers it to "
        + join([f"{q[s][CAL]['ece']:.3f}" for s in systems])
        + " (point estimates)"
        + ("; accuracy is unchanged to four decimals. " if unchanged else ". ")
        + "Mean top probability minus accuracy goes "
        + join([f"from {gap[s][RAW]} to {gap[s][CAL]} for {NAME[s]}" for s in systems])
        + ".",
        "- Probability quality (calibrated unless marked raw; paired differences):"
        + sub([*probability, between["probability"]]),
        f"- Cost at the electricity estimate (${rate:.4f} per GPU-hour):"
        + sub(
            [
                f"Jev costs ${j['usd_per_1k_items']:.4f} per 1k items and "
                f"${j['usd_per_1k_correct']:.4f} per 1k correct answers.",
                *cost,
            ]
        ),
        "- GPU time by benchmark:" + sub(gpu),
        f"- Jev bills {j['input_tokens_per_item']:.1f} input tokens per item; "
        + join([f"{NAME[s]} reads {t[s]['input_tokens_per_item']:.1f}" for s in emulators])
        + " prompt tokens. Jev also reports "
        f"{j['output_tokens_per_item']:.1f} output tokens per item (not billed); "
        + ("the emulators generate " if len(emulators) > 1 else "the emulator generates ")
        + (
            join([f"{t[s]['output_tokens_per_item']:.1f}" for s in emulators])
            + " (one per model call)"
            if all(one_per_call.values())
            else join(
                [
                    f"{t[s]['output_tokens_per_item']:.1f}"
                    + (" (one per model call)" if one_per_call[s] else "")
                    for s in emulators
                ]
            )
        )
        + ".",
        "- Latency per item:"
        + sub(
            [
                f"Jev: median {j['latency_p50_ms']:,.0f} ms, p95 {j['latency_p95_ms']:,.0f} ms.",
                *latency,
            ]
        ),
        f"- Throughput, with the emulators on {GPU}:"
        + sub(
            [
                *throughput,
                between["throughput"],
                f"Jev's {j['items_per_second']:.2f} q/s "
                f"({m['systems'][JEV]['in_flight']} in flight) is our client's rate limit of "
                f"{c['throughput']['jev_client_max_requests_per_second']:g} requests/s (one "
                f"question per request); Jev's capacity was not measured {DAGGER}.",
            ]
        ),
    ]


def figure(name: str, alt: str) -> str:
    return f"![{alt}](figures/{name}.png)"


def emulator_setup(m: Mapping[str, Any], s: str) -> str:
    e = m["systems"][s]
    return (
        f"  - {NAME[s]}: `{e['backend_model']}` (revision `{e['model_revision'][:7]}`) on "
        f"vLLM {e['engine_version']}. Preset `{e['preset']}`, `{e['renderer']['layout']}` "
        f"prompt layout (`{e['renderer']['template_id']}`), strategy `{e['strategy']}`, no "
        f"debiaser, {e['in_flight']} items in flight, calibrated with `{e['calibrator']}`."
    )


FP16_SIZE = {QWEN: "27B: about 54 GB", FAST: "35B-A3B: about 70 GB"}
"""Each emulator model's 16-bit weights, against the GPU's 24 GB."""


def readme(m: Mapping[str, Any]) -> str:
    q = m["quality"]["benchmarks"]
    n = q[MACRO]["n"]
    n_benchmarks = len(q) - 1
    smallest = min((b for b in q if b != MACRO), key=lambda b: q[b]["n"])
    systems, emulators = systems_of(m), emulators_of(m)
    every = "both systems" if len(systems) == 2 else f"all {len(systems)} systems"
    jev_sys = m["systems"][JEV]
    c = m["cost"]
    gp = c["gpu_price"]
    cross = c["crosscheck"]
    registries = " ".join(
        f"{NAME[s]}'s deployed registry (`{e['registry']}`) is fitted on its whole holdout run."
        if e["registry"]
        else f"{NAME[s]} has no deployed registry (no `calibration/{e['preset']}/registry.json`)."
        for s, e in ((s, m["systems"][s]) for s in emulators)
    )
    between = (
        f" For {vs(FAST, QWEN)}, the same bootstrap is re-run on the study's per-item "
        "probabilities and resamples."
    )
    setup = [
        f"- Jev is TypeSafe's `{jev_sys['model']}` at `{jev_sys['base_url']}`, "
        f"with {jev_sys['in_flight']} items in flight. "
        + ("Both emulators run" if len(emulators) > 1 else "The emulator runs")
        + f" on {GPU}:",
        *[emulator_setup(m, s) for s in emulators],
        f"  - {CALL_NOTE}",
        f"- The `{STUDY_SPLIT}` half of {n_benchmarks} benchmarks has {n:,} items. "
        f"{every.capitalize()} answered them, and none of them were used to choose the "
        "configurations.",
        "- *Raw* means each system's returned probabilities, renormalized."
        + f" *Calibrated* means `{ARM[CAL]}`: one temperature per question signature, that is, "
        "per answer space (GPQA and LEXam share one, for example). It is cross-fitted in "
        f"{m['quality']['n_folds']} folds by question id, so every item is scored with a "
        "temperature fitted on the other four folds. Jev is calibrated the same way.",
        "- Intervals are 95% paired cluster-bootstrap intervals "
        f"({m['quality']['n_resamples']:,} resamples). For each emulator against Jev they are "
        f"the calibration study's.{between} *Macro* gives each benchmark equal weight.",
        f"- Cost, tokens and latency come from the `{COST_SPLIT}` half "
        f"({c[JEV]['total']['n_items']:,} items) for {every}, from the same run directories as "
        f"the `{STUDY_SPLIT}` answers. Calibration is a local rescaling and adds no calls.",
        "- Jev is priced at its list price, $0.042 per 1M input tokens; output is free. "
        + ("The emulators are" if len(emulators) > 1 else "The emulator is")
        + f" priced at GPU wall time {TIMES} {gp['watts']:g} W {TIMES} ${gp['usd_per_wh']:g}/Wh "
        f"= ${gp['usd_per_gpu_hour']:.4f} per GPU-hour. This electricity estimate uses the "
        "draw of about 410 W observed while running (not metered per run). It leaves out "
        "hardware amortization, the rest of the machine and cooling.",
    ]
    terms = [
        "Terms used below:",
        "",
        "- NLL (negative log-likelihood) is minus the log of the probability given to the "
        "correct answer.",
        "- The Brier score is the squared error between the probability vector and the correct "
        "answer (multiclass, 0 to 2).",
        "- ECE (expected calibration error) is the gap between top probability and accuracy, "
        "averaged over 10 equal-mass bins of top probability.",
        "- Lower is better for NLL, Brier and ECE. Mean top probability is the average "
        "probability of the top answer. Minus accuracy, it shows over-confidence (positive) or "
        "under-confidence (negative).",
        "- Δ is a paired difference: both systems on the same items.",
        "- q/s is questions answered per second. p50/p95 are the median and "
        "95th-percentile latency per item. Items in flight are the items sent at once.",
        '- Benchmarks ending in `_idk` add an "I don\'t know" option.',
    ]
    excluded = join([label(b) for b in m["excluded_benchmarks"]])
    exclusion = (
        f"{excluded} is excluded because its star-rating labels are ambiguous. Every table, "
        f"figure, macro and pooled number below covers the other {n_benchmarks} benchmarks. "
        "Per-benchmark numbers are the calibration study's. Macros are recomputed from them, "
        "with intervals from the study's bootstrap replicates of those benchmarks. "
        "docs/research/* keep the 10-benchmark numbers."
    )
    parts = [
        "# Jev vs the Qwen3.6 emulators (dense 27B and fast 35B-A3B)",
        "",
        "Generated by [`scripts/make_report.py`](../../scripts/make_report.py) from the recorded "
        "runs. Every number below is also in [`metrics.json`](metrics.json). Method and full "
        "tables: [calibration study](../../docs/research/calibration.md), "
        "[configuration selection](../../docs/research/selection.md).",
        "",
        *setup,
        "",
        *terms,
        "",
        exclusion,
        "",
        "## Summary",
        "",
        f"Quality on `{STUDY_SPLIT}`, macro over the {n_benchmarks} benchmarks. ECE and mean top "
        "probability are point estimates; Δ rows are paired differences.",
        "",
        headline_quality(m),
        "",
        f"Cost, tokens, latency and throughput on `{COST_SPLIT}`, all items of the "
        f"{n_benchmarks} benchmarks pooled. $ / 1k correct divides by the correct answers on "
        "that half. Calibration does not change these numbers.",
        "",
        headline_cost(m),
        "",
        jev_footnote(m),
        "",
        figure("accuracy_by_dataset", "Accuracy by dataset"),
        "",
        figure("accuracy_gap_by_dataset", "Accuracy gap by dataset"),
        "",
        "## Results",
        "",
        *reading(m),
        "",
        "## Results by dataset",
        "",
        f"### Accuracy (`{STUDY_SPLIT}`)",
        "",
        "Temperature scaling keeps the order of the options, so accuracy is the same raw and "
        "calibrated (to four decimals). The table shows raw. Δ columns are paired.",
        "",
        accuracy_table(m),
        "",
        f"### NLL, Brier and ECE (`{STUDY_SPLIT}`)",
        "",
        "Each system's cells show the raw value, then the calibrated one. The Δ columns are "
        "paired differences of the calibrated values.",
        "",
        "NLL:",
        "",
        calibrated_table(m, "nll"),
        "",
        "Brier:",
        "",
        calibrated_table(m, "brier"),
        "",
        "ECE (top label, 10 equal-mass bins, point estimates). GPQA has only 99 items, so each "
        "bin holds about 10 and its ECE is noisy (noise floor about 0.11):",
        "",
        ece_table(m),
        "",
        f"### Reliability (`{STUDY_SPLIT}`)",
        "",
        "Each diagram plots accuracy against mean top probability in 10 equal-mass bins, raw "
        "and calibrated. The diagonal is perfect calibration. Marker area is proportional to a "
        "bin's item count. The strip under each diagram is a histogram of top probability "
        "(0.05-wide bins, share of items).",
        "",
        f"The overall figure pools all {n:,} items into one set of bins, so its pooled ECE "
        "differs from the tables' macro ECE (the mean of the per-benchmark ECEs). The "
        "per-dataset figures show the bins behind each benchmark's ECE. "
        f"{label(smallest)} has {q[smallest]['n']:,} `{STUDY_SPLIT}` items, so its bins hold "
        f"about {q[smallest]['n'] / 10:.0f} items each.",
        "",
        figure("reliability_overall", "Reliability diagrams, pooled"),
        "",
        figure("reliability_by_dataset_raw", "Reliability diagrams by dataset, raw"),
        "",
        figure("reliability_by_dataset_calibrated", "Reliability diagrams by dataset, calibrated"),
        "",
        f"### Confidence (`{STUDY_SPLIT}`)",
        "",
        "Mean top probability, raw and then calibrated:",
        "",
        confidence_table(m),
        "",
        "The same NLL and ECE numbers as bar charts:",
        "",
        figure("nll_by_dataset", "NLL by dataset"),
        "",
        figure("ece_by_dataset", "ECE raw vs calibrated by dataset"),
        "",
        f"### Tokens, latency and throughput (`{COST_SPLIT}`)",
        "",
        f"Cells are {slash([NAME[s] for s in systems])}.",
        "",
        "- An emulator's tokens are the prompt and generated tokens of its one model call per "
        "item. Jev's are the tokens it reports; input is billed and output is not.",
        f"- Latency is per item, at each system's items in flight ({in_flight(m)}).",
        "- Throughput (q/s) is the items a run answered divided by its summed wall time "
        "(vLLM server startup excluded).",
        "",
        usage_table(m),
        "",
        jev_footnote(m),
        "",
        figure("tokens_by_dataset", "Tokens per item by dataset"),
        "",
        figure("latency_by_dataset", "Latency by dataset"),
        "",
        figure("throughput_by_dataset", "Throughput by dataset"),
        "",
        f"### Cost (`{COST_SPLIT}`)",
        "",
        f"Cells are {slash([NAME[s] for s in systems])}; GPU-h is "
        f"{slash([NAME[s] for s in emulators])}. An emulator's GPU time for a benchmark is the "
        "wall time of the run that answered it, priced at the electricity estimate.",
        "",
        cost_table(m),
        "",
        figure("cost_by_dataset", "Cost by dataset"),
        "",
        figure("accuracy_vs_cost", "Accuracy vs cost by dataset"),
        "",
        f"### Cost check against the `{STUDY_SPLIT}` runs",
        "",
        crosscheck_text(m),
        "",
        crosscheck_table(m),
        "",
        jev_footnote(m),
        "",
        "## Caveats",
        "",
        f"- The electricity price is a placeholder. The {gp['watts']:g} W average draw is an "
        f"assumption and was not measured; ${gp['usd_per_wh']:g}/Wh is the New Jersey average. "
        "The estimate leaves out the GPU's purchase price, the rest of the machine and cooling. "
        "With those, a local emulator may cost more than Jev. The break-even rates above show "
        "how much headroom there is.",
        "- The latency numbers are not directly comparable. An emulator's latency is the "
        f"end-to-end time of one item on {GPU}, sharing the server with the other items in "
        "flight. Jev's is the HTTP round trip over the internet from the same client. Items in "
        f"flight: {in_flight(m)}. Each emulator's GPU time per item was measured at this "
        "concurrency only.",
        "- "
        + ("Neither Qwen model fits" if len(emulators) > 1 else "Qwen does not fit")
        + " in the GPU's 24 GB with 16-bit weights ("
        + "; ".join(FP16_SIZE[s] for s in emulators)
        + "), so "
        + ("both run quantized builds: " if len(emulators) > 1 else "it runs a quantized build: ")
        + join([f"`{m['systems'][s]['backend_model']}` ({NAME[s]})" for s in emulators])
        + ". Another 4-bit build of the 27B model (cyankiwi) is statistically tied with "
        "QuantTrio's; see the calibration study.",
        f"- Quality is measured on `{STUDY_SPLIT}` and cost on `{COST_SPLIT}`. The two halves "
        "share the benchmarks and prompt template but have disjoint items. Jev's cost per 1k "
        f"items is ${c[JEV]['total']['usd_per_1k_items']:.4f} on `{COST_SPLIT}` and "
        f"${cross[JEV]['total']['usd_per_1k_items']:.4f} on `{STUDY_SPLIT}`. $ / 1k correct "
        f"uses the answers on `{COST_SPLIT}` (macro accuracy "
        + join([f"{c[s]['total']['accuracy']:.4f} {NAME[s]}" for s in systems])
        + ", scoring each system's returned answer).",
        "- The calibrated numbers are out of fold: each fold's temperatures are fitted on the "
        f"other four folds of this holdout half. {registries} New question types fall back to "
        "one global temperature.",
        "- Accuracy uses the first argmax of the renormalized probabilities (the first "
        "option with the highest probability), for every system. `run_split.py summarize` "
        "scores Jev's returned `choice` instead, which differs on near-ties (macro "
        f"{cross[JEV]['total']['accuracy']:.4f} vs {q[MACRO][JEV][RAW]['accuracy'][0]:.4f} on "
        f"`{STUDY_SPLIT}`).",
        "- The $ / 1k items and $ / 1k correct rows pool all items, while macro accuracy weighs "
        "every benchmark equally.",
        "",
        "## Regenerate",
        "",
        "```bash",
        "uv run --extra plot python scripts/make_report.py",
        "```",
        "",
        "The inputs are the gitignored `runs/` artifacts listed in the script's docstring. The "
        "report holds only aggregates, with no question text or per-item outputs.",
        "",
    ]
    return "\n".join(parts)


# --- figures -----------------------------------------------------------------------------------

POOLED = "All (pooled)"
MARKER = {JEV: "o", QWEN: "s", FAST: "D"}


def system_palette(m: Mapping[str, Any]) -> dict[str, str]:
    return {NAME[s]: COLOR[s] for s in systems_of(m)}


def series_palette(m: Mapping[str, Any]) -> dict[str, str]:
    """Hue levels of the raw-vs-calibrated plots: light = raw, dark = calibrated."""
    return {
        f"{NAME[s]} {arm}": (LIGHT if arm == RAW else COLOR)[s]
        for s in systems_of(m)
        for arm in (RAW, CAL)
    }


def in_flight_title(m: Mapping[str, Any]) -> str:
    """``items in flight: Jev 16, Qwen 27B 16, Qwen 35B-A3B 64`` for figure titles."""
    return "items in flight: " + ", ".join(
        f"{NAME[s]} {m['systems'][s]['in_flight']}" for s in systems_of(m)
    )


def quality_frame(m: Mapping[str, Any]) -> pd.DataFrame:
    """One row per dataset (macro last), system, arm and metric; ``low``/``high`` are NaN for
    point estimates."""
    rows = []
    for benchmark, entry in m["quality"]["benchmarks"].items():
        for s in systems_of(m):
            for arm in (RAW, CAL):
                for metric, v in entry[s][arm].items():
                    est, low, high = v if isinstance(v, list) else (v, math.nan, math.nan)
                    rows.append(
                        {
                            "dataset": label(benchmark),
                            "system": NAME[s],
                            "series": f"{NAME[s]} {arm}",
                            "arm": arm,
                            "metric": metric,
                            "value": est,
                            "low": low,
                            "high": high,
                        }
                    )
    return pd.DataFrame(rows)


def cost_frame(m: Mapping[str, Any]) -> pd.DataFrame:
    """One row per dataset (pooled last) and system, with every :func:`usage` column."""
    c = m["cost"]
    rows = []
    for s in systems_of(m):
        rows += [
            {"dataset": label(b), "system": NAME[s], **u} for b, u in c[s]["benchmarks"].items()
        ]
        rows.append({"dataset": POOLED, "system": NAME[s], **c[s]["total"]})
    return pd.DataFrame(rows)


def bars(
    ax: Axes,
    data: pd.DataFrame,
    y: str,
    hue: str,
    palette: Mapping[str, str],
    *,
    ncols: int,
    errors: bool = False,
    log: bool = False,
    legend: bool = True,
) -> None:
    """Grouped bars per dataset from precomputed values (``low``/``high`` error bars), the last
    group (macro or pooled) set off by a dotted line; the legend (upper left, ``ncols``
    columns) is left out with ``legend=False``. A log axis is set after drawing: seaborn's own
    ``log_scale`` leaves the bars undrawn here."""
    order = list(dict.fromkeys(data["dataset"]))
    levels = list(palette)
    sns.barplot(
        data=data,
        x="dataset",
        y=y,
        hue=hue,
        order=order,
        hue_order=levels,
        palette=palette,
        errorbar=None,
        saturation=1.0,
        ax=ax,
    )
    if log:
        ax.set_yscale("log")
        log_axis(ax.yaxis)
    if errors:
        for container, level in zip(list(ax.containers), levels, strict=True):
            sub = data[data[hue] == level].set_index("dataset").loc[order]
            ax.errorbar(
                [bar.get_x() + bar.get_width() / 2 for bar in container],
                sub[y],
                yerr=[sub[y] - sub["low"], sub["high"] - sub[y]],
                fmt="none",
                ecolor="#333333",
                elinewidth=0.8,
                capsize=1.5,
            )
    ax.axvline(len(order) - 1.5, color="#888888", linewidth=0.8, linestyle=":")
    ax.set_xlabel("")
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_horizontalalignment("right")
        tick.set_rotation_mode("anchor")
    if legend:
        sns.move_legend(ax, "upper left", ncols=ncols, title=None, frameon=False)
    elif (drawn := ax.get_legend()) is not None:
        drawn.remove()


def log_axis(axis: Any) -> None:
    """Log ticks at 1, 2 and 5 per decade, no minor labels (``axis`` is an x or y axis)."""
    axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    axis.set_minor_formatter(NullFormatter())


def save(fig: Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=DPI, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def fig_accuracy(qf: pd.DataFrame, m: Mapping[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2), layout="constrained")
    data = qf[(qf["metric"] == "accuracy") & (qf["arm"] == RAW)]
    palette = system_palette(m)
    bars(ax, data, "value", "system", palette, ncols=len(palette), errors=True)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_ylabel("Accuracy (95% CI)")
    n = m["quality"]["benchmarks"][MACRO]["n"]
    ax.set_title(f"Accuracy by dataset, {STUDY_SPLIT} (n = {n:,})")
    save(fig, out, "accuracy_by_dataset")


def fig_accuracy_gap(m: Mapping[str, Any], out: Path) -> None:
    """Each emulator's paired accuracy gap to Jev per dataset; a hollow marker's interval
    includes zero."""
    emulators = emulators_of(m)
    offset = {s: 0.36 * (i - (len(emulators) - 1) / 2) for i, s in enumerate(emulators)}
    rows = []
    for pos, (benchmark, entry) in enumerate(m["quality"]["benchmarks"].items()):
        for s in emulators:
            est, low, high = entry[delta_key(s, JEV)][RAW]["accuracy"]
            rows.append(
                {
                    "pos": pos + offset[s],
                    "dataset": label(benchmark),
                    "system": s,
                    "delta": est,
                    "low": low,
                    "high": high,
                    "excludes": high < 0.0 or low > 0.0,
                }
            )
    df = pd.DataFrame(rows)
    n_rows = len(m["quality"]["benchmarks"])
    fig, ax = plt.subplots(figsize=(7.5, 6.2), layout="constrained")
    for s in emulators:
        sub = df[df["system"] == s]
        ax.hlines(sub["pos"], sub["low"], sub["high"], colors=COLOR[s], linewidth=1.6)
        ax.scatter(
            sub["delta"],
            sub["pos"],
            s=30,
            c=[COLOR[s] if e else "white" for e in sub["excludes"]],
            edgecolors=COLOR[s],
            linewidths=1.2,
            zorder=3,
        )
    for row in df.itertuples():
        ax.annotate(
            f"{row.delta:+.3f}".replace("-", MINUS),
            (max(row.high, 0.0), row.pos),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color="#333333",
        )
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.axhline(n_rows - 1.5, color="#888888", linewidth=0.8, linestyle=":")
    ax.set_yticks(range(n_rows), list(dict.fromkeys(df["dataset"])))
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=max(0.08, float(df["high"].max()) + 0.06))
    ax.set_ylabel("")
    ax.set_xlabel(f"Δ accuracy, emulator {MINUS} Jev (raw, paired, 95% CI)")
    ax.set_title("Accuracy gap to Jev by dataset")
    marker: dict[str, Any] = {"marker": "o", "markersize": 5, "markeredgewidth": 1.2}
    handles = [Line2D([], [], color=COLOR[s], label=NAME[s], **marker) for s in emulators] + [
        Line2D([], [], color="#777777", linestyle="", label="95% interval excludes 0", **marker),
        Line2D(
            [],
            [],
            color="#777777",
            linestyle="",
            markerfacecolor="white",
            label="95% interval includes 0",
            **marker,
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=4, frameon=False, fontsize=7.5)
    save(fig, out, "accuracy_gap_by_dataset")


def fig_raw_vs_calibrated(
    qf: pd.DataFrame, m: Mapping[str, Any], out: Path, metric: str, ylabel: str
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.4), layout="constrained")
    data = qf[qf["metric"] == metric]
    errors = bool(data["low"].notna().all())
    bars(ax, data, "value", "series", series_palette(m), ncols=len(m["systems"]), errors=errors)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric.upper()}, raw vs calibrated ({ARM[CAL]}), {STUDY_SPLIT}")
    save(fig, out, f"{metric}_by_dataset")


def reliability_panel(
    fig: Figure,
    spec: Any,
    bins: Mapping[str, Sequence[Mapping[str, float]]],
    hist: Mapping[str, Sequence[float]],
    edges: Sequence[float],
    *,
    title: str,
    ylabels: bool,
    fontsize: float,
    marker_area: float,
) -> tuple[Axes, Axes]:
    """A reliability diagram over a confidence-histogram strip in ``spec`` (a gridspec cell):
    per system (the keys of ``bins``), the bins' accuracy against mean top probability with
    marker area ``marker_area x items / most items in any bin of the panel`` (proportional to
    the count), and the share of items per equal-width confidence bin, as a step line."""
    grid = spec.subgridspec(2, 1, height_ratios=(4, 1), hspace=0.05)
    main = fig.add_subplot(grid[0])
    strip = fig.add_subplot(grid[1], sharex=main)
    main.plot([0, 1], [0, 1], color="#999999", linewidth=0.8, linestyle="--", zorder=0)
    largest = max(b["n"] for s in bins for b in bins[s])
    for s in bins:
        x = [b["confidence"] for b in bins[s]]
        y = [b["accuracy"] for b in bins[s]]
        # unclipped: bins at confidence or accuracy 1 sit on the frame
        main.plot(x, y, color=COLOR[s], linewidth=1.2, zorder=2, clip_on=False)
        main.scatter(
            x,
            y,
            s=[marker_area * b["n"] / largest for b in bins[s]],
            color=COLOR[s],
            edgecolors="white",
            linewidths=0.6,
            alpha=0.85,
            zorder=3,
            clip_on=False,
        )
        strip.stairs(hist[s], edges, color=COLOR[s], linewidth=1.0)
    main.set_xlim(0, 1)
    main.set_ylim(0, 1)
    main.set_title(title, fontsize=fontsize + 1)
    main.tick_params(labelbottom=False, labelsize=fontsize)
    strip.tick_params(labelsize=fontsize)
    strip.set_ylim(bottom=0)
    strip.grid(axis="y", visible=False)
    strip.set_xlabel("Mean top probability", fontsize=fontsize)
    if ylabels:
        main.set_ylabel("Accuracy in bin", fontsize=fontsize)
        strip.set_ylabel("Share", fontsize=fontsize)
    return main, strip


def reliability_legend(fig: Figure, title: str, systems: Sequence[str]) -> None:
    """One legend of the systems above the panels, headed by the figure's title (a separate
    suptitle would collide with an outside legend under the constrained layout)."""
    fig.legend(
        handles=[
            Line2D([], [], color=COLOR[s], marker="o", markersize=5, label=NAME[s]) for s in systems
        ],
        loc="outside upper center",
        ncols=len(systems),
        frameon=False,
        title=title,
        title_fontproperties={"weight": "bold", "size": 11},
    )


def fig_reliability_overall(m: Mapping[str, Any], out: Path) -> None:
    rel = m["reliability"]
    titles = {RAW: "Raw", CAL: f"Calibrated ({ARM[CAL]})"}
    fig = plt.figure(figsize=(10, 5.6), layout="constrained")
    grid = fig.add_gridspec(1, len(titles))
    for col, (arm, title) in enumerate(titles.items()):
        main, _ = reliability_panel(
            fig,
            grid[0, col],
            {s: rel["pooled"][s][arm]["bins"] for s in systems_of(m)},
            {s: rel["histogram"]["pooled"][s][arm] for s in systems_of(m)},
            rel["histogram"]["edges"],
            title=title,
            ylabels=col == 0,
            fontsize=8.5,
            marker_area=60.0,
        )
        text = "\n".join(f"{NAME[s]} {rel['pooled'][s][arm]['ece']:.3f}" for s in systems_of(m))
        main.text(
            0.97,
            0.04,
            f"pooled ECE\n{text}",
            ha="right",
            va="bottom",
            fontsize=8,
            transform=main.transAxes,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#cccccc"},
        )
    reliability_legend(
        fig,
        f"Reliability, {STUDY_SPLIT}: all {rel['pooled'][JEV][RAW]['n']:,} items pooled, "
        "10 equal-mass bins (marker area ∝ items); strip: top-probability histogram",
        systems_of(m),
    )
    save(fig, out, "reliability_overall")


def fig_reliability_grid(m: Mapping[str, Any], out: Path, arm: str) -> None:
    q = m["quality"]["benchmarks"]
    rel = m["reliability"]
    benchmarks = list(rel["benchmarks"])
    n_cols = 5
    n_rows = math.ceil(len(benchmarks) / n_cols)
    fig = plt.figure(figsize=(14, 3.6 * n_rows + 0.6), layout="constrained")
    grid = fig.add_gridspec(n_rows, n_cols)
    for index, b in enumerate(benchmarks):
        reliability_panel(
            fig,
            grid[index // n_cols, index % n_cols],
            {s: rel["benchmarks"][b][s][arm] for s in systems_of(m)},
            {s: rel["histogram"]["benchmarks"][b][s][arm] for s in systems_of(m)},
            rel["histogram"]["edges"],
            title=f"{label(b)} (n = {q[b]['n']:,})",
            ylabels=index % n_cols == 0,
            fontsize=7.5,
            marker_area=36.0,
        )
    kind = "raw" if arm == RAW else f"calibrated ({ARM[CAL]})"
    reliability_legend(
        fig,
        f"Reliability by dataset, {STUDY_SPLIT}, {kind}: 10 equal-mass bins per benchmark "
        "(marker area ∝ items); strip: top-probability histogram",
        systems_of(m),
    )
    save(fig, out, f"reliability_by_dataset_{arm}")


def two_panels(
    cf: pd.DataFrame,
    m: Mapping[str, Any],
    out: Path,
    name: str,
    title: str,
    panels: Sequence[tuple[str, str, bool]],
    fmt: Callable[[float, int], str],
) -> None:
    """One bar per system and dataset, one panel per (column, y label, log scale), one legend
    below the panels."""
    palette = system_palette(m)
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 4.6), layout="constrained")
    for ax, (column, ylabel, log) in zip(axes, panels, strict=True):
        bars(ax, cf, column, "system", palette, ncols=len(palette), log=log, legend=False)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(fmt)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(palette), frameon=False)
    fig.suptitle(title, fontweight="bold")
    save(fig, out, name)


def fig_throughput(cf: pd.DataFrame, m: Mapping[str, Any], out: Path) -> None:
    limit = m["cost"]["throughput"]["jev_client_max_requests_per_second"]
    fig, ax = plt.subplots(figsize=(11, 4.2), layout="constrained")
    palette = system_palette(m)
    bars(ax, cf, "items_per_second", "system", palette, ncols=len(palette), log=True, legend=False)
    # headroom for the legend above the tallest bars
    ax.set_ylim(top=3.0 * max(limit, float(cf["items_per_second"].max())))
    ax.axhline(limit, color=COLOR[JEV], linewidth=0.9, linestyle="--", zorder=0)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=COLOR[JEV], linewidth=0.9, linestyle="--"))
    labels.append(f"Jev client limit, {limit:g} requests/s (1 question per request)")
    ax.legend(handles, labels, loc="upper left", ncols=len(labels), frameon=False)
    ax.set_ylabel("Items / s (log)")
    ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:g}")
    ax.set_title(
        f"Throughput by dataset, {COST_SPLIT}, {in_flight_title(m)} "
        "(Jev capped by our client's rate limit)"
    )
    save(fig, out, "throughput_by_dataset")


def fig_accuracy_vs_cost(m: Mapping[str, Any], out: Path) -> None:
    c, q = m["cost"], m["quality"]["benchmarks"]
    df = pd.DataFrame(
        [
            {
                "dataset": label(b),
                "system": NAME[s],
                "cost": c[s]["benchmarks"][b]["usd_per_1k_items"],
                "accuracy": q[b][s][RAW]["accuracy"][0],
            }
            for b in c[JEV]["benchmarks"]
            for s in systems_of(m)
        ]
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.6), layout="constrained")
    for _, sub in df.groupby("dataset", sort=False):
        ax.plot(sub["cost"], sub["accuracy"], color="#bbbbbb", linewidth=0.9, zorder=1)
    sns.scatterplot(
        data=df,
        x="cost",
        y="accuracy",
        hue="system",
        style="system",
        palette=system_palette(m),
        markers={NAME[s]: MARKER[s] for s in systems_of(m)},
        s=40,
        zorder=2,
        ax=ax,
    )
    ax.set_xscale("log")
    log_axis(ax.xaxis)
    ax.xaxis.set_major_formatter(lambda v, _pos: f"${v:g}")
    ax.set_xlabel(f"$ per 1k items ({COST_SPLIT}; emulators at the electricity estimate)")
    ax.set_ylabel(f"Accuracy ({STUDY_SPLIT})")
    ax.set_title("Accuracy vs cost by dataset (grey lines join a dataset's systems)")
    sns.move_legend(ax, "lower right", title=None, frameon=False)
    label_jev_points(fig, ax, df[df["system"] == NAME[JEV]])
    save(fig, out, "accuracy_vs_cost")


LABEL_OFFSETS: tuple[tuple[float, float, str, str], ...] = (
    (5, 3, "left", "bottom"),
    (5, -3, "left", "top"),
    (-5, 3, "right", "bottom"),
    (-5, -3, "right", "top"),
    (0, 6, "center", "bottom"),
    (0, -6, "center", "top"),
)
"""Dataset label positions around Jev's point (offset in points, alignment), preferred first."""


def label_jev_points(fig: Figure, ax: Axes, jev: Any) -> None:
    """Name each dataset at Jev's point, at the first of :data:`LABEL_OFFSETS` whose text
    overlaps no marker (of any system) and no label placed before, else the least overlap."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    radius = 4.0 * fig.dpi / 72
    xy = ax.transData.transform(ax.collections[0].get_offsets())
    markers = [Bbox.from_extents(x - radius, y - radius, x + radius, y + radius) for x, y in xy]
    placed: list[Bbox] = []
    for row in jev.itertuples():
        best: tuple[float, Any, Bbox] | None = None
        for dx, dy, ha, va in LABEL_OFFSETS:
            note = ax.annotate(
                str(row.dataset),
                (row.cost, row.accuracy),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=7.5,
            )
            box = note.get_window_extent(renderer)
            cost = sum(box_overlap(box, other) for other in [*markers, *placed])
            if best is None or cost < best[0]:
                if best is not None:
                    best[1].remove()
                best = (cost, note, box)
                if cost == 0.0:
                    break
            else:
                note.remove()
        assert best is not None
        placed.append(best[2])


def box_overlap(a: Bbox, b: Bbox) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if w > 0 and h > 0 else 0.0


def write_figures(m: Mapping[str, Any], out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    sns.set_theme(context="paper", style="whitegrid", font="DejaVu Sans", rc=STYLE)
    qf, cf = quality_frame(m), cost_frame(m)
    fig_accuracy(qf, m, out)
    fig_accuracy_gap(m, out)
    fig_raw_vs_calibrated(qf, m, out, "nll", "NLL (nats, 95% CI)")
    fig_raw_vs_calibrated(qf, m, out, "ece", "ECE (top label, 10 equal-mass bins)")
    fig_reliability_overall(m, out)
    for arm in ARM:
        fig_reliability_grid(m, out, arm)
    two_panels(
        cf,
        m,
        out,
        "tokens_by_dataset",
        f"Tokens per item by dataset, {COST_SPLIT}",
        [
            ("input_tokens_per_item", "Input tokens / item", False),
            ("output_tokens_per_item", "Output tokens / item (log; Jev's not billed)", True),
        ],
        lambda v, _pos: f"{v:,.0f}" if v >= 1 else f"{v:g}",
    )
    two_panels(
        cf,
        m,
        out,
        "latency_by_dataset",
        f"Latency per item by dataset, {COST_SPLIT}, {in_flight_title(m)}",
        [
            ("latency_p50_ms", "p50 latency (ms, log)", True),
            ("latency_p95_ms", "p95 latency (ms, log)", True),
        ],
        lambda v, _pos: f"{v:,.0f}",
    )
    two_panels(
        cf,
        m,
        out,
        "cost_by_dataset",
        f"Cost by dataset, {COST_SPLIT} (Jev list price; emulators at the electricity estimate, "
        f"${m['cost']['gpu_price']['usd_per_gpu_hour']:.4f}/GPU-h)",
        [
            ("usd_per_1k_items", "$ per 1k items", False),
            ("usd_per_1k_correct", "$ per 1k correct", False),
        ],
        lambda v, _pos: f"${v:.3f}",
    )
    fig_throughput(cf, m, out)
    fig_accuracy_vs_cost(m, out)
    return sorted(p.name for p in out.glob("*.png"))


# --- main --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", type=Path, default=OUT_DIR, help=f"output directory (default {OUT_DIR})"
    )
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else REPO / args.out
    try:
        c = compared()
        study = load_study(c)
        predictions = {}
        for system in c.systems:
            name = STUDY_SYSTEM[system]
            predictions[system] = load_predictions(c.study_dir, name)
            check_items(study, name, predictions[system].items)
        check_paired(predictions)
        costs = load_costs(c, study)
        reps = replicates(study, predictions)
        check_intervals(study, reps)
        metrics = build_metrics(c, study, predictions, reps, costs)
    except ReportError as error:
        print(f"make_report: {error}", file=sys.stderr)
        return 1
    figures_dir = out / "figures"
    for stale in figures_dir.glob("*.png"):
        stale.unlink()
    figures = write_figures(metrics, figures_dir)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n", encoding="utf-8")
    (out / "README.md").write_text(readme(metrics), encoding="utf-8")
    print(f"wrote {out / 'README.md'}, {out / 'metrics.json'} and {len(figures)} figures:")
    print("\n".join(f"  {figures_dir / name}" for name in figures))
    return 0


if __name__ == "__main__":
    sys.exit(main())
