"""Regenerate the speed/quality trade-off report, ``reports/speed_quality/``, from the screen runs.

    uv run --extra plot python scripts/make_tradeoff_report.py

Writes ``README.md`` (GitHub markdown: tables and relative image links), ``metrics.json`` (every
number in the tables and figures) and ``figures/*.png`` (matplotlib/seaborn, Agg backend, fixed
theme, 150 dpi). The output depends only on the inputs below: a rerun on the same runs rewrites
the same files.

Inputs (``runs/`` is gitignored; a missing or inconsistent input stops the script with its path):

- ``runs/select/<preset>.auto_single.state_first`` for every preset in :data:`CANDIDATES`, split
  ``screen`` (2,499 items: GPQA-Diamond 99, 300 per other benchmark; the report uses the
  :data:`BENCHMARKS`, i.e. all but :data:`EXCLUDED_BENCHMARKS`). Every run must have
  answered every item without error in exactly one model call (diagnostics
  ``n_backend_calls == 1``), with the ``state_first`` layout, the ``auto_single`` strategy, no
  debiaser and :data:`IN_FLIGHT` items in flight (the tuned fast preset: its
  ``Candidate.in_flight``) (checked). Accuracy with its 95%
  stratified bootstrap interval comes from ``jevemu.bench.compare.summarize`` (the numbers
  ``scripts/run_split.py summarize`` prints); q/s and latency from ``jevemu.bench.compare.stats``
  (``run_split.py stats``); paired accuracy differences from ``jevemu.bench.compare.compare``.
- ``runs/select/jev``: split ``screen`` (Jev's ``select`` answers replayed from the response
  cache, so it has quality but no screen q/s) and split ``select`` (its measured q/s, which is
  our client's rate limit, footnoted).
- Calibration is computed here: ``jevemu.eval.report.run_calibration`` on all runs above, split
  ``screen``, 5-fold cross-fit ``temperature@signature`` (seed 0; the folds are shared by every
  system). ``raw`` is the recorded probabilities, renormalized.

The report's reliability section compares four contestants (:data:`CONTESTANTS`): the most
accurate small candidate, the best MoE (the tuned :data:`FAST_PRESET` once its Candidate row
exists, else the most accurate MoE candidate, marked untuned), the most accurate dense 27B
candidate and Jev, with reliability diagrams from the same cross-fitted study (its per-item
probabilities and bins).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray

from jevemu.bench.compare import (
    JEV_THROUGHPUT_NOTE,
    RunStats,
    RunSummary,
    load_run,
    stats,
    summarize,
)
from jevemu.eval.costs import DEFAULT_GPU_WATTS, DEFAULT_USD_PER_WH, GpuTimePrice, PriceBook
from jevemu.eval.item_scores import Interval, macro_intervals
from jevemu.eval.metrics import ece
from jevemu.eval.metrics_calib import ItemMetrics, reliability
from jevemu.eval.report import CalibrationReport, run_calibration
from jevemu.eval.splits import DATASETS, SplitName
from jevemu.scoring.single_call import LETTER_LIMIT

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns  # type: ignore[import-untyped]
    from matplotlib.axes import Axes
    from matplotlib.cbook import STEP_LOOKUP_MAP
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.text import Annotation, Text
    from matplotlib.ticker import FixedLocator, NullFormatter
    from matplotlib.transforms import Bbox
except ImportError:  # pragma: no cover - depends on the environment
    sys.exit("make_tradeoff_report: matplotlib/seaborn missing; run with `uv run --extra plot`")

REPO = Path(__file__).resolve().parent.parent
RUNS = Path("runs/select")
JEV_RUN = RUNS / "jev"
OUT_DIR = Path("reports/speed_quality")
SPLIT: SplitName = "screen"
EXCLUDED_BENCHMARKS: tuple[str, ...] = ("yelp_stars",)
"""Left out of this report (user decision): Yelp's labels ("1 star" .. "5 stars") read ambiguously.
Every table, macro average, pooled q/s and latency, figure and metrics.json entry uses the rest;
docs/research keeps the 10-benchmark numbers."""
BENCHMARKS: tuple[str, ...] = tuple(b for b in DATASETS if b not in EXCLUDED_BENCHMARKS)
JEV_QPS_SPLIT: SplitName = "select"
SUFFIX = "auto_single.state_first"
STRATEGY = "auto_single"
"""The manifest's strategy: every question in one model call (``jevemu.scoring.single_call``)."""
CALL_NOTE = (
    f"Every question is one model call. Questions with more than {LETTER_LIMIT} options are "
    "labelled with two-capital single-token codes (AA, AB, …) instead of letters."
)
LAYOUT = "state_first"
IN_FLIGHT = 16
GPU = "one RTX 3090 (24 GB)"
CALIBRATOR = "temperature@signature"
RAW, CAL = "raw", "calibrated"
ARM = {RAW: "raw", CAL: CALIBRATOR}
FOLDS = 5
SEED = 0
TIERS = (0.0, 5.0, 10.0, 20.0, 30.0, 50.0)
"""Speed tiers (minimum pooled q/s) of the best-per-tier table; empty tiers are left out."""
MINUS, TIMES, DAGGER = "\u2212", "\u00d7", "\u2020"

LABEL = {
    "gpqa_diamond_idk": "GPQA",
    "lexam_en_idk": "LEXam",
    "mmlu_pro": "MMLU-Pro",
    "arc_challenge": "ARC",
    "ag_news": "AG News",
    "banking77": "banking77",
    "clinc150": "CLINC150",
    "boolq": "BoolQ",
    "sst5": "SST-5",
    "yelp_stars": "Yelp",
}

TINY, SMALL, MID, LARGE, MOE = (
    "2-3B",
    "4B class",
    "9-14B",
    "27-32B dense",
    "26-35B MoE, 3-4B active",
)
SIZE_MARKER = {TINY: "v", SMALL: "o", MID: "s", LARGE: "D", MOE: "^"}
FAMILY_COLOR = {
    "Qwen3.6": "#d62728",
    "Qwen3.8": "#1f77b4",
    "Qwen3.5": "#ff7f0e",
    "Qwen3": "#2ca02c",
    "Qwen2.5": "#8c564b",
    "Gemma 4": "#9467bd",
    "Phi-4": "#17becf",
    "SmolLM3": "#bcbd22",
    "Llama 3.2": "#7f7f7f",
}
JEV_COLOR = "#111111"
STYLE: dict[str, Any] = {
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
}
DPI = 150

FAST_PRESET = "qwen3.6-35b-a3b-fast"
"""The tuned Qwen3.6-35B-A3B preset (it runs at its tuned ``Candidate.in_flight``). With its
Candidate row it is the best-MoE contestant of the reliability section; without it the most
accurate MoE candidate stands in, marked untuned. A screen run of it without a row stops the
script."""
SMALL_CLASSES = (TINY, SMALL)
CONTESTANTS = ("small", "moe", "dense", "jev")
"""The reliability section's contestants: the most accurate small model (:data:`SMALL_CLASSES`),
the best MoE (:data:`FAST_PRESET`), the most accurate dense 27B model, and Jev."""
CONTESTANT_COLOR = {"small": "#805ad5", "moe": "#2f855a", "dense": "#dd6b20", "jev": "#2b6cb0"}
"""One hue per contestant (dense and MoE as in reports/jev_vs_qwen)."""
HIST_EDGES = np.round(np.linspace(0.0, 1.0, 21), 2)
"""Equal-width 0.05 bins of the top-probability histograms under the reliability diagrams."""
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Candidate:
    preset: str
    """``docker/vllm/presets/<preset>.env``; the run is ``runs/select/<preset>.<SUFFIX>``."""
    label: str
    """Short name on the plots."""
    family: str
    size: str
    params: str
    """Parameter count as the model card states it."""
    weights: str
    in_flight: int = IN_FLIGHT
    """Items in flight of the run (the tuned fast preset runs at its tuned concurrency)."""


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        "qwen3.6-27b-int4-quanttrio", "Q3.6-27B QuantTrio", "Qwen3.6", LARGE, "27B", "INT4 AWQ"
    ),
    Candidate(
        "qwen3.6-27b-int4-cyankiwi", "Q3.6-27B cyankiwi", "Qwen3.6", LARGE, "27B", "INT4 AWQ"
    ),
    Candidate(
        "qwen3.8-27b-int4-palmfuture", "Q3.8-27B palmfuture", "Qwen3.8", LARGE, "27B", "INT4 GPTQ"
    ),
    Candidate("qwen3-32b-int4", "Q3-32B", "Qwen3", LARGE, "32B", "INT4 AWQ"),
    Candidate("qwen2.5-32b-int4", "Q2.5-32B", "Qwen2.5", LARGE, "32B", "INT4 AWQ"),
    Candidate(
        "qwen3.6-35b-a3b-int4-palmfuture", "Q3.6-35B-A3B", "Qwen3.6", MOE, "35B-A3B", "INT4 GPTQ"
    ),
    Candidate(
        "qwen3.6-35b-a3b-fast",
        "Q3.6-35B-A3B tuned",
        "Qwen3.6",
        MOE,
        "35B-A3B",
        "INT4 AWQ",
        in_flight=16,
    ),
    Candidate("qwen3.5-35b-a3b-int4", "Q3.5-35B-A3B", "Qwen3.5", MOE, "35B-A3B", "INT4 GPTQ"),
    Candidate(
        "qwen3-30b-a3b-2507-int4-redhat", "Q3-30B-A3B-2507", "Qwen3", MOE, "30B-A3B", "INT4 W4A16"
    ),
    Candidate(
        "gemma-4-26b-a4b-int4-cyankiwi",
        "Gemma4-26B-A4B",
        "Gemma 4",
        MOE,
        "26B-A4B",
        "INT4 QAT (AWQ/W4A16)",
    ),
    Candidate("qwen3-14b-int4", "Q3-14B", "Qwen3", MID, "14B", "INT4 AWQ"),
    Candidate("qwen3.5-9b-bf16", "Q3.5-9B", "Qwen3.5", MID, "9B", "bf16"),
    Candidate("qwen3.5-9b-int8", "Q3.5-9B int8", "Qwen3.5", MID, "9B", "INT8 W8A16"),
    Candidate("gemma-4-12b-int4", "Gemma4-12B", "Gemma 4", MID, "12B", "INT4 QAT (W4A16)"),
    Candidate("qwen3.5-4b-bf16", "Q3.5-4B", "Qwen3.5", SMALL, "4B", "bf16"),
    Candidate("qwen3-4b-2507-bf16", "Q3-4B-2507", "Qwen3", SMALL, "4B", "bf16"),
    Candidate("qwen3-4b-bf16", "Q3-4B", "Qwen3", SMALL, "4B", "bf16"),
    Candidate("gemma-4-e4b-bf16", "Gemma4-E4B", "Gemma 4", SMALL, "E4B (8B with PLE)", "bf16"),
    Candidate("phi-4-mini-bf16", "Phi-4-mini", "Phi-4", SMALL, "3.8B", "bf16"),
    Candidate("qwen3.5-2b-bf16", "Q3.5-2B", "Qwen3.5", TINY, "2B", "bf16"),
    Candidate("smollm3-3b-bf16", "SmolLM3-3B", "SmolLM3", TINY, "3B", "bf16"),
    Candidate("llama-3.2-3b-bf16", "Llama-3.2-3B", "Llama 3.2", TINY, "3.2B", "bf16"),
)


class ReportError(RuntimeError):
    """An input is missing or disagrees with the protocol."""


def fail(message: str) -> NoReturn:
    raise ReportError(message)


def require(path: Path) -> Path:
    full = REPO / path
    if not full.exists():
        fail(f"missing input {path} (inputs: the docstring of scripts/make_tradeoff_report.py)")
    return full


def run_dir(candidate: Candidate) -> Path:
    return RUNS / f"{candidate.preset}.{SUFFIX}"


# --- inputs ------------------------------------------------------------------------------------


def check_one_call(path: Path, split: SplitName) -> None:
    """Every record of :data:`BENCHMARKS` on ``split`` was answered in exactly one model call
    (its diagnostics' ``n_backend_calls``)."""
    for name, bench in load_run(REPO / path, split=split).items():
        if name not in BENCHMARKS:
            continue
        for item_id, record in bench.records.items():
            calls = None if record.diagnostics is None else record.diagnostics.n_backend_calls
            if calls != 1:
                fail(f"{path}: {name} item {item_id} took {calls} model calls, expected 1")


def check_run(
    path: Path, summary: RunSummary, run: RunStats, *, emulator: bool, in_flight: int = IN_FLIGHT
) -> None:
    """The protocol every point shares: all benchmarks, every item answered, same settings, and
    for an emulator one model call per item."""
    missing = [b for b in DATASETS if b not in summary.benchmarks]
    if missing:
        fail(f"{path}: no {SPLIT} records for {missing}")
    for name, b in summary.benchmarks.items():
        if b.n_errors or b.scores.n != b.n_items:
            fail(f"{path}: {name} has {b.scores.n}/{b.n_items} answered, {b.n_errors} errors")
    if not emulator:
        return
    system = summary.system
    if system.get("strategy") != STRATEGY:
        fail(f"{path}: strategy {system.get('strategy')!r}, expected {STRATEGY!r}")
    if (system.get("renderer") or {}).get("layout") != LAYOUT:
        fail(f"{path}: layout {system.get('renderer')!r}, expected {LAYOUT!r}")
    if system.get("debiaser") is not None:
        fail(f"{path}: debiaser {system['debiaser']!r}, expected none")
    if run.total.items_per_second is None or run.total.concurrency != (in_flight,):
        fail(f"{path}: throughput must be measured at {in_flight} in flight")
    check_one_call(path, SPLIT)


@dataclass(frozen=True)
class Loaded:
    candidates: dict[str, tuple[RunSummary, RunStats]]
    jev: RunSummary
    jev_select: RunStats
    study: CalibrationReport


def load() -> Loaded:
    prices = PriceBook(gpu=GpuTimePrice.from_power(DEFAULT_GPU_WATTS, DEFAULT_USD_PER_WH))
    fast_run = RUNS / f"{FAST_PRESET}.{SUFFIX}"
    listed = any(c.preset == FAST_PRESET for c in CANDIDATES)
    if not listed and any((REPO / fast_run).glob(f"*.{SPLIT}.jsonl")):
        fail(
            f"{fast_run} has {SPLIT} records but CANDIDATES has no {FAST_PRESET!r} row: add it "
            "with in_flight = its tuned concurrency (the report would silently leave it out)"
        )
    loaded: dict[str, tuple[RunSummary, RunStats]] = {}
    for c in CANDIDATES:
        path = run_dir(c)
        require(path / "manifest.json")
        summary = summarize(REPO / path, split=SPLIT)
        run = stats(REPO / path, prices, split=SPLIT)
        check_run(path, summary, run, emulator=True, in_flight=c.in_flight)
        preset = summary.system.get("launch_env", {}).get("JEVEMU_VLLM_PRESET")
        if preset != c.preset:
            fail(f"{path}: launched from preset {preset!r}, expected {c.preset!r}")
        loaded[c.preset] = (summary, run)
    require(JEV_RUN / "manifest.json")
    jev = summarize(REPO / JEV_RUN, split=SPLIT)
    check_run(JEV_RUN, jev, stats(REPO / JEV_RUN, prices, split=SPLIT), emulator=False)
    jev_select = stats(REPO / JEV_RUN, prices, split=JEV_QPS_SPLIT)
    if jev_select.throughput_note is None:
        fail(f"{JEV_RUN}: expected the client rate-limit caveat on its {JEV_QPS_SPLIT} q/s")
    study = run_calibration(
        {"jev": REPO / JEV_RUN, **{c.preset: REPO / run_dir(c) for c in CANDIDATES}},
        split=SPLIT,
        calibrators=[CALIBRATOR],
        primary=CALIBRATOR,
        reference="jev",
        n_folds=FOLDS,
        seed=SEED,
        benchmarks=list(BENCHMARKS),
    )
    expected = sum(jev.benchmarks[b].n_items for b in BENCHMARKS)
    if sum(study.n_items.values()) != expected:
        fail(f"calibration aligned {sum(study.n_items.values())} items, expected {expected}")
    return Loaded(loaded, jev, jev_select, study)


# --- metrics.json ------------------------------------------------------------------------------


def triple(value: Any) -> list[float]:
    return [value.estimate, value.low, value.high]


def macro_accuracy(path: Path) -> Interval:
    """Macro accuracy over :data:`BENCHMARKS` with its stratified bootstrap interval (the
    computation of ``jevemu.bench.compare.summarize``, restricted to these benchmarks)."""
    runs = load_run(path, split=SPLIT)
    columns = {
        name: np.column_stack([r.table.correct.astype(np.float64), r.table.nll, r.table.brier])
        for name, r in runs.items()
        if name in BENCHMARKS
    }
    return macro_intervals(columns)[0]


def pooled(path: Path, run: RunStats, split: SplitName = SPLIT) -> dict[str, Any]:
    """Throughput, latency and tokens pooled over :data:`BENCHMARKS`: items over summed wall
    time of the timed invocations, latency percentiles over the answered items' records."""
    parts = [run.benchmarks[b] for b in BENCHMARKS]
    seconds = math.fsum(p.wall_seconds or 0.0 for p in parts)
    items = sum(p.throughput_items for p in parts)
    latencies = np.array(
        [
            r.latency_ms
            for name, bench in load_run(path, split=split).items()
            if name in BENCHMARKS
            for r in bench.ok_records
            if r.latency_ms is not None
        ],
        dtype=np.float64,
    )
    p50, p95 = np.percentile(latencies, [50, 95])
    cached_n = sum(p.n_cached_reported for p in parts)
    calls_n = sum(p.n_diagnostics for p in parts)
    usage_n = sum(p.n_usage for p in parts)
    return {
        "qps": items / seconds if seconds else None,
        "latency_p50_ms": float(p50),
        "latency_p95_ms": float(p95),
        "wall_seconds": seconds,
        "calls_per_item": (sum(p.backend_calls or 0 for p in parts) / calls_n if calls_n else None),
        "input_tokens_per_item": sum(p.input_tokens for p in parts) / usage_n if usage_n else None,
        "cached_tokens_per_item": (
            sum(p.cached_tokens or 0 for p in parts) / cached_n if cached_n else None
        ),
    }


def paired_delta(a: Path, b: Path) -> tuple[Interval, Interval]:
    """``a - b`` macro accuracy and Brier over :data:`BENCHMARKS`, paired on shared items with
    a stratified bootstrap (``jevemu.bench.compare.compare`` restricted to these benchmarks)."""
    runs_a, runs_b = load_run(a, split=SPLIT), load_run(b, split=SPLIT)
    diffs = {}
    for name in BENCHMARKS:
        ra, rb = runs_a[name], runs_b[name]
        answered = set(rb.table.item_ids)
        shared = [i for i in ra.table.item_ids if i in answered]
        ta, tb = ra.table.take(shared), rb.table.take(shared)
        diffs[name] = np.column_stack(
            [
                ta.correct.astype(np.float64) - tb.correct.astype(np.float64),
                ta.nll - tb.nll,
                ta.brier - tb.brier,
            ]
        )
    macro = macro_intervals(diffs)
    return macro[0], macro[2]


def calibration(study: CalibrationReport, system: str) -> dict[str, dict[str, float]]:
    return {
        key: {
            metric: study.estimates[(system, arm, "macro")][metric]
            for metric in ("ece", "brier", "nll")
        }
        for key, arm in ARM.items()
    }


def frontier(points: Sequence[Mapping[str, Any]], key: str, *, higher: bool) -> list[str]:
    """Presets not dominated in (q/s up, ``key`` up if ``higher`` else down), fastest first."""
    sign = 1.0 if higher else -1.0
    ordered = sorted(points, key=lambda p: (-p["qps"], -sign * p[key], p["preset"]))
    best = -math.inf
    kept = []
    for p in ordered:
        if sign * p[key] > best:
            kept.append(p["preset"])
            best = sign * p[key]
    return kept


def item_metrics(study: CalibrationReport, system: str) -> dict[str, dict[str, ItemMetrics]]:
    """Arm key -> benchmark -> ``system``'s per-item metrics from the study's predictions."""
    probs: dict[str, dict[str, list[FloatArray]]] = {k: {b: [] for b in BENCHMARKS} for k in ARM}
    gold: dict[str, list[int]] = {b: [] for b in BENCHMARKS}
    for row in study.predictions[system]:
        gold[row["benchmark"]].append(int(row["gold"]))
        for key, arm in ARM.items():
            probs[key][row["benchmark"]].append(np.asarray(row["arms"][arm], dtype=np.float64))
    return {k: {b: ItemMetrics.from_probs(probs[k][b], gold[b]) for b in BENCHMARKS} for k in ARM}


def shares(confidence: FloatArray) -> list[float]:
    """Share of ``confidence`` in each :data:`HIST_EDGES` bin (the last bin includes 1)."""
    counts, _ = np.histogram(confidence, bins=HIST_EDGES)
    return [float(c) / confidence.shape[0] for c in counts]


def bins_json(bins: Sequence[Any]) -> list[dict[str, float]]:
    return [{"n": b.n, "confidence": b.confidence, "accuracy": b.accuracy} for b in bins]


def contestant_quality(study: CalibrationReport, system: str) -> dict[str, Any]:
    """``system``'s macro and per-benchmark accuracy, NLL, Brier and ECE raw and calibrated (the
    study's), its mean top probability, its reliability bins (all items pooled into equal-mass
    bins, and each benchmark's own) and the top-probability histograms under the diagrams."""
    items = item_metrics(study, system)
    n_bins = int(study.config["n_bins"])
    metrics = ("accuracy", "nll", "brier", "ece")
    out: dict[str, Any] = {
        "macro": {},
        "benchmarks": {b: {} for b in BENCHMARKS},
        "reliability": {"pooled": {}, "benchmarks": {b: {} for b in BENCHMARKS}},
        "histogram": {"pooled": {}, "benchmarks": {b: {} for b in BENCHMARKS}},
    }
    for key, arm in ARM.items():
        confidence = np.concatenate([items[key][b].confidence for b in BENCHMARKS])
        correct = np.concatenate([items[key][b].correct for b in BENCHMARKS])
        mean_top = math.fsum(float(items[key][b].confidence.mean()) for b in BENCHMARKS)
        out["macro"][key] = {m: study.estimates[(system, arm, "macro")][m] for m in metrics} | {
            "mean_top_prob": mean_top / len(BENCHMARKS)
        }
        out["reliability"]["pooled"][key] = {
            "n": int(confidence.shape[0]),
            "ece": ece(confidence, correct, n_bins=n_bins),
            "bins": bins_json(reliability(confidence, correct, n_bins=n_bins)),
        }
        out["histogram"]["pooled"][key] = shares(confidence)
        for b in BENCHMARKS:
            out["benchmarks"][b][key] = {m: study.estimates[(system, arm, b)][m] for m in metrics}
            out["reliability"]["benchmarks"][b][key] = bins_json(
                study.reliability[(system, arm, b)]
            )
            out["histogram"]["benchmarks"][b][key] = shares(items[key][b].confidence)
    return out


def qualifies(p: Mapping[str, Any], role: str) -> bool:
    if role == "small":
        return bool(p["size_class"] in SMALL_CLASSES)
    if role == "moe":
        return bool(p["size_class"] == MOE)
    return bool(p["size_class"] == LARGE and p["params"] == "27B")


def most_accurate(points: Sequence[Mapping[str, Any]], role: str) -> Mapping[str, Any]:
    eligible = [p for p in points if qualifies(p, role)]
    if not eligible:
        fail(f"no candidate qualifies as the {role} contestant")
    return max(eligible, key=lambda p: (p["accuracy"][0], p["preset"]))


def contestants(
    study: CalibrationReport, points: Sequence[Mapping[str, Any]], jev: Mapping[str, Any]
) -> dict[str, Any]:
    """The reliability section's :data:`CONTESTANTS`, each with :func:`contestant_quality`."""
    tuned = [p for p in points if p["preset"] == FAST_PRESET]
    chosen = {
        "small": most_accurate(points, "small"),
        "moe": tuned[0] if tuned else most_accurate(points, "moe"),
        "dense": most_accurate(points, "dense"),
    }
    title = {
        "small": f"Best small ({' / '.join(SMALL_CLASSES)})",
        "moe": "Best MoE, tuned" if tuned else "Best MoE, untuned",
        "dense": "Best dense 27B",
        "jev": "Jev",
    }
    entries = []
    for role in CONTESTANTS:
        entry: dict[str, Any] = {"role": role, "title": title[role]}
        if role == "jev":
            entry |= {
                "preset": None,
                "label": "Jev",
                "system_id": jev["system_id"],
                "accuracy": jev["accuracy"],
                "qps": None,
                "select_qps": jev["select_qps"],
                "latency_p50_ms": jev["select_latency_p50_ms"],
                "in_flight": IN_FLIGHT,
                "benchmark_accuracy": {b: jev["benchmarks"][b]["accuracy"] for b in BENCHMARKS},
            }
            system = "jev"
        else:
            p = chosen[role]
            entry |= {
                "preset": p["preset"],
                "label": p["label"],
                "accuracy": p["accuracy"],
                "qps": p["qps"],
                "latency_p50_ms": p["latency_p50_ms"],
                "in_flight": p["in_flight"],
                "benchmark_accuracy": {b: p["benchmarks"][b]["accuracy"] for b in BENCHMARKS},
            }
            system = p["preset"]
        entries.append(entry | contestant_quality(study, system))
    return {
        "tuned_moe": bool(tuned),
        "selection": "most accurate (macro, screen) small-class candidate, the tuned "
        f"{FAST_PRESET} preset (else the most accurate MoE candidate, untuned), the most "
        "accurate dense 27B candidate, and Jev",
        "reliability": "top label, the study's 10 equal-mass bins; pooled = all screen items of "
        "the reported benchmarks in one binning",
        "histogram_edges": [float(e) for e in HIST_EDGES],
        "entries": entries,
    }


def build_metrics(data: Loaded) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for c in CANDIDATES:
        summary, run = data.candidates[c.preset]
        cal = calibration(data.study, c.preset)
        points.append(
            {
                "preset": c.preset,
                "label": c.label,
                "family": c.family,
                "size_class": c.size,
                "params": c.params,
                "weights": c.weights,
                "in_flight": c.in_flight,
                "run_dir": str(run_dir(c)),
                "checkpoint": summary.system.get("backend_model"),
                "revision": summary.system.get("model_revision"),
                "template_id": summary.system["renderer"]["template_id"],
                "n_items": sum(summary.benchmarks[b].n_items for b in BENCHMARKS),
                "accuracy": triple(macro_accuracy(REPO / run_dir(c))),
                **pooled(REPO / run_dir(c), run),
                "ece_raw": cal[RAW]["ece"],
                "ece_cal": cal[CAL]["ece"],
                "brier_raw": cal[RAW]["brier"],
                "brier_cal": cal[CAL]["brier"],
                "nll_raw": cal[RAW]["nll"],
                "nll_cal": cal[CAL]["nll"],
                "benchmarks": {
                    name: {
                        "n": b.n_items,
                        "accuracy": triple(b.scores.accuracy),
                        "qps": run.benchmarks[name].items_per_second,
                    }
                    for name, b in summary.benchmarks.items()
                    if name in BENCHMARKS
                },
            }
        )
    by_preset = {p["preset"]: p for p in points}
    for p in points:
        p["accuracy_estimate"] = p["accuracy"][0]
    fronts = {
        "accuracy": frontier(points, "accuracy_estimate", higher=True),
        "ece_cal": frontier(points, "ece_cal", higher=False),
        "brier_cal": frontier(points, "brier_cal", higher=False),
        "ece_raw": frontier(points, "ece_raw", higher=False),
        "brier_raw": frontier(points, "brier_raw", higher=False),
    }
    for p in points:
        del p["accuracy_estimate"]
        p["frontier"] = {k: p["preset"] in v for k, v in fronts.items()}

    steps = []
    acc_front = fronts["accuracy"][::-1]  # most accurate (slowest) first
    for slow, fast in pairwise(acc_front):
        d_acc, d_brier = paired_delta(
            REPO / by_preset[fast]["run_dir"], REPO / by_preset[slow]["run_dir"]
        )
        steps.append(
            {
                "from": slow,
                "to": fast,
                "speedup": by_preset[fast]["qps"] / by_preset[slow]["qps"],
                "delta_accuracy": triple(d_acc),
                "delta_brier": triple(d_brier),
            }
        )

    best_overall = max(points, key=lambda p: (p["accuracy"][0], p["preset"]))
    tiers = []
    for floor in TIERS:
        eligible = [p for p in points if p["qps"] >= floor]
        if not eligible:
            continue
        best = max(eligible, key=lambda p: (p["accuracy"][0], p["preset"]))
        d_acc, _ = paired_delta(REPO / best["run_dir"], REPO / best_overall["run_dir"])
        tiers.append(
            {
                "min_qps": floor,
                "n_candidates": len(eligible),
                "best": best["preset"],
                "delta_vs_best_overall": triple(d_acc),
            }
        )

    jev_cal = calibration(data.study, "jev")
    jev_select = pooled(REPO / JEV_RUN, data.jev_select, split=JEV_QPS_SPLIT)
    jev = {
        "system_id": data.jev.system_id,
        "accuracy": triple(macro_accuracy(REPO / JEV_RUN)),
        "accuracy_argmax": data.study.estimates[("jev", ARM[RAW], "macro")]["accuracy"],
        "ece_raw": jev_cal[RAW]["ece"],
        "ece_cal": jev_cal[CAL]["ece"],
        "brier_raw": jev_cal[RAW]["brier"],
        "brier_cal": jev_cal[CAL]["brier"],
        "nll_raw": jev_cal[RAW]["nll"],
        "nll_cal": jev_cal[CAL]["nll"],
        "screen_qps": None,
        "select_qps": jev_select["qps"],
        "select_latency_p50_ms": jev_select["latency_p50_ms"],
        "qps_note": JEV_THROUGHPUT_NOTE,
        "benchmarks": {
            name: {"n": b.n_items, "accuracy": triple(b.scores.accuracy)}
            for name, b in data.jev.benchmarks.items()
            if name in BENCHMARKS
        },
    }
    return {
        "split": SPLIT,
        "excluded_benchmarks": list(EXCLUDED_BENCHMARKS),
        "n_items": dict(data.study.n_items),
        "protocol": {
            "layout": LAYOUT,
            "strategy": STRATEGY,
            "model_calls_per_question": 1,
            "labels": CALL_NOTE,
            "in_flight": IN_FLIGHT,
            "gpu": GPU,
            "engine": "vLLM 0.30.0 (pinned image), one server at a time, fresh container per run",
            "qps": "items sent (response-cache hits excluded) / summed wall time of the runner "
            "invocations (startup excluded)",
        },
        "calibration": {
            "calibrator": CALIBRATOR,
            "folds": FOLDS,
            "seed": SEED,
            "ece": "top label, 10 equal-mass bins per benchmark, macro = mean over benchmarks",
            "brier": "multiclass, sum over options, range [0, 2]; macro = mean over benchmarks",
        },
        "jev": jev,
        "contestants": contestants(data.study, points, jev),
        "candidates": sorted(points, key=lambda p: (-p["accuracy"][0], p["preset"])),
        "frontiers": fronts,
        "frontier_steps": steps,
        "tiers": tiers,
    }


# --- README ------------------------------------------------------------------------------------


def num(value: float, digits: int, *, signed: bool = False) -> str:
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return text.replace("-", MINUS)


def ci(value: Sequence[float], digits: int, *, signed: bool = False) -> str:
    est, low, high = value
    return (
        f"{num(est, digits, signed=signed)} "
        f"[{num(low, digits, signed=signed)}, {num(high, digits, signed=signed)}]"
    )


def qps(value: float) -> str:
    return f"{value:.1f}" if value >= 10 else f"{value:.2f}"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


FRONTIER_MARKS = (("acc.", "accuracy"), ("ECE", "ece_cal"), ("Brier", "brier_cal"))


def marks(p: Mapping[str, Any]) -> str:
    return ", ".join(name for name, key in FRONTIER_MARKS if p["frontier"][key])


def main_table(m: Mapping[str, Any]) -> str:
    rows = []
    for rank, p in enumerate(m["candidates"], start=1):
        rows.append(
            [
                str(rank),
                f"**{p['label']}**" if p["frontier"]["accuracy"] else p["label"],
                f"`{p['preset']}`",
                p["params"],
                p["weights"],
                ci(p["accuracy"], 4),
                qps(p["qps"]),
                f"{p['latency_p50_ms']:,.0f}",
                f"{p['latency_p95_ms']:,.0f}",
                f"{p['ece_raw']:.3f}",
                f"{p['ece_cal']:.3f}",
                f"{p['brier_raw']:.3f}",
                f"{p['brier_cal']:.3f}",
                marks(p),
            ]
        )
    j = m["jev"]
    rows.append(
        [
            "",
            "**Jev**",
            f"`{j['system_id']}`",
            "",
            "",
            ci(j["accuracy"], 4),
            f"— ({qps(j['select_qps'])} {DAGGER})",
            f"{j['select_latency_p50_ms']:,.0f} {DAGGER}",
            "",
            f"{j['ece_raw']:.3f}",
            f"{j['ece_cal']:.3f}",
            f"{j['brier_raw']:.3f}",
            f"{j['brier_cal']:.3f}",
            "",
        ]
    )
    header = [
        "#",
        "Model",
        "Preset",
        "Params",
        "Weights",
        "Macro acc. [95% CI]",
        "q/s",
        "p50 ms",
        "p95 ms",
        "ECE raw",
        "ECE cal.",
        "Brier raw",
        "Brier cal.",
        "Frontier",
    ]
    return table(header, rows)


def frontier_table(m: Mapping[str, Any]) -> str:
    by = {p["preset"]: p for p in m["candidates"]}
    rows = []
    for preset in m["frontiers"]["accuracy"][::-1]:
        p = by[preset]
        rows.append([p["label"], ci(p["accuracy"], 4), qps(p["qps"]), f"{p['ece_cal']:.3f}"])
    return table(["Accuracy frontier (slowest first)", "Macro acc.", "q/s", "ECE cal."], rows)


def steps_table(m: Mapping[str, Any]) -> str:
    by = {p["preset"]: p for p in m["candidates"]}
    rows = [
        [
            f"{by[s['from']]['label']} → {by[s['to']]['label']}",
            f"{s['speedup']:.2f}{TIMES}",
            f"{qps(by[s['from']]['qps'])} → {qps(by[s['to']]['qps'])}",
            ci(s["delta_accuracy"], 4, signed=True),
            ci(s["delta_brier"], 3, signed=True),
        ]
        for s in m["frontier_steps"]
    ]
    header = [
        "Step down the frontier",
        "Speed-up",
        "q/s",
        "Δ macro acc. [95% CI]",
        "Δ macro Brier (raw) [95% CI]",
    ]
    return table(header, rows)


def tiers_table(m: Mapping[str, Any]) -> str:
    by = {p["preset"]: p for p in m["candidates"]}
    rows = []
    for t in m["tiers"]:
        p = by[t["best"]]
        rows.append(
            [
                "any" if t["min_qps"] == 0 else f"≥ {t['min_qps']:g}",
                str(t["n_candidates"]),
                p["label"],
                ci(p["accuracy"], 4),
                qps(p["qps"]),
                "—" if t["min_qps"] == 0 else ci(t["delta_vs_best_overall"], 4, signed=True),
                f"{p['ece_cal']:.3f}",
                f"{p['brier_cal']:.3f}",
            ]
        )
    header = [
        "Speed tier (q/s)",
        "Candidates",
        "Most accurate",
        "Macro acc.",
        "q/s",
        "Δ vs most accurate overall [95% CI]",
        "ECE cal.",
        "Brier cal.",
    ]
    return table(header, rows)


def per_dataset_table(m: Mapping[str, Any]) -> str:
    by = {p["preset"]: p for p in m["candidates"]}
    names = list(m["n_items"])
    rows = []
    for preset in m["frontiers"]["accuracy"][::-1]:
        p = by[preset]
        rows.append(
            [
                p["label"],
                *[f"{p['benchmarks'][b]['accuracy'][0]:.3f}" for b in names],
                f"{p['accuracy'][0]:.3f}",
            ]
        )
    j = m["jev"]
    rows.append(
        [
            "**Jev**",
            *[f"{j['benchmarks'][b]['accuracy'][0]:.3f}" for b in names],
            f"{j['accuracy'][0]:.3f}",
        ]
    )
    return table(["Model", *[LABEL.get(b, b) for b in names], "Macro"], rows)


def arrow(raw: float, cal: float, digits: int = 3, *, signed: bool = False) -> str:
    return f"{num(raw, digits, signed=signed)} → {num(cal, digits, signed=signed)}"


def contestant_table(m: Mapping[str, Any]) -> str:
    rows = []
    for e in m["contestants"]["entries"]:
        raw, cal = e["macro"][RAW], e["macro"][CAL]
        jev = e["role"] == "jev"
        rows.append(
            [
                e["title"],
                e["label"],
                f"`{e['system_id'] if jev else e['preset']}`",
                ci(e["accuracy"], 4),
                arrow(raw["nll"], cal["nll"]),
                arrow(raw["brier"], cal["brier"]),
                arrow(raw["ece"], cal["ece"]),
                arrow(
                    raw["mean_top_prob"] - raw["accuracy"],
                    cal["mean_top_prob"] - cal["accuracy"],
                    signed=True,
                ),
                f"— ({qps(e['select_qps'])} {DAGGER})" if jev else qps(e["qps"]),
                f"{e['latency_p50_ms']:,.0f}" + (f" {DAGGER}" if jev else ""),
                str(e["in_flight"]),
            ]
        )
    header = [
        "Role",
        "Model",
        "Preset",
        "Macro acc. [95% CI]",
        "NLL raw → cal.",
        "Brier raw → cal.",
        "ECE raw → cal.",
        f"Mean top p {MINUS} acc., raw → cal.",
        "q/s",
        "p50 ms",
        "In flight",
    ]
    return table(header, rows)


def contestant_dataset_table(m: Mapping[str, Any]) -> str:
    """Per benchmark (and macro), each contestant's accuracy and its ECE raw → calibrated."""
    entries = m["contestants"]["entries"]
    rows = []
    for b in [*BENCHMARKS, "macro"]:
        macro = b == "macro"
        row = [
            "**Macro**" if macro else LABEL.get(b, b),
            f"{sum(m['n_items'].values()) if macro else m['n_items'][b]:,}",
        ]
        for e in entries:
            values = e["macro"] if macro else e["benchmarks"][b]
            accuracy = e["accuracy"] if macro else e["benchmark_accuracy"][b]
            row += [f"{accuracy[0]:.3f}", arrow(values[RAW]["ece"], values[CAL]["ece"])]
        rows.append(row)
    header = ["Benchmark", "n"]
    for e in entries:
        header += [f"{e['label']} acc.", f"{e['label']} ECE raw → cal."]
    return table(header, rows)


def contestants_section(m: Mapping[str, Any]) -> list[str]:
    c = m["contestants"]
    by = {e["role"]: e for e in c["entries"]}
    n = sum(m["n_items"].values())
    smallest = min(m["n_items"], key=lambda b: m["n_items"][b])
    moe = by["moe"]
    stand_in = (
        ""
        if c["tuned_moe"]
        else f" The tuned Qwen3.6-35B-A3B preset (`{FAST_PRESET}`) has no screen run yet, so the "
        f"most accurate MoE candidate stands in: {moe['label']} (`{moe['preset']}`, untuned, "
        f"{moe['in_flight']} in flight)."
    )
    small = " or ".join(SMALL_CLASSES).replace("-", " to ")
    return [
        "## Calibration and reliability of four systems",
        "",
        f"This section compares four systems on the same {n:,} `{SPLIT}` items: the most accurate "
        f"small model ({small}), the best MoE (mixture-of-experts) model, the most accurate dense "
        f"27B model, and Jev.{stand_in} The calibrated values use the same cross-fitted "
        f"`{CALIBRATOR}` as the table above. NLL is the negative log-likelihood of the correct "
        "answer. Mean top probability minus accuracy is the overconfidence that the reliability "
        "diagrams show.",
        "",
        contestant_table(m),
        "",
        f"{DAGGER} Jev's `{JEV_QPS_SPLIT}` q/s and p50 reflect our client's rate limit (see the "
        "footnote above).",
        "",
        "Each reliability diagram plots accuracy against mean top probability in 10 equal-mass "
        "bins. The diagonal is perfect calibration. Marker area is proportional to a bin's item "
        "count. The strip under each diagram is a histogram of top probability (0.05-wide bins, "
        f"share of items). The overall figure pools all {n:,} items into one set of bins, so its "
        "pooled ECE differs from the tables' macro ECE (the mean of the per-benchmark ECEs).",
        "",
        figure(
            "reliability_overall",
            "Reliability diagrams of the four systems, all screen items pooled, raw and calibrated",
        ),
        "",
        "The next table gives each system's accuracy and ECE (raw, then calibrated) per "
        "benchmark. Each panel of the figures under it shows the bins behind one benchmark's ECE. "
        f"{LABEL[smallest]} has {m['n_items'][smallest]} items, so its "
        f"bins hold about {m['n_items'][smallest] / 10:.0f} items each (the others about "
        f"{max(m['n_items'].values()) / 10:.0f}).",
        "",
        contestant_dataset_table(m),
        "",
        figure("reliability_by_dataset_raw", "Reliability diagrams by dataset, raw"),
        "",
        figure("reliability_by_dataset_calibrated", "Reliability diagrams by dataset, calibrated"),
        "",
    ]


def findings(m: Mapping[str, Any]) -> list[str]:
    by = {p["preset"]: p for p in m["candidates"]}
    top = m["candidates"][0]
    front = [by[x] for x in m["frontiers"]["accuracy"][::-1]]
    fastest = max(m["candidates"], key=lambda p: p["qps"])
    out = [
        f"{top['label']} is the most accurate candidate: {num(top['accuracy'][0], 4)} macro "
        f"accuracy at {qps(top['qps'])} q/s. Jev scores {num(m['jev']['accuracy'][0], 4)} on the "
        f"same items. The fastest candidate, {fastest['label']}, answers {qps(fastest['qps'])} "
        f"q/s at {num(fastest['accuracy'][0], 4)}.",
        "The accuracy frontier, from slowest to fastest: "
        + ", ".join(
            f"{p['label']} ({num(p['accuracy'][0], 3)} at {qps(p['qps'])} q/s)" for p in front
        )
        + ".",
    ]
    small = [p for p in m["candidates"] if p["size_class"] in (SMALL, TINY)]
    if small:
        best_small = max(small, key=lambda p: (p["accuracy"][0], p["preset"]))
        out.append(
            f"{best_small['label']} is the best small model (≤ 4B class): "
            f"{num(best_small['accuracy'][0], 4)} at {qps(best_small['qps'])} q/s, "
            f"{num(best_small['accuracy'][0] - top['accuracy'][0], 4, signed=True)} against "
            f"{top['label']}."
        )
    tier_text = []
    for t in m["tiers"]:
        if t["min_qps"] == 0:
            continue
        p = by[t["best"]]
        tier_text.append(
            f"≥ {t['min_qps']:g} q/s: {p['label']}, {num(p['accuracy'][0], 3)} "
            f"({ci(t['delta_vs_best_overall'], 3, signed=True)})"
        )
    if tier_text:
        out.append(
            "Most accurate candidate per speed tier (accuracy, and Δ against the most accurate "
            "candidate overall):" + "".join(f"\n   - {x}" for x in tier_text)
        )
    step_text = [
        f"{by[s['from']]['label']} to {by[s['to']]['label']}: {s['speedup']:.1f}{TIMES} faster, "
        f"{ci(s['delta_accuracy'], 3, signed=True)} accuracy"
        for s in m["frontier_steps"]
    ]
    if step_text:
        out.append(
            "Accuracy cost of each step down the frontier (paired on the screen items):"
            + "".join(f"\n   - {x}" for x in step_text)
        )
    cal = sorted(m["candidates"], key=lambda p: p["ece_cal"])
    over = [p for p in m["candidates"] if p["ece_raw"] > 2 * p["ece_cal"] and p["ece_raw"] > 0.15]
    out.append(
        f"Cross-fitted `{CALIBRATOR}` brings every candidate's macro ECE into the range "
        f"{num(cal[0]['ece_cal'], 3)} to {num(cal[-1]['ece_cal'], 3)} (Jev: "
        f"{num(m['jev']['ece_raw'], 3)} raw, {num(m['jev']['ece_cal'], 3)} calibrated). "
        + (
            "Candidates whose raw ECE is above 0.15 and more than halved by calibration: "
            + ", ".join(
                f"{p['label']} ({num(p['ece_raw'], 3)} to {num(p['ece_cal'], 3)})" for p in over
            )
            + "."
            if over
            else "No candidate has a raw ECE above 0.15 that calibration more than halves."
        )
    )
    return out


def figure(name: str, alt: str) -> str:
    return f"![{alt}](figures/{name}.png)"


def off_protocol(m: Mapping[str, Any]) -> str:
    """The candidates run at another in-flight count than :data:`IN_FLIGHT` (the tuned fast
    preset), e.g. ``Q3.6-35B-A3B tuned at 64``; empty without any."""
    return ", ".join(
        f"{p['label']} at {p['in_flight']}" for p in m["candidates"] if p["in_flight"] != IN_FLIGHT
    )


def readme(m: Mapping[str, Any]) -> str:
    n = sum(m["n_items"].values())
    j = m["jev"]
    tuned = off_protocol(m)
    except_tuned = f" (except {tuned}, its tuned client concurrency)" if tuned else ""
    parts = [
        "# Speed/quality trade-off: every emulator candidate on the screen split",
        "",
        f"Every emulator candidate answered the same {n:,} `{SPLIT}` items from "
        f"{len(m['n_items'])} benchmarks ({m['n_items']['gpqa_diamond_idk']} for GPQA-Diamond, "
        "300 for each other). The plots show one point per candidate: throughput on x, quality "
        "on y. Method and candidate notes: "
        "[docs/research/selection.md](../../docs/research/selection.md) "
        '("Speed/quality trade-off") and '
        "[docs/research/candidates.md](../../docs/research/candidates.md).",
        "",
        f"All runs used the same settings: `{LAYOUT}` layout, `{STRATEGY}` scoring, no "
        f"debiaser, {IN_FLIGHT} items in flight{except_tuned}, one vLLM 0.30.0 server on {GPU}, "
        f"and a fresh container per run. {CALL_NOTE}",
        "",
        "- Yelp review stars is left out because of its label wording. Macro averages cover "
        f"the other {len(m['n_items'])} benchmarks; docs/research/* keep the 10-benchmark "
        "numbers.",
        "- q/s (questions per second) is pooled: all items sent (response-cache hits "
        "excluded) divided by the summed wall time of the runner invocations (server startup "
        f"excluded), at {IN_FLIGHT} in flight"
        f"{except_tuned}. p50 and p95 are the median and 95th-percentile latency per item under "
        "that load.",
        f"- Accuracy is macro (each benchmark weighs 1/{len(m['n_items'])}), with a 95% "
        'stratified bootstrap interval. An "I don\'t know" answer counts as wrong.',
        "- ECE (expected calibration error: top label, 10 equal-mass bins) and Brier score "
        "(multiclass, 0 to 2) are macro averages; lower is better. Raw means the recorded "
        f"probabilities. Calibrated means `{CALIBRATOR}`, cross-fitted in {FOLDS} folds on these "
        "screen items, with every system on the same folds.",
        "- Calibration is out of sample: each item is calibrated by a fit that never saw it, so "
        "the numbers carry no in-sample optimism. The setup differs from the holdout study "
        "([calibration.md](../../docs/research/calibration.md)) only in its smaller n: about "
        "2,240 training items per fold instead of a whole split. The per-signature temperatures "
        "are therefore noisier, and ECE on 300-item benchmarks carries more binning noise.",
        "- Jev is the horizontal reference line in the plots. Its screen answers are replayed "
        "from the response cache, so it has no screen q/s. Its measured `select` q/s "
        f"({qps(j['select_qps'])}) is our client's rate limit ({DAGGER}).",
        "- Jev's accuracy here counts its returned `choice`, as `run_split.py summarize` "
        "does. [reports/summary](../summary/index.html) scores every system by the argmax of its "
        f"probabilities, which gives Jev {num(j['accuracy_argmax'], 4)} on these items. The two "
        "differ only on near-ties.",
        "- A candidate is on a Pareto frontier when no other candidate is both at least as fast "
        "and at least as good. Depending on the frontier, better means higher accuracy, lower "
        "calibrated ECE or lower calibrated Brier.",
        "",
        "## Findings",
        "",
        *[f"{i}. {text}" for i, text in enumerate(findings(m), start=1)],
        "",
        "## Accuracy",
        "",
        figure("accuracy_vs_qps", "Macro accuracy against q/s, one point per candidate"),
        "",
        "## Calibration",
        "",
        figure("ece_vs_qps", "Macro ECE raw and calibrated against q/s"),
        "",
        figure("brier_vs_qps", "Macro Brier raw and calibrated against q/s"),
        "",
        "## All candidates",
        "",
        "Ranked by macro accuracy. Bold names are on the accuracy frontier. The Frontier column "
        "lists the frontiers a candidate is on: accuracy, calibrated ECE, calibrated Brier.",
        "",
        main_table(m),
        "",
        f"{DAGGER} {j['qps_note']}",
        "",
        *contestants_section(m),
        "## Frontier",
        "",
        frontier_table(m),
        "",
        steps_table(m),
        "",
        "Most accurate candidate per speed tier. Tiers use pooled q/s, and Δ is paired on the "
        "screen items:",
        "",
        tiers_table(m),
        "",
        "Accuracy per benchmark for the accuracy frontier:",
        "",
        per_dataset_table(m),
        "",
        "## Caveats",
        "",
        "- The screen split is small: 99 GPQA-Diamond items and 300 for each other "
        "benchmark. A macro accuracy interval is about ±0.017 wide, and paired differences "
        "below about 0.01 are not resolved.",
        f"- q/s is measured at {IN_FLIGHT} items in flight, the protocol of every run"
        + (f" but {tuned} (its tuned client concurrency)" if tuned else "")
        + ". The fastest models answer an item in a few hundred milliseconds, so the in-flight "
        "limit and the client bound them, not the GPU. More in flight would raise their q/s "
        "more than the dense 27B models' [INFERENCE: not measured].",
        "- banking77 and CLINC150 (77 and 150 options) took several model calls per item in "
        "earlier runs (a trie over the intent names). Here they take one call, with code labels.",
        "- Five other 4-bit builds of plotted base models are not shown. They were screened only "
        "in the earlier Stage 1 of [selection.md](../../docs/research/selection.md) and not "
        "re-run.",
        "",
        "## Regenerate",
        "",
        "```bash",
        "uv run --extra plot python scripts/make_tradeoff_report.py",
        "```",
        "",
        "The script reads the run directories listed in its docstring (gitignored `runs/`). It "
        "checks that every run used the same protocol, computes the cross-fitted calibration "
        "and rewrites this directory. Only aggregates are written.",
        "",
    ]
    return "\n".join(parts)


# --- figures -----------------------------------------------------------------------------------


def log_qps_axis(ax: Axes, m: Mapping[str, Any]) -> None:
    values = [p["qps"] for p in m["candidates"]]
    ax.set_xscale("log")
    low, high = min(values) / 1.35, max(values) * 1.6
    ax.set_xlim(low, high)
    ticks = [t for t in (1, 2, 3, 5, 10, 20, 30, 50, 100, 200) if low <= t <= high]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(lambda v, _pos: f"{v:g}")
    ax.xaxis.set_minor_formatter(NullFormatter())
    tuned = off_protocol(m)
    ax.set_xlabel(
        f"Throughput, q/s (log scale; pooled over the screen, {IN_FLIGHT} in flight"
        + (f"; {tuned}" if tuned else "")
        + ")"
    )


def legends(ax: Axes, m: Mapping[str, Any], extra: Sequence[Line2D] = ()) -> None:
    families = list(dict.fromkeys(p["family"] for p in m["candidates"]))
    sizes = [s for s in SIZE_MARKER if any(p["size_class"] == s for p in m["candidates"])]
    fam = [
        Line2D([], [], marker="o", linestyle="", color=FAMILY_COLOR[f], label=f, markersize=6)
        for f in families
    ]
    size = [
        Line2D([], [], marker=SIZE_MARKER[s], linestyle="", color="#555555", label=s, markersize=6)
        for s in sizes
    ]
    first = ax.legend(
        handles=fam,
        title="Family",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7.5,
        title_fontsize=8,
    )
    ax.add_artist(first)
    ax.legend(
        handles=[*size, *extra],
        title="Size",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.52),
        fontsize=7.5,
        title_fontsize=8,
    )


OFFSETS: tuple[tuple[float, float, str, str], ...] = tuple(
    (dx * r, dy * r, ha, va)
    for r in (1.0, 2.2, 3.6, 5.2, 7.0)
    for dx, dy, ha, va in (
        (5, 3, "left", "bottom"),
        (5, -3, "left", "top"),
        (-5, 3, "right", "bottom"),
        (-5, -3, "right", "top"),
        (0, 6, "center", "bottom"),
        (0, -6, "center", "top"),
    )
)
"""Candidate label positions (offset in points, alignment), nearest first."""
LABEL_GAP = 4.0
"""Clearance kept between a label's text and the text of a placed label or fixed text (pixels),
so two labels never touch end to end."""
TEXT_OVERLAP_WEIGHT = 10.0
"""How much worse text over text is than a label (or its leader line) over a marker."""


def overlap(a: Bbox, b: Bbox) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if w > 0 and h > 0 else 0.0


def line_boxes(ax: Axes) -> list[Bbox]:
    """Pixel boxes, 2 px thick, of the straight segments drawn in ``ax``: reference lines,
    frontier steps, error bars and raw-to-calibrated connectors (all axis-aligned)."""
    paths = []
    for line in ax.get_lines():
        x, y = STEP_LOOKUP_MAP[line.get_drawstyle()](*np.asarray(line.get_xydata()).T)
        paths.append(line.get_transform().transform(np.column_stack([x, y])))
    for collection in ax.collections:
        if isinstance(collection, LineCollection):
            paths += [collection.get_transform().transform(s) for s in collection.get_segments()]
    return [
        Bbox.from_extents(min(x0, x1) - 1, min(y0, y1) - 1, max(x0, x1) + 1, max(y0, y1) + 1)
        for path in paths
        for (x0, y0), (x1, y1) in pairwise(path)
        if np.isfinite([x0, y0, x1, y1]).all()
    ]


def place_labels(
    fig: Figure,
    ax: Axes,
    items: Sequence[tuple[float, float, str, str]],
    *,
    points: Sequence[tuple[float, float]] = (),
    texts: Sequence[Annotation] = (),
) -> None:
    """Greedy, deterministic label placement: each label (x, y, text, color) takes the nearest
    offset whose box (text and leader line) overlaps no marker (its own, the other labels' and
    ``points``), placed box or drawn line (:func:`line_boxes`), whose text comes no closer than
    :data:`LABEL_GAP` to a placed label's text or fixed text (``texts``), and that stays inside
    the axes; else the least overlap, text over text weighted by :data:`TEXT_OVERLAP_WEIGHT`.
    Labels moved away from their point get a thin leader line."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    frame = ax.get_window_extent(renderer)
    radius = 4.5 * fig.dpi / 72
    markers = []
    for x, y in [*((x, y) for x, y, _, _ in items), *points]:
        px, py = ax.transData.transform((x, y))
        markers.append(Bbox.from_extents(px - radius, py - radius, px + radius, py + radius))
    placed: list[Bbox] = [text.get_window_extent(renderer) for text in texts]
    lines = line_boxes(ax)
    words = list(placed)
    for x, y, text, color in items:
        best: tuple[float, Annotation, Bbox, Bbox] | None = None
        for dx, dy, ha, va in OFFSETS:
            far = abs(dx) > 6 or abs(dy) > 7
            note = ax.annotate(
                text,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=6.5,
                color=color,
                arrowprops=(
                    {
                        "arrowstyle": "-",
                        "color": "#aaaaaa",
                        "linewidth": 0.5,
                        "shrinkA": 0,
                        "shrinkB": 3,
                    }
                    if far
                    else None
                ),
                zorder=5,
            )
            box = note.get_window_extent(renderer)  # the text and its leader line
            label = Text.get_window_extent(note, renderer)  # the text alone
            cost = sum(overlap(box, other) for other in [*markers, *placed, *lines])
            cost += TEXT_OVERLAP_WEIGHT * sum(
                overlap(label, other.padded(LABEL_GAP)) for other in words
            )
            outside = box.width * box.height - overlap(box, frame)
            cost += 4.0 * outside
            if best is None or cost < best[0]:
                if best is not None:
                    best[1].remove()
                best = (cost, note, box, label)
                if cost == 0.0:
                    break
            else:
                note.remove()
        assert best is not None
        placed.append(best[2])
        words.append(best[3])


def jev_line(ax: Axes, value: float, text: str, style: str, *, below: bool = False) -> Annotation:
    """Jev's reference line with its label at the left edge, above the line (``below``: under
    it, so two close lines keep their labels apart)."""
    ax.axhline(value, color=JEV_COLOR, linewidth=0.9, linestyle=style, zorder=1)
    return ax.annotate(
        text,
        (0.0, value),
        xycoords=("axes fraction", "data"),
        xytext=(4, -2 if below else 2),
        textcoords="offset points",
        fontsize=7,
        color=JEV_COLOR,
        va="top" if below else "bottom",
    )


def frontier_line(ax: Axes, points: Sequence[Mapping[str, Any]], key: str, **kw: Any) -> None:
    """The boundary of the region the frontier dominates: at each q/s, the best value any
    candidate at least that fast reaches (a staircase through the frontier points)."""
    ordered = sorted(points, key=lambda p: p["qps"])
    xs = [p["qps"] for p in ordered]
    ys = [p[key] for p in ordered]
    ax.plot(xs, ys, drawstyle="steps-pre", zorder=1, **kw)


LOG_Y_TICKS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
"""Labeled ticks of a log y axis (:func:`fig_calibration`)."""


def fig_accuracy(m: Mapping[str, Any], out: Path) -> None:
    pts = m["candidates"]
    fig, ax = plt.subplots(figsize=(10.5, 6.2), layout="constrained")
    front = [p for p in pts if p["frontier"]["accuracy"]]
    frontier_line(
        ax,
        [{"qps": p["qps"], "acc": p["accuracy"][0]} for p in front],
        "acc",
        color="#444444",
        linewidth=1.1,
        alpha=0.6,
        label="Pareto frontier",
    )
    for p in pts:
        est, low, high = p["accuracy"]
        color = FAMILY_COLOR[p["family"]]
        ax.errorbar(
            p["qps"],
            est,
            yerr=[[est - low], [high - est]],
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            alpha=0.6,
            capsize=0,
            zorder=2,
        )
        ax.scatter(
            p["qps"],
            est,
            marker=SIZE_MARKER[p["size_class"]],
            s=46,
            color=color,
            edgecolors="black" if p["frontier"]["accuracy"] else "white",
            linewidths=1.0 if p["frontier"]["accuracy"] else 0.5,
            zorder=3,
        )
    j = m["jev"]
    jev_text = jev_line(ax, j["accuracy"][0], f"Jev (screen): {j['accuracy'][0]:.3f}", "--")
    log_qps_axis(ax, m)
    lows = [p["accuracy"][1] for p in pts]
    ax.set_ylim(min(lows) - 0.02, max(j["accuracy"][0], max(p["accuracy"][2] for p in pts)) + 0.02)
    ax.set_ylabel("Macro accuracy (95% CI)")
    ax.set_title(f"Accuracy vs throughput, {SPLIT} split (n = {sum(m['n_items'].values()):,})")
    extra = [
        Line2D([], [], color="#444444", alpha=0.6, label="Pareto frontier"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            label="on the frontier",
        ),
    ]
    legends(ax, m, extra)
    order = sorted(pts, key=lambda p: (not p["frontier"]["accuracy"], -p["accuracy"][0]))
    place_labels(
        fig,
        ax,
        [(p["qps"], p["accuracy"][0], p["label"], "#222222") for p in order],
        texts=[jev_text],
    )
    save(fig, out, "accuracy_vs_qps")


def fig_calibration(
    m: Mapping[str, Any], out: Path, metric: str, ylabel: str, *, log_y: bool = False
) -> None:
    """Raw and calibrated ``metric`` against q/s, the y axis fitted to the values (and Jev's)
    so the labels have room; ``log_y`` spreads the calibrated values, which crowd at the low
    end of a linear axis (ECE)."""
    pts = m["candidates"]
    raw_key, cal_key = f"{metric}_raw", f"{metric}_cal"
    fig, ax = plt.subplots(figsize=(10.5, 6.2), layout="constrained")
    for key, style in ((raw_key, ":"), (cal_key, "-")):
        front = [p for p in pts if p["frontier"][key]]
        frontier_line(ax, front, key, color="#444444", linewidth=1.1, alpha=0.6, linestyle=style)
    for p in pts:
        color = FAMILY_COLOR[p["family"]]
        marker = SIZE_MARKER[p["size_class"]]
        ax.plot(
            [p["qps"], p["qps"]],
            [p[raw_key], p[cal_key]],
            color=color,
            linewidth=0.7,
            alpha=0.5,
            zorder=2,
        )
        ax.scatter(
            p["qps"],
            p[raw_key],
            marker=marker,
            s=40,
            facecolors="white",
            edgecolors=color,
            linewidths=1.0,
            zorder=3,
        )
        ax.scatter(
            p["qps"],
            p[cal_key],
            marker=marker,
            s=46,
            color=color,
            edgecolors="black" if p["frontier"][cal_key] else "white",
            linewidths=1.0 if p["frontier"][cal_key] else 0.5,
            zorder=3,
        )
    j = m["jev"]
    jev_texts = [
        jev_line(ax, j[raw_key], f"Jev raw: {j[raw_key]:.3f}", ":"),
        jev_line(ax, j[cal_key], f"Jev calibrated: {j[cal_key]:.3f}", "--", below=True),
    ]
    log_qps_axis(ax, m)
    top = max(max(p[raw_key] for p in pts), j[raw_key])
    low = min(min(p[cal_key] for p in pts), j[cal_key])
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(low / 1.15, top * 1.15)
        ax.yaxis.set_major_locator(FixedLocator(LOG_Y_TICKS))
        ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:g}")
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        margin = 0.08 * (top - low)
        ax.set_ylim(low - margin, top + margin)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{metric.upper() if metric == 'ece' else metric.capitalize()} vs throughput, "
        f"{SPLIT} split: raw (hollow) and cross-fitted {CALIBRATOR} (filled)"
    )
    extra = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="#555555",
            label="raw",
        ),
        Line2D([], [], marker="o", linestyle="", color="#555555", label="calibrated"),
        Line2D([], [], color="#444444", alpha=0.6, label="frontier, calibrated"),
        Line2D([], [], color="#444444", alpha=0.6, linestyle=":", label="frontier, raw"),
    ]
    legends(ax, m, extra)
    order = sorted(pts, key=lambda p: (not p["frontier"][cal_key], p[cal_key]))
    place_labels(
        fig,
        ax,
        [(p["qps"], p[cal_key], p["label"], "#222222") for p in order],
        points=[(p["qps"], p[raw_key]) for p in pts],
        texts=jev_texts,
    )
    save(fig, out, f"{metric}_vs_qps")


def reliability_panel(
    fig: Figure,
    spec: Any,
    entries: Sequence[Mapping[str, Any]],
    bins: Sequence[Sequence[Mapping[str, float]]],
    hist: Sequence[Sequence[float]],
    edges: Sequence[float],
    *,
    title: str,
    ylabels: bool,
    fontsize: float,
    marker_area: float,
) -> Axes:
    """A reliability diagram over a confidence-histogram strip in ``spec`` (a gridspec cell):
    per contestant (``entries``, with its ``bins`` and ``hist``), the bins' accuracy against
    mean top probability, marker area ``marker_area x items / most items in any bin of the
    panel`` (proportional to the count), and the share of items per equal-width confidence bin
    as a step line. Returns the diagram's axes."""
    grid = spec.subgridspec(2, 1, height_ratios=(4, 1), hspace=0.05)
    main = fig.add_subplot(grid[0])
    strip = fig.add_subplot(grid[1], sharex=main)
    main.plot([0, 1], [0, 1], color="#999999", linewidth=0.8, linestyle="--", zorder=0)
    largest = max(b["n"] for panel in bins for b in panel)
    for e, panel, shares_ in zip(entries, bins, hist, strict=True):
        color = CONTESTANT_COLOR[e["role"]]
        x = [b["confidence"] for b in panel]
        y = [b["accuracy"] for b in panel]
        # unclipped: bins at confidence or accuracy 1 sit on the frame
        main.plot(x, y, color=color, linewidth=1.2, zorder=2, clip_on=False)
        main.scatter(
            x,
            y,
            s=[marker_area * b["n"] / largest for b in panel],
            color=color,
            edgecolors="white",
            linewidths=0.6,
            alpha=0.85,
            zorder=3,
            clip_on=False,
        )
        strip.stairs(shares_, edges, color=color, linewidth=1.0)
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
    return main


def reliability_legend(fig: Figure, m: Mapping[str, Any], title: str) -> None:
    """One legend of the contestants above the panels, headed by the figure's title."""
    entries = m["contestants"]["entries"]
    legend = fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=CONTESTANT_COLOR[e["role"]],
                marker="o",
                markersize=5,
                label=e["label"] if e["role"] == "jev" else f"{e['title']}: {e['label']}",
            )
            for e in entries
        ],
        loc="outside upper center",
        ncols=len(entries),
        frameon=False,
        title=title,
        title_fontproperties={"weight": "bold", "size": 11},
    )
    legend.get_title().set_multialignment("center")


def fig_reliability_overall(m: Mapping[str, Any], out: Path) -> None:
    c = m["contestants"]
    entries = c["entries"]
    titles = {RAW: "Raw", CAL: f"Calibrated (cross-fitted {CALIBRATOR})"}
    fig = plt.figure(figsize=(11, 6.0), layout="constrained")
    grid = fig.add_gridspec(1, len(titles))
    for col, (arm, title) in enumerate(titles.items()):
        main = reliability_panel(
            fig,
            grid[0, col],
            entries,
            [e["reliability"]["pooled"][arm]["bins"] for e in entries],
            [e["histogram"]["pooled"][arm] for e in entries],
            c["histogram_edges"],
            title=title,
            ylabels=col == 0,
            fontsize=8.5,
            marker_area=60.0,
        )
        text = "\n".join(
            f"{e['label']} {e['reliability']['pooled'][arm]['ece']:.3f}" for e in entries
        )
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
    n = entries[0]["reliability"]["pooled"][RAW]["n"]
    reliability_legend(
        fig,
        m,
        f"Reliability, {SPLIT} split: all {n:,} items pooled, 10 equal-mass bins "
        "(marker area ∝ items); strip: top-probability histogram",
    )
    save(fig, out, "reliability_overall")


def fig_reliability_grid(m: Mapping[str, Any], out: Path, arm: str) -> None:
    c = m["contestants"]
    entries = c["entries"]
    n_cols = 5
    n_rows = math.ceil(len(BENCHMARKS) / n_cols)
    fig = plt.figure(figsize=(15, 3.6 * n_rows + 0.8), layout="constrained")
    grid = fig.add_gridspec(n_rows, n_cols)
    for index, b in enumerate(BENCHMARKS):
        reliability_panel(
            fig,
            grid[index // n_cols, index % n_cols],
            entries,
            [e["reliability"]["benchmarks"][b][arm] for e in entries],
            [e["histogram"]["benchmarks"][b][arm] for e in entries],
            c["histogram_edges"],
            title=f"{LABEL.get(b, b)} (n = {m['n_items'][b]:,})",
            ylabels=index % n_cols == 0,
            fontsize=7.5,
            marker_area=36.0,
        )
    kind = "raw" if arm == RAW else f"calibrated (cross-fitted {CALIBRATOR})"
    reliability_legend(
        fig,
        m,
        f"Reliability by dataset, {SPLIT} split, {kind}\n10 equal-mass bins per benchmark "
        "(marker area ∝ items); strip: top-probability histogram",
    )
    save(fig, out, f"reliability_by_dataset_{arm}")


def save(fig: Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=DPI, metadata={"Software": None})
    plt.close(fig)


def write_figures(m: Mapping[str, Any], out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    sns.set_theme(context="paper", style="whitegrid", font="DejaVu Sans", rc=STYLE)
    fig_accuracy(m, out)
    fig_calibration(
        m, out, "ece", "Macro ECE (log scale; top label, 10 equal-mass bins)", log_y=True
    )
    fig_calibration(m, out, "brier", "Macro Brier (multiclass, 0 to 2)")
    fig_reliability_overall(m, out)
    for arm in ARM:
        fig_reliability_grid(m, out, arm)
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
        metrics = build_metrics(load())
    except ReportError as error:
        print(f"make_tradeoff_report: {error}", file=sys.stderr)
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
