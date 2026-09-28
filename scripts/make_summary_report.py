"""Regenerate the summary report, ``reports/summary/index.html``, with interactive Plotly figures.

    uv run python scripts/make_summary_report.py

Writes ``index.html`` (one page: prose, tables and Plotly figures; plotly.js comes from its CDN,
the figure data is inline JSON, so no Python plotting dependency) and ``metrics.json`` (every
number on the page). The same inputs give byte-identical output. Only aggregates are written: no
question text and no per-item outputs.

Two splits answer two questions:

- **Landscape** (split ``screen``, 2,499 items): every emulator candidate listed in
  ``reports/speed_quality/metrics.json`` (its ``candidates``: preset, label, size class, run
  directory; that report checks their protocol), plus Jev, the API-served models of
  :data:`FINALISTS` and :data:`CLM`, a dual encoder that answers Jev's wire format (kept out of
  the local models' frontier and counts). Macro accuracy and Brier score against questions per
  second.
- **Finalists** (split ``holdout``, the half model selection never saw): :data:`FINALISTS`.
  Accuracy with paired differences against Jev, Brier and ECE raw and calibrated, q/s, cost,
  latency, and reliability diagrams pooled and per dataset.

**Benchmark views.** OpenAI returns at most 5 top logprobs and OpenRouter 20, so API models skip
benchmarks with more options than that (MMLU-Pro, banking77, CLINC150 for GPT-6 Luna; banking77
and CLINC150 for DeepSeek). ``common`` is the benchmarks every shown system answered; ``all`` is
the 9 of :data:`jevemu.eval.splits.DATASETS`, shown only for systems that answered all of them.
Yelp is dropped (``DROPPED_DATASETS``).

**Quality.** Per-item probabilities from :func:`jevemu.eval.crossfit.load_run_items`,
renormalized; a run recorded with a debiaser is scored on its pre-debias probabilities (the
deployed configuration has none). Accuracy is the argmax of those probabilities (Jev's returned
``choice`` differs on near-ties). *Calibrated* is :data:`CALIBRATOR` cross-fitted in
:data:`FOLDS` folds (seed :data:`SEED`) on the same split's items of that system: every item is
scored by a fit that never saw it. Macros weigh benchmarks equally, with 95% stratified bootstrap
intervals (:func:`jevemu.eval.item_scores.macro_intervals`); ECE is top-label, 10 equal-mass bins
per benchmark, averaged. Reliability diagrams use equal-width confidence bins instead, so they
show the whole confidence range (:data:`POOLED_BIN_WIDTH` pooled over a view,
:data:`BENCHMARK_BIN_WIDTH` per benchmark; bins under :data:`MIN_BIN_ITEMS` questions are left
out).

**Speed and cost.** q/s is items sent (response-cache hits excluded) over the summed wall time of
the timed runner invocations of the view's benchmarks (:func:`jevemu.bench.compare.stats`), pooled
over questions (so it depends on the benchmark mix, unlike the equal-weight accuracy macro), from
each system's ``Finalist.speed`` run: the run of its deployed configuration. Cost is the price
book's (:class:`jevemu.eval.costs.PriceBook`: token prices; local GPUs at the estimated
electricity price). API systems ran behind our client's request limiter, so their q/s is a floor,
not the provider's capacity (flagged when q/s is within 5% of the limit); :func:`ceiling` computes
what the service's own limits (:class:`ServiceLimits`) and the in-flight bound would allow without
it. Local speed is one machine (:data:`LOCAL_SETUP`). Every local and API run the page reads (the
finalists' ``holdout`` and speed runs, the landscape's ``screen`` runs) must have answered in
exactly one model call per question (diagnostics ``n_backend_calls == 1``,
:func:`check_one_call`); Jev's and CLM's records carry no diagnostics.

**Unanswered items.** A run with failed items counts as incomplete, except for the items a
finalist declares in ``Finalist.unanswered``: the failed items of each declared split must be
exactly those. The system is scored on the items it answered, and paired comparisons use the
items both systems answered. The page and ``metrics.json`` report the counts.

Systems marked ``in_progress`` are shown as pending until every split the page needs is complete
in their run manifest; then they join every table and figure. No finalist is marked today.
"""

from __future__ import annotations

import argparse
import html
import itertools
import json
import math
import sys
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn

import numpy as np
from numpy.typing import NDArray

from jevemu.bench.compare import BenchmarkRun, UsageStats, load_run, stats
from jevemu.bench.manifest import RunManifest
from jevemu.calibrate.temperature import T_MAX, T_MIN
from jevemu.eval.costs import DEFAULT_GPU_WATTS, DEFAULT_USD_PER_WH, GpuTimePrice, PriceBook
from jevemu.eval.crossfit import (
    DEFAULT_MIN_GROUP,
    CalibratorSpec,
    CalItem,
    assign_folds,
    crossfit_predict,
    debias_offline,
    load_run_items,
)
from jevemu.eval.item_scores import Interval, macro_intervals
from jevemu.eval.metrics import DEFAULT_BOOTSTRAP_RESAMPLES, NLL_EPS
from jevemu.eval.metrics_calib import ItemMetrics
from jevemu.eval.splits import DATASETS, DROPPED_DATASETS, FrozenSplit, SplitName
from jevemu.scoring.single_call import LETTER_LIMIT

FloatArray = NDArray[np.float64]

REPO = Path(__file__).resolve().parents[1]
RUNS = Path("runs/select")
OUT = Path("reports/summary")
SPEED_QUALITY = Path("reports/speed_quality/metrics.json")
CALIBRATOR = "temperature@signature"
FOLDS = 5
SEED = 0
N_BINS = 10
"""Equal-mass bins per benchmark of the ECE."""
POOLED_BIN_WIDTH = 0.05
BENCHMARK_BIN_WIDTH = 0.1
MIN_BIN_ITEMS = 10
"""Reliability bins holding fewer questions are not drawn (their accuracy is noise)."""
LIMITED_SHARE = 0.95
"""An API system's q/s at or above this share of its request limit is marked client-limited."""
SHOWN_CEILING_SHARE = 1.05
"""A ceiling is drawn when it exceeds the measured q/s by at least this factor."""
LOCAL_SETUP = "vLLM 0.30.0 on one RTX 3090 (24 GB)"
"""The only local machine and engine measured (``docker/vllm/compose.yaml``)."""
PLOTLY_JS = "https://cdn.plot.ly/plotly-2.35.2.min.js"

LANDSCAPE_SPLIT: SplitName = "screen"
FINAL_SPLIT: SplitName = "holdout"
PAGE_SPLITS: tuple[SplitName, ...] = ("screen", "select", "holdout")
"""The splits the page describes, in the order they were used."""
SPLITS_DIR = RUNS / "splits"
"""Frozen split files, ``<benchmark>.<split>.jsonl`` (one question per line)."""

GPQA = "gpqa_diamond_idk"
"""Few questions (99 per half), yet the equal-weight macro weighs it like AG News's 3,800:
:func:`accuracy_lead` checks how much of Jev's lead rests on it."""

BENCHMARK_LABELS = {
    "gpqa_diamond_idk": "GPQA-Diamond",
    "lexam_en_idk": "LEXam",
    "mmlu_pro": "MMLU-Pro",
    "arc_challenge": "ARC-Challenge",
    "ag_news": "AG News",
    "banking77": "banking77",
    "clinc150": "CLINC150",
    "boolq": "BoolQ",
    "sst5": "SST-5",
}

Kind = Literal["jev", "local", "api"]


@dataclass(frozen=True)
class ServiceLimits:
    """An API service's own limits: what bounds its q/s once our client's limiter is gone."""

    source: str
    """Where the limits come from (HTML)."""
    requests_per_min: int | None = None
    tokens: tuple[int, Literal["s", "min"]] | None = None
    """Token limit and the unit it is published in."""
    batched: bool = False
    """The API answers many questions per request, so neither its request limit nor the
    per-request latency bounds q/s; only its token limit does."""


@dataclass(frozen=True)
class Finalist:
    key: str
    label: str
    kind: Kind
    color: str
    detail: str
    run: Path
    """Its run directory: ``holdout`` for the finalists, ``screen`` for the landscape (API
    models and Jev; local models come from the speed/quality registry)."""
    speed: tuple[Path, SplitName]
    """Run and split whose timing and cost describe it (its deployed configuration)."""
    landscape_speed: tuple[Path, SplitName]
    """The same for its landscape point (Jev's screen answers are cache replays)."""
    preset: str | None = None
    """Local models: the preset of its landscape point in the speed/quality registry."""
    max_rpm: int | None = None
    """API systems: our client's request-start limit per minute for these runs."""
    limits: ServiceLimits | None = None
    """API systems: the service's own limits, for the q/s ceiling without our limiter."""
    unanswered: frozenset[tuple[SplitName, str]] = frozenset()
    """(split, item id) of the items of ``run`` it is known never to answer: their error records
    are accepted (module docstring)."""
    in_progress: bool = False

    def unanswered_on(self, split: SplitName) -> frozenset[str]:
        return frozenset(item for s, item in self.unanswered if s == split)


def _run(name: str) -> Path:
    return RUNS / name


QWEN27 = _run("qwen3.6-27b-int4-quanttrio.auto_single.state_first")
FAST_MOE = _run("qwen3.6-35b-a3b-fast.auto_single.state_first")
GPTQ_MOE = _run("qwen3.6-35b-a3b-int4-palmfuture.auto_single.state_first")
GEMMA_MOE = _run("gemma-4-26b-a4b-int4-cyankiwi.auto_single.state_first")
LUNA = _run("gpt-6-luna.constrained.state_first")
DEEPSEEK = _run("deepseek-v4.1-flash@makora.constrained.state_first")
JEV = _run("jev")

FINALISTS: tuple[Finalist, ...] = (
    Finalist(
        key="jev",
        label="Jev",
        kind="jev",
        color="#24292f",
        detail="TypeSafe's jev-1.13.0 API: the system being emulated",
        run=JEV,
        speed=(JEV, "holdout"),
        landscape_speed=(JEV, "holdout"),
        max_rpm=1200,  # JevClient's default, Jev's documented limit
        limits=ServiceLimits(
            source="Jev documents limits of 1,200 requests/min and 250,000 tokens/s. These runs "
            "sent one question per request, so 1,200 requests/min is exactly 20 q/s. One request "
            "can carry many questions, though, so the token limit is the actual bound, assuming "
            "it counts the tokens Jev reports, internal ones included. This bound is inferred from "
            "the docs and was not measured.",
            requests_per_min=1200,
            tokens=(250_000, "s"),
            batched=True,
        ),
    ),
    Finalist(
        key="qwen27",
        label="Qwen3.6-27B (local)",
        kind="local",
        color="#0969da",
        detail="The selected emulator: dense 27B, QuantTrio INT4 AWQ, vLLM on one RTX 3090",
        run=QWEN27,
        speed=(QWEN27, "holdout"),
        landscape_speed=(QWEN27, "screen"),
        preset="qwen3.6-27b-int4-quanttrio",
    ),
    Finalist(
        key="moe",
        label="Qwen3.6-35B-A3B AWQ (local)",
        kind="local",
        color="#1a7f37",
        detail="The fast emulator: mixture of experts (MoE) with 3B active parameters, QuantTrio "
        "INT4 AWQ, vLLM flags tuned on one RTX 3090",
        run=FAST_MOE,
        speed=(FAST_MOE, "holdout"),
        landscape_speed=(FAST_MOE, "screen"),
        preset="qwen3.6-35b-a3b-fast",
    ),
    Finalist(
        key="gptq",
        label="Qwen3.6-35B-A3B GPTQ (local)",
        kind="local",
        color="#0f8b8d",
        detail="The same MoE model, palmfuture INT4 GPTQ, the preset's default vLLM flags on one "
        "RTX 3090",
        run=GPTQ_MOE,
        speed=(GPTQ_MOE, "holdout"),
        landscape_speed=(GPTQ_MOE, "screen"),
        preset="qwen3.6-35b-a3b-int4-palmfuture",
    ),
    Finalist(
        key="gemma",
        label="Gemma 4 26B-A4B (local)",
        kind="local",
        color="#bf3989",
        detail="Google's Gemma 4 MoE with 3.8B of 25.2B parameters active, cyankiwi INT4 AWQ of "
        "Google's QAT checkpoint, the preset's default vLLM flags on one RTX 3090",
        run=GEMMA_MOE,
        speed=(GEMMA_MOE, "holdout"),
        landscape_speed=(GEMMA_MOE, "screen"),
        preset="gemma-4-26b-a4b-int4-cyankiwi",
    ),
    Finalist(
        key="luna",
        label="GPT-6 Luna (API)",
        kind="api",
        color="#8250df",
        detail="OpenAI gpt-6-luna, reasoning off, answers through Structured Outputs (a JSON enum "
        "over the labels); reads the top-5 logprobs where the answer value starts",
        run=LUNA,
        speed=(LUNA, "holdout"),
        landscape_speed=(LUNA, "screen"),
        max_rpm=450,  # run_split.py --max-rpm default
        limits=ServiceLimits(
            source="The OpenAI account used here has limits of 500 requests and 200,000 tokens "
            "per minute (read from response headers); higher usage tiers allow more. Beyond the "
            "account limits, q/s is bounded by the requests in flight divided by the measured "
            "latency.",
            requests_per_min=500,
            tokens=(200_000, "min"),
        ),
    ),
    Finalist(
        key="deepseek",
        label="DeepSeek V4.1 Flash (API)",
        kind="api",
        color="#e16f24",
        detail="deepseek/deepseek-v4.1-flash via OpenRouter (Makora provider), reasoning off, "
        "answers through Structured Outputs (a JSON enum over the labels); reads the top-20 "
        "logprobs where the answer value starts",
        run=DEEPSEEK,
        speed=(DEEPSEEK, "holdout"),
        landscape_speed=(DEEPSEEK, "screen"),
        max_rpm=450,  # run_split.py --max-rpm default
        limits=ServiceLimits(
            source="OpenRouter sets no request limit on paid models, and the Makora provider's "
            "own limit was not measured. These runs kept 16 requests in flight.",
        ),
        # Makora's reply loops on whitespace before the answer value, even when resent
        # (docs/research/openrouter_probe_report.md)
        unanswered=frozenset(
            {
                ("select", "sst5:407"),
                ("select", "sst5:563"),
                ("select", "sst5:1332"),
                ("holdout", "sst5:372"),
                ("holdout", "sst5:751"),
                ("holdout", "sst5:1340"),
            }
        ),
    ),
)

SIZE_CLASS_COLORS = {  # keys: the registry's size classes
    "27-32B dense": "#80b8f0",
    "26-35B MoE, 3-4B active": "#7fc995",
    "9-14B": "#d9b24c",
    "4B class": "#8c959f",
    "2-3B": "#c4cad1",
}


def fail(message: str) -> NoReturn:
    sys.exit(f"make_summary_report: {message}")


# --- run status --------------------------------------------------------------------------------


def failed_items(runs: Mapping[str, BenchmarkRun]) -> dict[str, set[str]]:
    """Benchmark -> the items whose latest attempt failed (benchmarks without any left out)."""
    failed = {
        name: {item_id for item_id, record in run.records.items() if not record.ok}
        for name, run in runs.items()
    }
    return {name: items for name, items in failed.items() if items}


def complete_benchmarks(
    run_dir: Path, split: SplitName, unanswered: frozenset[str] = frozenset()
) -> set[str]:
    """Benchmarks of ``split`` the run's manifest records as complete without errors, or (with
    ``unanswered``) as tried in full with failed items that are all in ``unanswered``."""
    manifest = RunManifest.load(run_dir) if run_dir.is_dir() else None
    if manifest is None:
        return set()
    failed = failed_items(load_run(REPO / run_dir, split=split)) if unanswered else {}
    return {
        entry.benchmark
        for entry in manifest.runs.values()
        if entry.split == split
        and entry.benchmark not in DROPPED_DATASETS
        and (
            (entry.status == "complete" and entry.n_errors == 0 and entry.n_ok == entry.n_items)
            or (
                entry.status == "incomplete"
                and entry.n_ok + entry.n_errors == entry.n_items
                and entry.benchmark in failed
                and failed[entry.benchmark] <= unanswered
            )
        )
    }


def needs(finalist: Finalist) -> list[tuple[Path, SplitName]]:
    return [
        (finalist.run, FINAL_SPLIT),
        finalist.speed,
        (finalist.run if finalist.kind != "local" else finalist.landscape_speed[0], "screen"),
        finalist.landscape_speed,
    ]


def readiness(finalist: Finalist) -> dict[str, str] | None:
    """``None`` when every run the page needs is complete, else split -> progress text.

    The benchmarks expected on every split are those the system completed on ``screen`` (the
    API models skip what their top-k cannot show), or all of :data:`DATASETS` for local runs
    and Jev."""
    screen_run = finalist.run if finalist.kind != "local" else finalist.landscape_speed[0]
    expected = (
        set(DATASETS) if finalist.kind != "api" else complete_benchmarks(screen_run, "screen")
    )
    progress: dict[str, str] = {}
    ready = bool(expected)
    for run_dir, split in dict.fromkeys(needs(finalist)):
        done = complete_benchmarks(run_dir, split, finalist.unanswered_on(split)) & expected
        progress[split] = f"{len(done)}/{len(expected)} benchmarks"
        ready &= done == expected
    return None if ready else progress


# --- quality -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scores:
    """One system's items on one split: per benchmark, raw and calibrated item metrics."""

    ids: dict[str, list[str]]
    raw: dict[str, ItemMetrics]
    cal: dict[str, ItemMetrics]
    unanswered: frozenset[str] = frozenset()
    """Items of the split it is declared never to answer (``Finalist.unanswered``)."""

    @property
    def benchmarks(self) -> list[str]:
        return list(self.ids)


def score(run_dir: Path, split: SplitName, unanswered: frozenset[str] = frozenset()) -> Scores:
    try:
        items = load_run_items(REPO / run_dir, split=split)
    except Exception as error:  # the record format of an unmerged branch, a torn file, ...
        fail(f"{run_dir} ({split}): cannot load its records: {error}")
    if not items:
        fail(f"{run_dir} has no {split} records")
    manifest = RunManifest.load(REPO / run_dir)
    debiaser = manifest.system.get("debiaser") if manifest else None
    flat: list[CalItem] = [item for rows in items.values() for item in rows]
    folds = assign_folds(flat, n_folds=FOLDS, seed=SEED)
    if debiaser:
        if any(item.raw is None for item in flat):
            fail(f"{run_dir}: debiaser {debiaser!r} but items without pre-debias probabilities")
        base = debias_offline(flat, "none", folds)
    else:
        base = {item.item_id: item.probs for item in flat}
    calibrated, _ = crossfit_predict(flat, base, folds, CalibratorSpec.parse(CALIBRATOR))
    ids, raw, cal = {}, {}, {}
    for benchmark, rows in items.items():
        gold = [item.gold for item in rows]
        ids[benchmark] = [item.item_id for item in rows]
        raw[benchmark] = ItemMetrics.from_probs([base[item.item_id] for item in rows], gold)
        cal[benchmark] = ItemMetrics.from_probs([calibrated[item.item_id] for item in rows], gold)
    return Scores(ids, raw, cal, unanswered)


# --- speed and cost ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Speed:
    split: SplitName
    usage: dict[str, UsageStats]
    latencies: dict[str, FloatArray]
    """Benchmark -> latencies (ms) of the answered items."""


def speed(run_dir: Path, split: SplitName, prices: PriceBook) -> Speed:
    run_stats = stats(REPO / run_dir, prices, split=split)
    latencies = {
        name: np.array(
            [r.latency_ms for r in run.ok_records if r.latency_ms is not None], dtype=np.float64
        )
        for name, run in load_run(REPO / run_dir, split=split).items()
    }
    return Speed(split, dict(run_stats.benchmarks), latencies)


def check_one_call(
    run_dir: Path, split: SplitName, unanswered: frozenset[str] = frozenset()
) -> None:
    """Every record of :data:`DATASETS` on ``split`` but the failed ones of ``unanswered`` was
    answered in exactly one model call (its diagnostics' ``n_backend_calls``); local and API
    runs (Jev's records carry no diagnostics)."""
    for name, run in load_run(REPO / run_dir, split=split).items():
        if name not in DATASETS:
            continue
        for item_id, record in run.records.items():
            if item_id in unanswered and not record.ok:
                continue
            calls = None if record.diagnostics is None else record.diagnostics.n_backend_calls
            if calls != 1:
                fail(f"{run_dir} ({split}): {name} item {item_id} took {calls} model calls")


def unanswered_counts(finalist: Finalist) -> dict[str, dict[str, dict[str, int]]]:
    """Split -> benchmark -> unanswered items (``n``) of its tried items (``of``), for every
    split ``finalist`` declares unanswered items on. Fails unless the run's failed items on that
    split are exactly the declared ones."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for split in PAGE_SPLITS:
        declared = finalist.unanswered_on(split)
        if not declared:
            continue
        runs = load_run(REPO / finalist.run, split=split)
        failed = failed_items(runs)
        found = {item for items in failed.values() for item in items}
        if found != declared:
            fail(
                f"{finalist.label} ({split}): failed items {sorted(found)} are not the declared "
                f"unanswered items {sorted(declared)}"
            )
        out[split] = {
            name: {"n": len(items), "of": len(runs[name].records)} for name, items in failed.items()
        }
    return out


def api_read(run_dir: Path, split: SplitName, benchmarks: Sequence[str]) -> dict[str, Any]:
    """How complete an API run's read of ``benchmarks`` was: the share of answered questions
    whose returned logprobs lacked at least one option label (which then got an upper bound
    instead of its logprob), the smallest probability the listed labels held on one question,
    and the questions whose top option is a missing one (bounded, not observed)."""
    runs = load_run(REPO / run_dir, split=split)
    flat = [
        r.diagnostics for b in benchmarks for r in runs[b].ok_records if r.diagnostics is not None
    ]
    return {
        "missing_label_share": sum(bool(d.missing_labels) for d in flat) / len(flat),
        "min_label_mass": min(d.observed_mass for d in flat),
        "n_missing_top": sum(
            bool(d.missing_labels)
            and max(d.raw_probabilities, key=d.raw_probabilities.__getitem__) in d.missing_labels
            for d in flat
        ),
    }


# --- per-view aggregates -----------------------------------------------------------------------


def interval(value: Interval) -> list[float]:
    return [value.estimate, value.low, value.high]


def view_summary(scores: Scores, spd: Speed, benchmarks: Sequence[str]) -> dict[str, Any]:
    """Macro quality, pooled speed and cost of one system over ``benchmarks``."""
    columns = {
        b: np.column_stack(
            [
                scores.raw[b].correct.astype(np.float64),
                scores.raw[b].nll,
                scores.raw[b].brier,
                scores.cal[b].nll,
                scores.cal[b].brier,
            ]
        )
        for b in benchmarks
    }
    acc, nll_raw, brier_raw, nll_cal, brier_cal = macro_intervals(
        columns, n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES, seed=SEED
    )
    usage = [spd.usage[b] for b in benchmarks]
    walls = [u.wall_seconds for u in usage]
    sent = sum(u.throughput_items for u in usage)
    wall = None if any(w is None for w in walls) else math.fsum(w for w in walls if w)
    costs = [u.cost_usd for u in usage]
    n_priced = sum(u.n_items for u in usage)
    cost = None if any(c is None for c in costs) else math.fsum(c for c in costs if c)
    latencies = np.concatenate([spd.latencies[b] for b in benchmarks])
    n_usage = sum(u.n_usage for u in usage)
    tokens_in = sum(u.input_tokens for u in usage) / n_usage if n_usage else None
    tokens_out = sum(u.output_tokens for u in usage) / n_usage if n_usage else None
    return {
        "n": sum(len(scores.ids[b]) for b in benchmarks),
        "accuracy": interval(acc),
        "nll_raw": interval(nll_raw),
        "n_nll_clipped": sum(
            int((scores.raw[b].nll >= -np.log(NLL_EPS)).sum()) for b in benchmarks
        ),
        "nll_cal": interval(nll_cal),
        "brier_raw": interval(brier_raw),
        "brier_cal": interval(brier_cal),
        "ece_raw": float(
            np.mean([scores.raw[b].estimate(n_bins=N_BINS)["ece"] for b in benchmarks])
        ),
        "ece_cal": float(
            np.mean([scores.cal[b].estimate(n_bins=N_BINS)["ece"] for b in benchmarks])
        ),
        "qps": sent / wall if wall else None,
        "speed_split": spd.split,
        "in_flight": sorted({c for u in usage for c in u.concurrency}),
        "latency_p50_ms": float(np.median(latencies)) if len(latencies) else None,
        "latency_mean_ms": float(np.mean(latencies)) if len(latencies) else None,
        "cost_per_1k_items": None if cost is None else cost / n_priced * 1000.0,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_per_question": (
            None if tokens_in is None or tokens_out is None else tokens_in + tokens_out
        ),
    }


def paired_columns(a: Scores, b: Scores, benchmarks: Sequence[str]) -> dict[str, FloatArray]:
    """Per benchmark, ``a - b`` per shared item: accuracy, calibrated NLL, calibrated Brier. An
    item only one of them answered must be declared unanswered by the other."""
    columns = {}
    for bench in benchmarks:
        where_b = {item_id: k for k, item_id in enumerate(b.ids[bench])}
        pairs = [(k, where_b[i]) for k, i in enumerate(a.ids[bench]) if i in where_b]
        ids_a, ids_b = set(a.ids[bench]), set(b.ids[bench])
        if not (ids_a - ids_b <= b.unanswered and ids_b - ids_a <= a.unanswered):
            fail(f"{bench}: the runs answered different items")
        ia = np.array([p[0] for p in pairs])
        ib = np.array([p[1] for p in pairs])
        columns[bench] = np.column_stack(
            [
                a.raw[bench].correct[ia].astype(np.float64)
                - b.raw[bench].correct[ib].astype(np.float64),
                a.cal[bench].nll[ia] - b.cal[bench].nll[ib],
                a.cal[bench].brier[ia] - b.cal[bench].brier[ib],
            ]
        )
    return columns


def paired_delta(a: Scores, b: Scores, benchmarks: Sequence[str]) -> dict[str, list[float]]:
    """``a - b`` on shared item ids: accuracy, calibrated NLL and calibrated Brier."""
    columns = paired_columns(a, b, benchmarks)
    acc, nll, brier = macro_intervals(columns, n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES, seed=SEED)
    return {"accuracy": interval(acc), "nll_cal": interval(nll), "brier_cal": interval(brier)}


def accuracy_lead(a: Scores, b: Scores, benchmarks: Sequence[str]) -> dict[str, Any]:
    """How much of the paired macro accuracy ``a - b`` rests on :data:`GPQA`: its share of the
    macro difference, the macro without it, and the item-weighted difference (every question
    weighs the same; stratified bootstrap, each benchmark scaled by its share of the items)."""
    columns = {k: v[:, :1] for k, v in paired_columns(a, b, benchmarks).items()}
    macro = float(np.mean([v.mean() for v in columns.values()]))
    rest = {k: v for k, v in columns.items() if k != GPQA}
    n = sum(len(v) for v in columns.values())
    weighted = {k: v * (len(columns) * len(v) / n) for k, v in columns.items()}
    (without,) = macro_intervals(rest, n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES, seed=SEED)
    (pooled,) = macro_intervals(weighted, n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES, seed=SEED)
    gpqa = float(columns[GPQA].mean()) / len(columns) if GPQA in columns else 0.0
    return {
        "gpqa_share": gpqa / macro if macro else None,
        "without_gpqa": interval(without),
        "item_weighted": interval(pooled),
    }


def bins(metrics: Iterable[ItemMetrics], width: float) -> list[dict[str, float]]:
    """Equal-width reliability bins over top-label confidence, those with at least
    :data:`MIN_BIN_ITEMS` questions; a confidence of exactly 1 falls in the last bin."""
    listed = list(metrics)
    confidence = np.concatenate([m.confidence for m in listed])
    correct = np.concatenate([m.correct for m in listed])
    n_bins = round(1 / width)
    index = np.minimum(np.floor(confidence * n_bins).astype(np.int64), n_bins - 1)
    out = []
    for k in range(n_bins):
        inside = index == k
        n = int(inside.sum())
        if n >= MIN_BIN_ITEMS:
            out.append(
                {
                    "n": n,
                    "low": k / n_bins,
                    "high": (k + 1) / n_bins,
                    "confidence": float(confidence[inside].mean()),
                    "accuracy": float(correct[inside].mean()),
                }
            )
    return out


def benchmark_row(scores: Scores, benchmark: str) -> dict[str, Any]:
    raw = scores.raw[benchmark].estimate(n_bins=N_BINS)
    cal = scores.cal[benchmark].estimate(n_bins=N_BINS)
    return {
        "n": len(scores.ids[benchmark]),
        "accuracy": raw["accuracy"],
        "brier_raw": raw["brier"],
        "brier_cal": cal["brier"],
        "ece_raw": raw["ece"],
        "ece_cal": cal["ece"],
    }


def frontier(
    points: Sequence[tuple[float, float]],
    *,
    higher_is_better: bool,
    x_higher_is_better: bool = True,
) -> list[int]:
    """Indices of the points no other point beats on both x (q/s: higher is better; tokens per
    question: lower) and quality, best x first."""
    sx = 1.0 if x_higher_is_better else -1.0
    sy = 1.0 if higher_is_better else -1.0
    order = sorted(range(len(points)), key=lambda i: (-sx * points[i][0], -sy * points[i][1]))
    kept: list[int] = []
    best = -math.inf
    for i in order:
        quality = sy * points[i][1]
        if quality > best:
            kept.append(i)
            best = quality
    return kept


# --- figures (Plotly JSON) ---------------------------------------------------------------------

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Roboto, Helvetica, Arial, sans-serif"


def base_layout(**overrides: Any) -> dict[str, Any]:
    layout: dict[str, Any] = {
        "font": {"family": FONT, "size": 13, "color": "#1f2328"},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "margin": {"l": 70, "r": 20, "t": 70, "b": 60},
        "hoverlabel": {"font": {"family": FONT}},
        "modebar": {"orientation": "v"},  # keeps clear of the button rows
        "legend": {"orientation": "h", "x": 0, "y": -0.18, "font": {"size": 12}},
    }
    layout.update(overrides)
    return layout


def axis(title: str, **extra: Any) -> dict[str, Any]:
    return {
        "title": {"text": title},
        "gridcolor": "#eaeef2",
        "zeroline": False,
        "linecolor": "#d0d7de",
        "ticks": "outside",
        "tickcolor": "#d0d7de",
        **extra,
    }


def menu(buttons: Sequence[dict[str, Any]], *, y: float) -> dict[str, Any]:
    return {
        "type": "buttons",
        "direction": "right",
        "x": 0,
        "xanchor": "left",
        "y": y,
        "yanchor": "bottom",
        "pad": {"r": 4, "t": 0},
        "showactive": True,
        "bgcolor": "#f6f8fa",
        "bordercolor": "#d0d7de",
        "font": {"size": 12},
        "buttons": list(buttons),
    }


XMODES: tuple[tuple[str, str, str], ...] = (
    ("qps", "Questions per second (log scale)", "x: questions per second"),
    ("tokens", "Tokens per question (log scale; prompt + generated)", "x: tokens per question"),
)
"""(mode, axis title, button label) of the speed figures' x axis; the first is the default."""
X_KEYS = ("x", "y", "customdata", "hovertemplate", "text", "textposition", "line.shape")
"""Trace attributes that differ between the x modes."""


def attribute(trace: Mapping[str, Any], key: str) -> Any:
    """A (dotted) trace attribute, ``None`` if unset."""
    value: Any = trace
    for part in key.split("."):
        value = value.get(part) if isinstance(value, Mapping) else None
    return value


@dataclass
class Figure:
    traces: list[dict[str, Any]]
    states: list[str]
    state_of: list[int | None]
    """Trace -> the state it belongs to (``None``: always visible)."""
    alternates: list[dict[str, Any] | None]
    """Trace -> its :data:`X_KEYS` under the second x mode (``None``: the same in both)."""

    @classmethod
    def empty(cls, states: Sequence[str]) -> Figure:
        return cls([], list(states), [], [])

    def add(
        self, trace: dict[str, Any], state: int | None, alternate: dict[str, Any] | None = None
    ) -> None:
        self.traces.append(trace)
        self.state_of.append(state)
        self.alternates.append(alternate)

    def add_modes(self, make: Callable[[str], dict[str, Any]], state: int | None) -> None:
        """Add the trace ``make(mode)`` builds for the first x mode, remembering the second's."""
        second = make(XMODES[1][0])
        self.add(make(XMODES[0][0]), state, {key: attribute(second, key) for key in X_KEYS})

    def build(self, layout: dict[str, Any]) -> dict[str, Any]:
        for trace, state in zip(self.traces, self.state_of, strict=True):
            trace["visible"] = state is None or state == 0
        menus = []
        with_modes = any(a is not None for a in self.alternates)
        top = 1.2 if with_modes else 1.12
        if len(self.states) > 1:
            buttons = [
                {
                    "label": label,
                    "method": "restyle",
                    "args": [{"visible": [s is None or s == k for s in self.state_of]}],
                }
                for k, label in enumerate(self.states)
            ]
            menus.append(menu(buttons, y=top))
        if with_modes:
            buttons = []
            for index, (_, title, label) in enumerate(XMODES):
                values = {
                    key: [
                        attribute(trace, key) if index == 0 or alternate is None else alternate[key]
                        for trace, alternate in zip(self.traces, self.alternates, strict=True)
                    ]
                    for key in X_KEYS
                }
                relayout = {"xaxis.title.text": title, "xaxis.autorange": True}
                buttons.append({"label": label, "method": "update", "args": [values, relayout]})
            menus.append(menu(buttons, y=1.1))
        if menus:
            layout = {**layout, "updatemenus": menus}
        return {"data": self.traces, "layout": layout}


@dataclass(frozen=True)
class Point:
    label: str
    x: float
    y: list[float]
    """Estimate, low, high."""
    note: str


X_HOVER = {"qps": "%{x:.2f} q/s", "tokens": "%{x:,.0f} tokens per question"}


def x_value(summary: Mapping[str, Any], mode: str) -> float | None:
    value: float | None = summary["qps" if mode == "qps" else "tokens_per_question"]
    return value


def x_note(summary: Mapping[str, Any], mode: str) -> str:
    """Hover note on what the x value measures."""
    if mode == "qps":
        note: str = summary["speed_note"]
        return note
    return (
        f"{summary['tokens_in']:,.0f} prompt + {summary['tokens_out']:,.0f} generated tokens "
        "(own tokenizer)"
    )


def points_trace(
    name: str,
    points: Sequence[Point],
    *,
    color: str,
    symbol: str = "circle",
    size: int = 11,
    hollow: bool = False,
    text_position: str | None = None,  # None: no text label; else a Plotly textposition
    mode: str,
    value: str,
    digits: int,
    group: str,
    showlegend: bool = True,
) -> dict[str, Any]:
    fmt = f".{digits}f"
    return {
        "type": "scatter",
        "mode": "markers" if text_position is None else "markers+text",
        "name": name,
        "legendgroup": group,
        "showlegend": showlegend,
        "x": [p.x for p in points],
        "y": [p.y[0] for p in points],
        "error_y": {
            "type": "data",
            "symmetric": False,
            "array": [p.y[2] - p.y[0] for p in points],
            "arrayminus": [p.y[0] - p.y[1] for p in points],
            "thickness": 1.2,
            "width": 0,
            "color": color,
        },
        "text": None if text_position is None else [p.label for p in points],
        "textposition": text_position or "top center",
        "textfont": {"size": 12, "color": color},
        "cliponaxis": False,
        "customdata": [[p.label, p.y[1], p.y[2], p.note] for p in points],
        "hovertemplate": (
            f"<b>%{{customdata[0]}}</b><br>{value} %{{y:{fmt}}} "
            f"[%{{customdata[1]:{fmt}}}, %{{customdata[2]:{fmt}}}]"
            f"<br>{X_HOVER[mode]}<br>%{{customdata[3]}}<extra></extra>"
        ),
        "marker": {
            "color": "white" if hollow else color,
            "size": size,
            "symbol": symbol,
            "line": {"color": color if hollow else "white", "width": 2 if hollow else 1},
        },
    }


def frontier_line(points: Sequence[Point], *, higher_is_better: bool, mode: str) -> list[Point]:
    """The Pareto frontier of ``points``, by x."""
    kept = frontier(
        [(p.x, p.y[0]) for p in points],
        higher_is_better=higher_is_better,
        x_higher_is_better=mode == "qps",
    )
    return sorted((points[i] for i in kept), key=lambda p: p.x)


def frontier_corners(line: Sequence[Point], mode: str) -> list[tuple[float, float]]:
    """The (x, y) vertices of the drawn frontier: a step at each frontier point (Plotly's line
    shape "vh" on the q/s axis, "hv" on the tokens axis)."""
    corners = []
    for a, b in itertools.pairwise(line):
        corner = (a.x, b.y[0]) if mode == "qps" else (b.x, a.y[0])
        corners += [(a.x, a.y[0]), corner]
    return corners + [(p.x, p.y[0]) for p in line[-1:]]


def frontier_trace(line: Sequence[Point], *, mode: str) -> dict[str, Any]:
    return {
        "type": "scatter",
        "mode": "lines",
        "name": "Pareto frontier (local)",
        "legendgroup": "frontier",
        "x": [p.x for p in line],
        "y": [p.y[0] for p in line],
        # the best quality reachable at a given x: a step at each frontier point
        "line": {
            "color": "#8c959f",
            "width": 1.5,
            "dash": "dot",
            "shape": "vh" if mode == "qps" else "hv",
        },
        "hoverinfo": "skip",
    }


# --- page data ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class LandscapeEntry:
    key: str
    label: str
    group: str
    color: str
    kind: Kind
    run: Path
    speed: tuple[Path, SplitName]
    note: str
    finalist: str | None
    """Key of the finalist this point is (highlighted), if any."""
    max_rpm: int | None = None
    limits: ServiceLimits | None = None
    one_call_checked: bool = True
    """Its records carry diagnostics, so :func:`check_one_call` applies. Not for :data:`CLM`,
    which answers Jev's wire format through :class:`jevemu.clm_client.ClmClient` (one request per
    question, records without diagnostics, as Jev's)."""


CLM_RUN = _run("clm")
CLM = LandscapeEntry(
    key="clm",
    label="CLM-8B (local)",
    group="Dual encoder",
    color="#cf222e",
    kind="local",
    run=CLM_RUN,
    speed=(CLM_RUN, LANDSCAPE_SPLIT),
    note="A dual encoder, not an emulator: Qwen3-8B embeddings scored by a 20M-parameter head, "
    "answering the same wire format as Jev, one RTX 3090",
    finalist=None,
    one_call_checked=False,
)
"""CLM-8B (https://github.com/Contrastive-LM/CLM), a landscape point only: it answers Jev's
``POST /v1/systemone`` wire format, but it is not an emulator candidate, so it stays out of the
local models' frontier and counts (:func:`is_emulator_candidate`)."""


def is_emulator_candidate(entry: Mapping[str, Any]) -> bool:
    """A local model of the speed/quality registry (grouped by size class), unlike :data:`CLM`."""
    return entry["kind"] == "local" and entry["group"] in SIZE_CLASS_COLORS


def landscape_entries(ready: Sequence[Finalist]) -> list[LandscapeEntry]:
    registry = json.loads((REPO / SPEED_QUALITY).read_text())
    finalist_by_preset = {f.preset: f for f in ready if f.preset}
    entries = []
    for candidate in registry["candidates"]:
        finalist = finalist_by_preset.get(candidate["preset"])
        run_dir = Path(candidate["run_dir"])
        entries.append(
            LandscapeEntry(
                key=candidate["preset"],
                label=finalist.label if finalist else candidate["label"],
                group=candidate["size_class"],
                color=finalist.color if finalist else SIZE_CLASS_COLORS[candidate["size_class"]],
                kind="local",
                run=run_dir,
                speed=(run_dir, LANDSCAPE_SPLIT),
                note=f"{candidate['params']}, {candidate['weights']}, one RTX 3090",
                finalist=finalist.key if finalist else None,
            )
        )
    missing = [f.label for f in ready if f.preset and f.preset not in {e.key for e in entries}]
    if missing:
        fail(f"{SPEED_QUALITY} lists no candidate for {missing}")
    for f in ready:
        if f.kind != "local":
            entries.append(
                LandscapeEntry(
                    key=f.key,
                    label=f.label,
                    group="API",
                    color=f.color,
                    kind=f.kind,
                    run=f.run,
                    speed=f.landscape_speed,
                    note=f.detail,
                    finalist=f.key,
                    max_rpm=f.max_rpm,
                    limits=f.limits,
                )
            )
    done = complete_benchmarks(CLM.run, LANDSCAPE_SPLIT)
    if done != set(DATASETS):
        fail(f"{CLM.label}: {LANDSCAPE_SPLIT} runs incomplete ({len(done)}/{len(DATASETS)})")
    return [*entries, CLM]


def speed_note(kind: Kind, summary: Mapping[str, Any], max_rpm: int | None) -> str:
    """Hover/table note on what a q/s measures."""
    in_flight = "/".join(str(c) for c in summary["in_flight"]) or "?"
    if kind == "local":
        return f"{in_flight} questions in flight, {LOCAL_SETUP}"
    qps = summary["qps"]
    if max_rpm and qps is not None and qps >= LIMITED_SHARE * max_rpm / 60.0:
        return (
            f"client-limited: capped by our limiter at {max_rpm:,} requests/min "
            f"({in_flight} in flight)"
        )
    return f"{in_flight} requests in flight over the internet"


def ceiling(limits: ServiceLimits | None, summary: Mapping[str, Any]) -> dict[str, Any] | None:
    """The q/s an API service could reach without our client's request limiter: the tightest of
    its own request and token limits and, unless it batches questions, the in-flight bound
    (Little's law: requests in flight over the measured mean latency). ``in_flight`` is set when
    that last bound binds: it is our client's setting, not a limit of the service."""
    if limits is None:
        return None
    caps: list[tuple[float, str]] = []
    tpq = summary["tokens_per_question"]
    if limits.tokens and tpq:
        count, unit = limits.tokens
        per_second = count / (1.0 if unit == "s" else 60.0)
        caps.append(
            (per_second / tpq, f"{count:,} tokens/{unit} at {tpq:,.0f} tokens per question")
        )
    in_flight_cap: tuple[float, str] | None = None
    if not limits.batched:
        if limits.requests_per_min:
            caps.append(
                (
                    limits.requests_per_min / 60.0,
                    f"{limits.requests_per_min:,} requests/min, one question per request",
                )
            )
        mean_ms, in_flight = summary["latency_mean_ms"], summary["in_flight"]
        if mean_ms and len(in_flight) == 1:
            in_flight_cap = (
                in_flight[0] * 1000.0 / mean_ms,
                f"{in_flight[0]} requests in flight at {mean_ms:,.0f} ms mean latency",
            )
            caps.append(in_flight_cap)
    if not caps:
        return None
    caps.sort()
    return {
        "qps": caps[0][0],
        "binding": caps[0][1],
        "in_flight": summary["in_flight"][0] if caps[0] == in_flight_cap else None,
        "caps": [{"qps": q, "why": why} for q, why in caps],
        "source": limits.source,
    }


def collect(prices: PriceBook) -> dict[str, Any]:
    status = {f.key: readiness(f) for f in FINALISTS}
    for f in FINALISTS:
        if status[f.key] is not None and not f.in_progress:
            fail(f"{f.label}: runs incomplete: {status[f.key]}")
    ready = [f for f in FINALISTS if status[f.key] is None]
    pending = [
        {"key": f.key, "label": f.label, "detail": f.detail, "progress": status[f.key]}
        for f in FINALISTS
        if status[f.key] is not None
    ]

    final_scores = {f.key: score(f.run, FINAL_SPLIT, f.unanswered_on(FINAL_SPLIT)) for f in ready}
    final_speed = {f.key: speed(*f.speed, prices) for f in ready}
    entries = landscape_entries(ready)
    checked_runs = [
        *[run for f in ready if f.kind != "jev" for run in ((f.run, FINAL_SPLIT), f.speed)],
        *[(e.run, LANDSCAPE_SPLIT) for e in entries if e.kind != "jev" and e.one_call_checked],
    ]
    declared = {f.run: f for f in ready}
    for run_dir, split in dict.fromkeys(checked_runs):
        f_run = declared.get(run_dir)
        check_one_call(run_dir, split, f_run.unanswered_on(split) if f_run else frozenset())
    land_scores = {e.key: score(e.run, LANDSCAPE_SPLIT) for e in entries}
    land_speed = {e.key: speed(*e.speed, prices) for e in entries}

    answered = [set(s.benchmarks) for s in [*final_scores.values(), *land_scores.values()]]
    common = [b for b in DATASETS if all(b in a for a in answered)]
    views = {
        "common": {
            "label": f"{len(common)} benchmarks every system answers",
            "short": f"{len(common)} shared benchmarks",
            "benchmarks": common,
        },
        "all": {
            "label": f"all {len(DATASETS)} benchmarks",
            "short": f"all {len(DATASETS)} benchmarks",
            "benchmarks": list(DATASETS),
        },
    }

    def members(scored: Mapping[str, Scores], view: str) -> list[str]:
        wanted = set(views[view]["benchmarks"])
        return [k for k, s in scored.items() if wanted <= set(s.benchmarks)]

    finalists: list[dict[str, Any]] = []
    for f in ready:
        s, spd = final_scores[f.key], final_speed[f.key]
        row: dict[str, Any] = {
            "key": f.key,
            "label": f.label,
            "kind": f.kind,
            "color": f.color,
            "detail": f.detail,
            "run": str(f.run),
            "speed_run": [str(f.speed[0]), f.speed[1]],
            "benchmarks": s.benchmarks,
            "unanswered": unanswered_counts(f),
            "views": {},
            "per_benchmark": {b: benchmark_row(s, b) for b in s.benchmarks},
            "reliability": {
                "by_benchmark": {
                    b: {
                        "raw": bins([s.raw[b]], BENCHMARK_BIN_WIDTH),
                        "cal": bins([s.cal[b]], BENCHMARK_BIN_WIDTH),
                    }
                    for b in s.benchmarks
                }
            },
        }
        for view in views:
            if f.key not in members(final_scores, view):
                continue
            chosen = views[view]["benchmarks"]
            summary = view_summary(s, spd, chosen)
            summary["speed_note"] = speed_note(f.kind, summary, f.max_rpm)
            summary["ceiling"] = ceiling(f.limits, summary)
            summary["delta_vs"] = {
                other: paired_delta(s, final_scores[other], chosen)
                for other in members(final_scores, view)
                if other != f.key
            }
            if f.kind == "jev":
                summary["accuracy_lead"] = {
                    other: accuracy_lead(s, final_scores[other], chosen)
                    for other in members(final_scores, view)
                    if other != f.key
                }
            if f.kind == "api":
                summary["read"] = api_read(f.run, FINAL_SPLIT, chosen)
            row["views"][view] = summary
            row["reliability"][view] = {
                "raw": bins((s.raw[b] for b in chosen), POOLED_BIN_WIDTH),
                "cal": bins((s.cal[b] for b in chosen), POOLED_BIN_WIDTH),
            }
        finalists.append(row)

    landscape: list[dict[str, Any]] = []
    for e in entries:
        s, spd = land_scores[e.key], land_speed[e.key]
        item: dict[str, Any] = {
            "key": e.key,
            "label": e.label,
            "group": e.group,
            "color": e.color,
            "kind": e.kind,
            "finalist": e.finalist,
            "note": e.note,
            "run": str(e.run),
            "speed_run": [str(e.speed[0]), e.speed[1]],
            "views": {},
        }
        for view in views:
            if e.key in members(land_scores, view):
                summary = view_summary(s, spd, views[view]["benchmarks"])
                summary["speed_note"] = speed_note(e.kind, summary, e.max_rpm)
                summary["ceiling"] = ceiling(e.limits, summary)
                item["views"][view] = summary
        if e.kind == "jev":
            # reports/speed_quality scores Jev by its returned choice, this page by the argmax
            registry = json.loads((REPO / SPEED_QUALITY).read_text())
            item["choice_accuracy_speed_quality"] = registry["jev"]["accuracy"][0]
        landscape.append(item)

    return {
        "config": {
            "calibrator": CALIBRATOR,
            "folds": FOLDS,
            "seed": SEED,
            "ece_bins": N_BINS,
            "reliability_bin_width": {"pooled": POOLED_BIN_WIDTH, "benchmark": BENCHMARK_BIN_WIDTH},
            "reliability_min_items": MIN_BIN_ITEMS,
            "bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
            "landscape_split": LANDSCAPE_SPLIT,
            "final_split": FINAL_SPLIT,
            "dropped_benchmarks": list(DROPPED_DATASETS),
            "landscape_registry": str(SPEED_QUALITY),
        },
        "splits": split_sizes(views),
        "views": views,
        "finalists": finalists,
        "landscape": landscape,
        "pending": pending,
    }


# --- figures from page data --------------------------------------------------------------------


VIEW_ORDER = ("common", "all")


PointOf = Callable[[Mapping[str, Any], str], Point]
"""(system, x mode) -> its point in one figure state."""
MODES = tuple(mode for mode, _, _ in XMODES)


def speed_states(data: Mapping[str, Any], metric: str) -> list[tuple[str, str, str]]:
    """(view, field, button label) of an accuracy or Brier (raw and calibrated) speed figure."""
    arms = (
        [("accuracy", "")]
        if metric == "accuracy"
        else [("brier_raw", "raw"), ("brier_cal", "calibrated")]
    )
    return [
        (view, field, f"{data['views'][view]['short']}" + (f" · {arm}" if arm else ""))
        for view in VIEW_ORDER
        for field, arm in arms
    ]


def split_sizes(views: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Questions per split and view, counted from the frozen split files."""

    def count(split: SplitName, benchmarks: Sequence[str]) -> int:
        total = 0
        for benchmark in benchmarks:
            path = REPO / SPLITS_DIR / f"{benchmark}.{split}.jsonl"
            if not path.is_file():
                fail(f"missing frozen split {path}")
            total += FrozenSplit.from_jsonl(path).metadata.n_items
        return total

    return {
        split: {view: count(split, spec["benchmarks"]) for view, spec in views.items()}
        for split in PAGE_SPLITS
    }


def speed_layout(
    height: int, higher: bool, plot: tuple[float, float], split: SplitName
) -> dict[str, Any]:
    y_title = (
        f"Macro accuracy on {split} (higher is better)"
        if higher
        else f"Macro Brier score on {split} (lower is better)"
    )
    layout = base_layout(
        height=height,
        margin={"l": 70, "r": 20, "t": 110, "b": 60},
        xaxis=axis(XMODES[0][1], type="log"),
        yaxis=axis(y_title, tickformat=".0%" if higher else ".2f"),
    )
    layout["legend"] = {**layout["legend"], "y": -LEGEND_DROP_PX / plot[1]}
    return layout


def point_maker(view: str, field: str, *, with_note: bool) -> PointOf:
    def point(system: Mapping[str, Any], mode: str) -> Point:
        v = system["views"][view]
        x = x_value(v, mode)
        if x is None:
            fail(f"{system['label']}: no {mode} for {view}")
        note = x_note(v, mode)
        return Point(
            system["label"], x, v[field], f"{system['note']}<br>{note}" if with_note else note
        )

    return point


def has_x(system: Mapping[str, Any], view: str) -> bool:
    v = system["views"].get(view)
    return v is not None and all(x_value(v, mode) for mode in MODES)


def landscape_figure(data: Mapping[str, Any], metric: str) -> dict[str, Any]:
    """Accuracy (``metric="accuracy"``) or Brier (raw and calibrated) against q/s or tokens per
    question, one point per system, local candidates grouped by size class, CLM labeled apart
    from them (outside their frontier)."""
    states = speed_states(data, metric)
    fig = Figure.empty([label for _, _, label in states])
    higher = metric == "accuracy"
    value = "accuracy" if higher else "Brier"
    for k, (view, field, _) in enumerate(states):
        entries = [e for e in data["landscape"] if has_x(e, view)]
        point = point_maker(view, field, with_note=True)
        local = [e for e in entries if is_emulator_candidate(e)]
        lines = {
            mode: frontier_line([point(e, mode) for e in local], higher_is_better=higher, mode=mode)
            for mode in MODES
        }

        def frontier_of(mode: str, _lines: dict[str, list[Point]] = lines) -> Any:
            return frontier_trace(_lines[mode], mode=mode)

        fig.add_modes(frontier_of, k)
        for group, color in SIZE_CLASS_COLORS.items():
            members = [e for e in local if e["group"] == group and not e["finalist"]]
            if not members:
                continue

            def group_of(
                mode: str,
                _members: list[Any] = members,
                _group: str = group,
                _color: str = color,
                _point: PointOf = point,
            ) -> Any:
                return points_trace(
                    _group,
                    [_point(e, mode) for e in _members],
                    color=_color,
                    mode=mode,
                    value=value,
                    digits=3,
                    group=_group,
                )

            fig.add_modes(group_of, k)
        highlighted = [e for e in entries if e["finalist"] or e["key"] == CLM.key]
        corners = {mode: frontier_corners(line, mode) for mode, line in lines.items()}
        add_labeled(
            fig, k, highlighted, entries, point, 13, value, view, LANDSCAPE_PLOT_PX, corners
        )
    return fig.build(speed_layout(580, higher, LANDSCAPE_PLOT_PX, LANDSCAPE_SPLIT))


def finalists_figure(data: Mapping[str, Any], metric: str) -> dict[str, Any]:
    states = speed_states(data, metric)
    fig = Figure.empty([label for _, _, label in states])
    higher = metric == "accuracy"
    value = "accuracy" if higher else "Brier"
    for k, (view, field, _) in enumerate(states):
        shown = [f for f in data["finalists"] if has_x(f, view)]
        point = point_maker(view, field, with_note=False)
        add_labeled(fig, k, shown, shown, point, 15, value, view, FINALISTS_PLOT_PX)
    return fig.build(speed_layout(520, higher, FINALISTS_PLOT_PX, FINAL_SPLIT))


LABEL_PX = (6.6, 15.0, 8.0, 3.0, 7.0)
"""Approximate size of a 12 px marker label: width of one character, line height, its gap from
the marker center, the clearance kept around a placed label, and the marker radius Plotly pads
the x range with (pixels)."""
LANDSCAPE_PLOT_PX = (822.0, 311.0)
FINALISTS_PLOT_PX = (822.0, 280.0)
"""Plot area of the landscape and finalists speed figures (pixels), as Plotly lays them out in
the page's 912 px column with the legend of the 6-benchmark view (the tallest legend, so the
smallest plot area), for label placement."""
LEGEND_DROP_PX = 60.0
"""Distance from the bottom of a speed figure's plot area to its legend (pixels): clear of the
two rows of log-axis tick labels and the axis title under them."""
AUTORANGE_PAD = 0.05
"""Share of the plot length Plotly's autorange leaves empty beyond the data on each side."""
LABEL_POSITIONS = (
    "top center",
    "bottom center",
    "middle right",
    "middle left",
    "top right",
    "top left",
    "bottom right",
    "bottom left",
)
"""Plotly text positions a label may take, in order of preference."""
HEAD_POSITIONS = (
    "middle right",
    "top center",
    "bottom center",
    "top right",
    "bottom right",
    "top left",
    "bottom left",
)
"""Text positions of the value at a ceiling arrow's head, in order of preference."""
LEADER_DIRECTIONS = (
    ("top center", 0.0, 1.0),
    ("bottom center", 0.0, -1.0),
    ("middle right", 1.0, 0.0),
    ("middle left", -1.0, 0.0),
    ("top right", math.sqrt(0.5), math.sqrt(0.5)),
    ("top left", -math.sqrt(0.5), math.sqrt(0.5)),
    ("bottom right", math.sqrt(0.5), -math.sqrt(0.5)),
    ("bottom left", -math.sqrt(0.5), -math.sqrt(0.5)),
    ("top right", math.sqrt(0.75), 0.5),
    ("top left", -math.sqrt(0.75), 0.5),
    ("bottom right", math.sqrt(0.75), -0.5),
    ("bottom left", -math.sqrt(0.75), -0.5),
    ("top right", 0.5, math.sqrt(0.75)),
    ("top left", -0.5, math.sqrt(0.75)),
    ("bottom right", 0.5, -math.sqrt(0.75)),
    ("bottom left", -0.5, -math.sqrt(0.75)),
)
"""Directions a leader line may take from its point (text position at its far end, unit vector
in pixels, y up), in order of preference: the axes and the diagonals, then 30 degrees off the
x axis and off the y axis."""
LEADER_LENGTHS = (26.0, 42.0, 58.0, 80.0, 110.0, 150.0)
"""Lengths a leader line may have, from the point's center to the text (pixels)."""
LEADER_COST = 1.0
"""The cost of a leader line per pixel of its length, in square pixels of text over something:
a text moves away from its point only to get clear of something."""
LEADER_GAP = 3.0
"""Space between a leader line's end and its text (pixels)."""
STROKE_PX = 6.0
"""Width a line (error bar, frontier, arrow, leader) counts with: a text crossed by one or right
next to it is hard to read (pixels)."""
TEXT_OVERLAP_WEIGHT = 3.0
"""How much worse text over text or over a leader line, text or a leader line over a marker, a
labeled system's error bar or a ceiling arrow (what the figure is about), or text past the plot
area's edge (onto tick labels or the menus) is than text or a leader line over another error bar
or the frontier."""

Box = tuple[float, float, float, float]
"""x0, x1, y0, y1 in pixels."""
XY = tuple[float, float]


@dataclass(frozen=True)
class Placement:
    """Where a text goes: at ``position`` next to its point or, with a ``leader`` (start, end and
    text anchor, in data coordinates), at ``position`` of the anchor, which a thin line from next
    to the point to just short of the text connects to the point."""

    position: str
    leader: tuple[XY, XY, XY] | None = None


def overlap(a: Box, b: Box) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[2], b[2]))


def hull(boxes: Sequence[Box]) -> Box:
    return (
        min(b[0] for b in boxes),
        max(b[1] for b in boxes),
        min(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


@dataclass(frozen=True)
class Candidate:
    """One placement of a text with the length of its leader line (0 without one), its
    estimated text box, the small boxes tracing the leader line and the hull of both, in
    pixels."""

    placement: Placement
    length: float
    box: Box
    line: tuple[Box, ...]
    extent: Box


def label_positions(
    labeled: Sequence[Point],
    every: Sequence[Point],
    plot: tuple[float, float],
    arrows: Sequence[tuple[Point, Point]] = (),
    polyline: Sequence[tuple[float, float]] = (),
) -> tuple[list[Placement], list[Placement]]:
    """Placements for the labels of ``labeled`` and for the value text at the head of each
    ceiling arrow (from a point to its head, whose label is that text), on a plot area of
    ``plot`` pixels: next to the point at one of :data:`LABEL_POSITIONS` (:data:`HEAD_POSITIONS`
    for heads), or at the end of a leader line of one of :data:`LEADER_LENGTHS` in one of
    :data:`LEADER_DIRECTIONS` (costing :data:`LEADER_COST` per pixel, and ending inside the data's
    extent so that Plotly's autorange stays as estimated). The chosen combination costs least:
    its leader lengths plus how much its estimated text boxes and leader lines cover of the other
    error bars and of the axis-aligned ``polyline`` (data coordinates) and (weighted by
    :data:`TEXT_OVERLAP_WEIGHT`) of each other, of the markers, of the error bars of ``labeled``,
    of the arrows and of the outside of the plot area. Ties go to the combination found first:
    texts from the highest label, each's placements from the cheapest alone (then in order of
    preference)."""
    heads = [head for _, head in arrows]
    xs = [math.log10(p.x) for p in (*every, *heads)]
    ys = [v for p in every for v in (p.y[1], p.y[2])]
    x_lo, x_span = min(xs), (max(xs) - min(xs)) or 1.0
    y_lo, y_span = min(ys), (max(ys) - min(ys)) or 1.0
    width, height = plot
    char, line, gap, pad, radius = LABEL_PX
    x_pad, y_pad = AUTORANGE_PAD * width + radius, AUTORANGE_PAD * height

    def px(x: float, y: float) -> XY:
        return (
            x_pad + (math.log10(x) - x_lo) / x_span * (width - 2 * x_pad),
            y_pad + (y - y_lo) / y_span * (height - 2 * y_pad),
        )

    def data(x: float, y: float) -> XY:
        return (
            10 ** (x_lo + (x - x_pad) / (width - 2 * x_pad) * x_span),
            y_lo + (y - y_pad) / (height - 2 * y_pad) * y_span,
        )

    def text_box(x: float, y: float, text: str, position: str, offset: float) -> Box:
        w = char * len(text)
        vertical, horizontal = position.split()
        x0 = {"center": x - w / 2, "right": x + offset, "left": x - offset - w}[horizontal]
        y0 = {"top": y + offset, "middle": y - line / 2, "bottom": y - offset - line}[vertical]
        return (x0, x0 + w, y0, y0 + line)

    def stroke(x0: float, x1: float, y0: float, y1: float) -> Box:
        half = STROKE_PX / 2
        return (x0 - half, x1 + half, y0 - half, y1 + half)

    def candidates(p: Point, positions: Sequence[str]) -> list[Candidate]:
        x, y = px(p.x, p.y[0])
        found = []
        for position in positions:
            b = text_box(x, y, p.label, position, gap)
            found.append(Candidate(Placement(position), 0.0, b, (), b))
        for length in LEADER_LENGTHS:
            for position, ux, uy in LEADER_DIRECTIONS:
                ax, ay = x + ux * length, y + uy * length
                if not (x_pad <= ax <= width - x_pad and y_pad <= ay <= height - y_pad):
                    continue  # a text anchored past the data would widen the autorange
                start, end = gap, length - LEADER_GAP
                # traced from 4 px past its start, clear of the box of the point's own marker
                steps = max(1, round((end - start - 4.0) / 3.0))
                ts = [start + 4.0 + (end - start - 4.0) * k / steps for k in range(steps + 1)]
                traced = tuple(stroke(x + ux * t, x + ux * t, y + uy * t, y + uy * t) for t in ts)
                b = text_box(ax, ay, p.label, position, 0.0)
                leader = (
                    data(x + ux * start, y + uy * start),
                    data(x + ux * end, y + uy * end),
                    data(ax, ay),
                )
                extent = hull([b, *traced])
                found.append(Candidate(Placement(position, leader), length, b, traced, extent))
        return found

    # a label never overlaps its own marker: every candidate box starts ``gap`` from its center
    def marker(p: Point) -> Box:
        x, y = px(p.x, p.y[0])
        return (x - gap, x + gap, y - gap, y + gap)

    # text over markers and over what the figure is about, the labeled systems' error bars and the
    # arrows, weighs like text over text; lines count :data:`STROKE_PX` wide, for the gap a text
    # needs
    at_labeled = {(p.x, p.y[0]) for p in labeled}
    salient: list[Box] = []
    obstacles: list[Box] = []
    for p in (*every, *heads):
        salient.append(marker(p))
        x = px(p.x, p.y[0])[0]
        low, high = px(p.x, p.y[1])[1], px(p.x, p.y[2])[1]
        bar = stroke(x, x, low, high)
        (salient if (p.x, p.y[0]) in at_labeled else obstacles).append(bar)
    for start, head in arrows:
        x0, y = px(start.x, start.y[0])
        x1 = px(head.x, head.y[0])[0]
        salient.append(stroke(x0 + gap, x1, y, y))
    for a, b in itertools.pairwise(px(x, y) for x, y in polyline):  # axis-aligned segments
        obstacles.append(stroke(min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1])))
    far = 1e6  # beyond the plot area: axis tick labels, the menus, the figure's edge
    edges: list[Box] = [
        (-far, 0.0, -far, far),
        (width, far, -far, far),
        (-far, far, -far, 0.0),
        (-far, far, height, far),
    ]

    def covered(b: Box, *, with_edges: bool) -> float:
        heavy = (*salient, *edges) if with_edges else salient
        return sum(overlap(b, o) for o in obstacles) + TEXT_OVERLAP_WEIGHT * sum(
            overlap(b, o) for o in heavy
        )

    def own_cost(c: Candidate) -> float:
        return (
            LEADER_COST * c.length
            + covered(c.box, with_edges=True)
            + sum(covered(d, with_edges=False) for d in c.line)
        )

    def padded(b: Box) -> Box:
        return (b[0] - pad, b[1] + pad, b[2] - pad, b[3] + pad)

    def between(a: Candidate, b: Candidate) -> float:
        """What placing both ``a`` and ``b`` costs on top of placing each alone."""
        if overlap(padded(a.extent), padded(b.extent)) == 0.0:
            return 0.0
        crossed = overlap(a.box, padded(b.box))
        crossed += sum(overlap(a.box, d) for d in b.line) + sum(overlap(d, b.box) for d in a.line)
        crossed += sum(overlap(d, e) for d in a.line for e in b.line)
        return TEXT_OVERLAP_WEIGHT * crossed

    texts = [*labeled, *heads]
    order = sorted(range(len(labeled)), key=lambda i: -labeled[i].y[0])
    order += range(len(labeled), len(texts))
    # each text's candidates cheapest first (ties in order of preference), so that the search
    # meets good combinations early and stops a text's candidates at the first that can't win
    options = []
    for k in order:
        row = candidates(texts[k], LABEL_POSITIONS if k < len(labeled) else HEAD_POSITIONS)
        options.append(sorted(((own_cost(c), c) for c in row), key=lambda fc: fc[0]))
    pairs: dict[tuple[int, int, int, int], float] = {}

    def pair(i: int, c: int, j: int, d: int) -> float:
        key = (i, c, j, d)
        if key not in pairs:
            pairs[key] = between(options[i][c][1], options[j][d][1])
        return pairs[key]

    # texts none of whose placements can interact are placed independently: the search over each
    # group of interacting texts is exact, and much smaller than over all texts at once
    reach = [padded(hull([c.extent for _, c in row])) for row in options]

    def interact(i: int, j: int) -> bool:
        return overlap(reach[i], reach[j]) > 0.0 and any(
            pair(i, c, j, d) > 0.0 for c in range(len(options[i])) for d in range(len(options[j]))
        )

    group_of = list(range(len(order)))
    for i in range(len(order)):
        for j in range(i):
            if group_of[i] != group_of[j] and interact(i, j):
                old, new = max(group_of[i], group_of[j]), min(group_of[i], group_of[j])
                group_of = [new if g == old else g for g in group_of]
    chosen = [0] * len(order)
    for group in sorted(set(group_of)):
        members = [i for i in range(len(order)) if group_of[i] == group]
        for i, c in zip(members, place(members, options, pair), strict=True):
            chosen[i] = c
    placements = [Placement("")] * len(texts)
    for i, k in enumerate(order):
        placements[k] = options[i][chosen[i]][1].placement
    return placements[: len(labeled)], placements[len(labeled) :]


def place(
    members: Sequence[int],
    options: Sequence[Sequence[tuple[float, Candidate]]],
    pair: Callable[[int, int, int, int], float],
) -> list[int]:
    """The cheapest choice among ``options`` (each text's candidates with their own cost,
    cheapest first) for the texts ``members``, the cost of a pair of choices given by ``pair``,
    found by branch and bound; ties go to the first found."""
    # the cheapest the members after the n-th could be placed, for pruning
    rest = [0.0] * (len(members) + 1)
    for n in reversed(range(len(members))):
        rest[n] = rest[n + 1] + options[members[n]][0][0]
    best: tuple[float, list[int]] = (math.inf, [])
    chosen: list[int] = []

    def search(partial: float) -> None:
        nonlocal best
        n = len(chosen)
        if n == len(members):
            best = (partial, list(chosen))
            return
        i = members[n]
        for c, (own, _) in enumerate(options[i]):
            total = partial + own
            if total + rest[n + 1] >= best[0]:
                break  # the candidates after ``c`` cost as much or more
            total += sum(pair(i, c, members[m], chosen[m]) for m in range(n))
            if total + rest[n + 1] < best[0]:
                chosen.append(c)
                search(total)
                chosen.pop()

    search(0.0)
    return best[1]


def wrap(text: str, width: int = 90) -> str:
    """Hover text broken into lines."""
    return "<br>".join(textwrap.wrap(text, width))


def ceiling_head(system: Mapping[str, Any], view: str, at: Point) -> Point | None:
    """The head of the ceiling arrow of ``system`` from its q/s point ``at``, labeled with the
    ceiling's q/s, when drawn: only when it is visibly above ``at``."""
    c: dict[str, Any] | None = system["views"][view].get("ceiling")
    if c is None or c["qps"] < SHOWN_CEILING_SHARE * at.x:
        return None
    return Point(f"\u2248{qps_text(c['qps'])}", c["qps"], [at.y[0]] * 3, "")


def leader_trace(
    name: str, group: str, color: str, text: str, size: int, placement: Callable[[str], Placement]
) -> Callable[[str], Any]:
    """``text`` at the end of its leader line, in the x modes whose ``placement`` has one."""

    def make(mode: str) -> Any:
        leader = placement(mode).leader
        ends = [] if leader is None else [leader[0], leader[1], (None, None), leader[2]]
        return {
            "type": "scatter",
            "mode": "lines+text",
            "name": name,
            "legendgroup": group,
            "showlegend": False,
            "x": [x for x, _ in ends],
            "y": [y for _, y in ends],
            "text": ["", "", "", text],
            "textposition": placement(mode).position,
            "textfont": {"size": size, "color": color},
            "cliponaxis": False,
            "line": {"color": color, "width": 0.8},
            "hoverinfo": "skip",
        }

    return make


def ceiling_traces(
    system: Mapping[str, Any], view: str, at: Point, head: Point, placement: Placement
) -> list[Callable[[str], Any]]:
    """A dotted arrow from a client-limited API point ``at`` to ``head``, the q/s its own limits
    would allow, with that q/s placed at ``placement`` (drawn only on the q/s axis)."""
    c = system["views"][view]["ceiling"]
    color, label = system["color"], system["label"]

    def arrow_line(mode: str) -> Any:
        shown = mode == "qps"
        return {
            "type": "scatter",
            "mode": "lines",
            "name": f"{label}: without our limiter",
            "legendgroup": system["key"],
            "showlegend": False,
            "x": [at.x, head.x] if shown else [],
            "y": [at.y[0], at.y[0]] if shown else [],
            "line": {"color": color, "width": 1.5, "dash": "dot"},
            "hoverinfo": "skip",
        }

    def arrow_head(mode: str) -> Any:
        shown = mode == "qps"
        return {
            "type": "scatter",
            "mode": "markers+text",
            "name": f"{label}: without our limiter",
            "legendgroup": system["key"],
            "showlegend": False,
            "x": [head.x] if shown else [],
            "y": [at.y[0]] if shown else [],
            "text": [head.label if placement.leader is None else ""],
            "textposition": placement.position,
            "textfont": {"size": 11, "color": color},
            "cliponaxis": False,
            "marker": {"symbol": "triangle-right", "size": 10, "color": color},
            "customdata": [[label, c["binding"], wrap(c["source"])]] if shown else [],
            "hovertemplate": (
                "<b>%{customdata[0]}</b> without our request limiter: "
                "\u2248%{x:.1f} q/s<br>bound by %{customdata[1]}<br>%{customdata[2]}"
                "<extra></extra>"
            ),
        }

    def placed(mode: str) -> Placement:
        return placement if mode == "qps" else Placement(placement.position)

    text = leader_trace(
        f"{label}: without our limiter", system["key"], color, head.label, 11, placed
    )
    return [arrow_line, text, arrow_head]


def add_labeled(
    fig: Figure,
    state: int,
    systems: Sequence[Mapping[str, Any]],
    every: Sequence[Mapping[str, Any]],
    point: PointOf,
    size: int,
    value: str,
    view: str,
    plot: tuple[float, float],
    polylines: Mapping[str, Sequence[tuple[float, float]]] | None = None,
) -> None:
    """One labeled trace per highlighted system (hollow for API services, a diamond for Jev),
    labels placed per x mode on a plot area of ``plot`` pixels, clear of that mode's line in
    ``polylines`` (the frontier), plus the ceiling arrow of client-limited API services."""
    arrows = {
        index: (at, head)
        for index, s in enumerate(systems)
        for at in [point(s, "qps")]
        for head in [ceiling_head(s, view, at)]
        if head is not None
    }
    placements = {
        mode: label_positions(
            [point(s, mode) for s in systems],
            [point(e, mode) for e in every],
            plot,
            list(arrows.values()) if mode == "qps" else (),
            polylines[mode] if polylines else (),
        )
        for mode in MODES
    }
    head_placements = dict(zip(arrows, placements["qps"][1], strict=True))
    for index, system in enumerate(systems):
        if index in arrows:
            at, head = arrows[index]
            for make in ceiling_traces(system, view, at, head, head_placements[index]):
                fig.add_modes(make, state)

        def placed(mode: str, _index: int = index) -> Placement:
            return placements[mode][0][_index]

        fig.add_modes(
            leader_trace(
                system["label"], system["key"], system["color"], system["label"], 12, placed
            ),
            state,
        )

        def labeled(mode: str, _system: Mapping[str, Any] = system, _index: int = index) -> Any:
            trace = points_trace(
                _system["label"],
                [point(_system, mode)],
                color=_system["color"],
                symbol="diamond" if _system["kind"] == "jev" else "circle",
                size=size,
                hollow=_system["kind"] != "local",
                text_position=placements[mode][0][_index].position,
                mode=mode,
                value=value,
                digits=3,
                group=_system["key"],
            )
            if placements[mode][0][_index].leader is not None:
                trace["text"] = [""]  # at the end of the leader line instead
            return trace

        fig.add_modes(labeled, state)


def reliability_trace(
    f: Mapping[str, Any],
    points: Sequence[Mapping[str, float]],
    *,
    xaxis: str,
    yaxis: str,
    showlegend: bool,
    where: str,
) -> dict[str, Any]:
    n_max = max(p["n"] for p in points)
    return {
        "type": "scatter",
        "mode": "lines+markers",
        "name": f["label"],
        "legendgroup": f["key"],
        "showlegend": showlegend,
        "xaxis": xaxis,
        "yaxis": yaxis,
        "x": [p["confidence"] for p in points],
        "y": [p["accuracy"] for p in points],
        "customdata": [[p["n"], p["low"], p["high"]] for p in points],
        "line": {"color": f["color"], "width": 1.5},
        "marker": {
            "color": f["color"],
            "size": [4 + 11 * math.sqrt(p["n"] / n_max) for p in points],
            "line": {"color": "white", "width": 0.5},
        },
        "hovertemplate": (
            f"<b>{html.escape(f['label'])}</b> · {html.escape(where)}<br>"
            "confidence %{customdata[1]:.0%} to %{customdata[2]:.0%} (mean %{x:.3f}), "
            "accuracy %{y:.3f}<br>%{customdata[0]:,} items<extra></extra>"
        ),
    }


def diagonal(xaxis: str, yaxis: str) -> dict[str, Any]:
    return {
        "type": "scatter",
        "mode": "lines",
        "x": [0, 1],
        "y": [0, 1],
        "xaxis": xaxis,
        "yaxis": yaxis,
        "line": {"color": "#afb8c1", "width": 1, "dash": "dash"},
        "hoverinfo": "skip",
        "showlegend": False,
    }


def reliability_pooled_figure(data: Mapping[str, Any]) -> dict[str, Any]:
    states = [data["views"][view]["short"] for view in VIEW_ORDER]
    fig = Figure.empty(states)
    panels = [("raw", "x", "y"), ("cal", "x2", "y2")]
    for _, xa, ya in panels:
        fig.add(diagonal(xa, ya), None)
    for k, view in enumerate(VIEW_ORDER):
        for f in data["finalists"]:
            if view not in f["reliability"]:
                continue
            for arm, xa, ya in panels:
                fig.add(
                    reliability_trace(
                        f,
                        f["reliability"][view][arm],
                        xaxis=xa,
                        yaxis=ya,
                        showlegend=arm == "raw",
                        where=f"{'raw' if arm == 'raw' else 'calibrated'}, "
                        f"{data['views'][view]['short']}",
                    ),
                    k,
                )
    unit = {"range": [0, 1], "tickformat": ".0%", "dtick": 0.2}
    layout = base_layout(
        height=520,
        xaxis=axis("Confidence (top option)", domain=[0, 0.47], **unit),
        yaxis=axis("Accuracy", **unit),
        xaxis2=axis("Confidence (top option)", domain=[0.53, 1], **unit),
        yaxis2=axis("", anchor="x2", **unit),
        annotations=[
            panel_title("Raw probabilities", 0.235, 1.0),
            panel_title(f"Calibrated ({CALIBRATOR}, out of fold)", 0.765, 1.0),
        ],
    )
    return fig.build(layout)


def panel_title(text: str, x: float, y: float) -> dict[str, Any]:
    return {
        "text": f"<b>{html.escape(text)}</b>",
        "x": x,
        "y": y,
        "xref": "paper",
        "yref": "paper",
        "xanchor": "center",
        "yanchor": "bottom",
        "showarrow": False,
        "font": {"size": 13},
    }


def reliability_by_dataset_figure(data: Mapping[str, Any]) -> dict[str, Any]:
    fig = Figure.empty(["Raw", "Calibrated"])
    cols, rows = 3, math.ceil(len(DATASETS) / 3)
    gap_x, gap_y = 0.06, 0.09
    width = (1 - gap_x * (cols - 1)) / cols
    height = (1 - gap_y * (rows - 1)) / rows
    layout = base_layout(height=980, margin={"l": 60, "r": 20, "t": 150, "b": 70})
    annotations = []
    unit = {"range": [0, 1], "dtick": 0.25, "tickformat": ".0%"}
    shown: set[tuple[str, str]] = set()
    for index, benchmark in enumerate(DATASETS):
        r, c = divmod(index, cols)
        suffix = "" if index == 0 else str(index + 1)
        xa, ya = f"x{suffix}", f"y{suffix}"
        x0 = c * (width + gap_x)
        y1 = 1 - r * (height + gap_y)
        layout[f"xaxis{suffix}"] = axis(
            "Confidence" if r == rows - 1 else "", domain=[x0, x0 + width], anchor=ya, **unit
        )
        layout[f"yaxis{suffix}"] = axis(
            "Accuracy" if c == 0 else "", domain=[y1 - height, y1], anchor=xa, **unit
        )
        n = next(
            (
                f["per_benchmark"][benchmark]["n"]
                for f in data["finalists"]
                if benchmark in f["per_benchmark"]
            ),
            0,
        )
        annotations.append(
            panel_title(f"{BENCHMARK_LABELS[benchmark]} (n={n:,})", x0 + width / 2, y1)
        )
        fig.add(diagonal(xa, ya), None)
        for k, arm in enumerate(("raw", "cal")):
            for f in data["finalists"]:
                points = f["reliability"]["by_benchmark"].get(benchmark)
                if points is None:
                    continue
                first = (f["key"], arm) not in shown
                shown.add((f["key"], arm))
                fig.add(
                    reliability_trace(
                        f,
                        points[arm],
                        xaxis=xa,
                        yaxis=ya,
                        showlegend=first,
                        where=f"{BENCHMARK_LABELS[benchmark]}, "
                        f"{'raw' if arm == 'raw' else 'calibrated'}",
                    ),
                    k,
                )
    layout["annotations"] = annotations
    layout["legend"] = {
        "orientation": "h",
        "x": 1,
        "xanchor": "right",
        "y": 1.045,
        "yanchor": "bottom",
    }
    return fig.build(layout)


# --- HTML --------------------------------------------------------------------------------------


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


MINUS = "\u2212"


def pts(value: float, digits: int = 1) -> str:
    """Percentage points with a typographic sign."""
    return f"{100 * value:+.{digits}f}".replace("-", MINUS)


def pct_ci(v: Sequence[float]) -> str:
    return f"{pct(v[0])} <span class=ci>[{pct(v[1])}, {pct(v[2])}]</span>"


def pts_ci(v: Sequence[float]) -> str:
    return f"{pts(v[0])} pts <span class=ci>[{pts(v[1])}, {pts(v[2])}]</span>"


def num_ci(v: Sequence[float], digits: int = 3) -> str:
    return f"{v[0]:.{digits}f} <span class=ci>[{v[1]:.{digits}f}, {v[2]:.{digits}f}]</span>"


def qps_text(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.0f}" if value >= 100 else f"{value:.1f}" if value >= 10 else f"{value:.2f}"


def money(value: float | None) -> str:
    return "—" if value is None else f"${value:.4f}"


def tokens_text(summary: Mapping[str, Any]) -> str:
    """Tokens per question with its prompt/generated split."""
    if summary["tokens_per_question"] is None:
        return "—"
    return (
        f"{summary['tokens_per_question']:,.0f}<br><span class=ci>"
        f"{summary['tokens_in']:,.0f} in + {summary['tokens_out']:,.0f} out</span>"
    )


def swatch(color: str, hollow: bool = False) -> str:
    style = f"border-color:{color};" + ("" if hollow else f"background:{color};")
    return f"<span class=swatch style='{style}'></span>"


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def finalists_table(data: Mapping[str, Any], view: str) -> str:
    head = (
        "<tr><th>System</th><th>Accuracy [95% CI]</th><th>Δ vs Jev [95% CI]</th>"
        "<th>Brier, raw → calibrated</th><th>ECE, raw → calibrated</th><th>q/s</th>"
        "<th>Tokens / question</th><th>Median latency</th><th>$ / 1k questions</th></tr>"
    )
    rows = []
    for f in data["finalists"]:
        v = f["views"].get(view)
        if v is None:
            continue
        delta = pts_ci(v["delta_vs"]["jev"]["accuracy"]) if f["key"] != "jev" else "reference"
        limited = v["speed_note"].startswith("client-limited")
        c = v.get("ceiling")
        without = (
            f"<br><span class=ci>&asymp;{qps_text(c['qps'])} without our limiter</span>"
            if limited and c and c["qps"] >= SHOWN_CEILING_SHARE * v["qps"]
            else ""
        )
        rows.append(
            "<tr>"
            f"<td>{swatch(f['color'], f['kind'] != 'local')}{esc(f['label'])}</td>"
            f"<td>{pct_ci(v['accuracy'])}</td><td>{delta}</td>"
            f"<td>{v['brier_raw'][0]:.3f} → <b>{v['brier_cal'][0]:.3f}</b></td>"
            f"<td>{v['ece_raw']:.3f} → <b>{v['ece_cal']:.3f}</b></td>"
            f"<td>{qps_text(v['qps'])}{'<sup>†</sup>' if limited else ''}{without}</td>"
            f"<td>{tokens_text(v)}</td>"
            f"<td>{v['latency_p50_ms']:,.0f} ms</td>"
            f"<td>{money(v['cost_per_1k_items'])}</td>"
            "</tr>"
        )
    n = max(f["views"][view]["n"] for f in data["finalists"] if view in f["views"])
    caption = (
        f"<caption>{esc(data['views'][view]['label'].capitalize())}: "
        f"{n:,} holdout questions each</caption>"
    )
    return f"<table>{caption}<thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def per_benchmark_table(data: Mapping[str, Any]) -> str:
    head = (
        "<tr><th>Benchmark</th>"
        + "".join(
            f"<th>{swatch(f['color'], f['kind'] != 'local')}{esc(f['label'])}</th>"
            for f in data["finalists"]
        )
        + "</tr>"
    )
    rows = []
    for b in DATASETS:
        cells = []
        for f in data["finalists"]:
            r = f["per_benchmark"].get(b)
            cells.append(
                "<td class=na>not run</td>"
                if r is None
                else f"<td>{pct(r['accuracy'])}<br><span class=ci>ECE {r['ece_raw']:.3f} → "
                f"{r['ece_cal']:.3f}</span></td>"
            )
        n = next(f["per_benchmark"][b]["n"] for f in data["finalists"] if b in f["per_benchmark"])
        rows.append(
            f"<tr><td>{BENCHMARK_LABELS[b]} <span class=ci>n={n:,}</span></td>{''.join(cells)}</tr>"
        )
    return f"<table class=wide><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def includes_zero(ci: Sequence[float]) -> bool:
    return ci[1] <= 0.0 <= ci[2]


def gpqa_lead(data: Mapping[str, Any], view: str) -> str | None:
    """TL;DR sentence on how much of Jev's paired accuracy lead in ``view`` rests on
    :data:`GPQA` (its share, the lead without it, and the item-weighted lead)."""
    benchmarks = data["views"][view]["benchmarks"]
    jev = next((f for f in data["finalists"] if f["key"] == "jev" and view in f["views"]), None)
    if jev is None or GPQA not in benchmarks or not jev["views"][view]["accuracy_lead"]:
        return None
    leads = list(jev["views"][view]["accuracy_lead"].values())
    n = {b: jev["per_benchmark"][b]["n"] for b in benchmarks}
    largest = max(benchmarks, key=lambda b: n[b])
    shares = [x["gpqa_share"] for x in leads if x["gpqa_share"] is not None]
    without = [x["without_gpqa"] for x in leads]
    weighted = [x["item_weighted"] for x in leads]

    def span(values: Sequence[Sequence[float]]) -> str:
        return f"{pts(min(v[0] for v in values))} to {pts(max(v[0] for v in values))} points"

    above = all(v[1] > 0.0 for v in [*without, *weighted])
    return (
        f"Much of Jev's lead comes from {BENCHMARK_LABELS[GPQA]}. Its {n[GPQA]} questions weigh "
        f"as much in the macro as the {n[largest]:,} of {BENCHMARK_LABELS[largest]}, and they "
        f"account for {pct(min(shares), 0)} to {pct(max(shares), 0)} of Jev's lead over each "
        f"challenger. Without {BENCHMARK_LABELS[GPQA]}, Jev leads by {span(without)}; with every "
        f"question weighed the same, by {span(weighted)}. "
        + (
            "Every one of these paired CIs is still above 0."
            if above
            else "Not all of these paired CIs are above 0."
        )
    )


def headline(data: Mapping[str, Any]) -> list[str]:
    """TL;DR sentences, computed from the page data (they stay true when systems change)."""
    view = "common"
    vname = data["views"][view]["label"]
    fins = [f for f in data["finalists"] if view in f["views"]]
    val = {f["key"]: f["views"][view] for f in fins}
    ranked = sorted(fins, key=lambda f: -val[f["key"]]["accuracy"][0])
    challengers = [f for f in ranked if f["key"] != "jev"]
    out = []
    if challengers:
        best = challengers[0]
        bv = val[best["key"]]
        d = bv["delta_vs"]["jev"]["accuracy"]
        lead = (
            "Jev is still the most accurate system."
            if ranked[0]["key"] == "jev"
            else f"{esc(best['label'])} is more accurate than Jev."
        )
        out.append(
            f"{lead} On the {esc(vname)} ({val['jev']['n']:,} holdout questions), Jev answers "
            f"{pct(val['jev']['accuracy'][0])} correctly. The best challenger, "
            f"{esc(best['label'])}, answers {pct(bv['accuracy'][0])}, a difference of "
            f"{pts(d[0])} points (paired 95% CI {pts(d[1])} to {pts(d[2])})."
        )
        gpqa = gpqa_lead(data, view) if ranked[0]["key"] == "jev" else None
        if gpqa:
            out.append(gpqa)
        tied = [
            f["label"]
            for f in challengers[1:]
            if includes_zero(bv["delta_vs"][f["key"]]["accuracy"])
        ]
        if tied:
            out.append(
                f"{esc(best['label'])} is statistically tied with {esc(' and '.join(tied))}: the "
                "95% CI of each paired difference includes 0."
            )
    local = [f for f in data["finalists"] if f["kind"] == "local" and "all" in f["views"]]
    if len(local) >= 2:
        by_speed = sorted(local, key=lambda f: -(f["views"]["all"]["qps"] or 0.0))
        fast, slow, middle = by_speed[0], by_speed[-1], by_speed[1:-1]
        fv, sv = fast["views"]["all"], slow["views"]["all"]
        d = fv["delta_vs"][slow["key"]]["accuracy"]
        between = (
            (
                " The other local finalists are in between: "
                if len(middle) > 1
                else " The other local finalist is in between: "
            )
            + ", ".join(
                f"{esc(f['label'])} ({qps_text(f['views']['all']['qps'])} q/s)" for f in middle
            )
            + "."
            if middle
            else ""
        )
        out.append(
            f"On one GPU and over all 9 benchmarks, the fastest local finalist, "
            f"{esc(fast['label'])}, answers {qps_text(fv['qps'])} questions per second, "
            f"{fv['qps'] / sv['qps']:.1f}&times; as many as the slowest, {esc(slow['label'])} "
            f"({qps_text(sv['qps'])} q/s). Their accuracy differs by {pts(d[0])} points (95% CI "
            f"{pts(d[1])} to {pts(d[2])}).{between}"
        )
    ece_raw = [val[f["key"]]["ece_raw"] for f in fins]
    ece_cal = [val[f["key"]]["ece_cal"] for f in fins]
    out.append(
        f"Before calibration, macro ECE ranges from {min(ece_raw):.3f} to {max(ece_raw):.3f}. "
        "One temperature per answer format, fitted without seeing the questions it scores "
        f"(<a href='#calibration'>how</a>), brings every system's macro ECE to between "
        f"{min(ece_cal):.3f} and {max(ece_cal):.3f}. Calibrated Brier score, best first: "
        + ", ".join(
            f"{esc(f['label'])} {val[f['key']]['brier_cal'][0]:.3f}"
            for f in sorted(fins, key=lambda f: val[f["key"]]["brier_cal"][0])
        )
        + "."
    )
    priced = [f for f in fins if val[f["key"]]["cost_per_1k_items"] is not None]
    if priced:
        out.append(
            "Cost per 1,000 questions on the shared benchmarks, cheapest first: "
            + ", ".join(
                f"{esc(f['label'])} {money(val[f['key']]['cost_per_1k_items'])}"
                for f in sorted(priced, key=lambda f: val[f["key"]]["cost_per_1k_items"])
            )
            + " (local models count electricity only)."
        )
    return out


def landscape_summary(data: Mapping[str, Any]) -> str:
    view = "all"
    local = [e for e in data["landscape"] if is_emulator_candidate(e) and view in e["views"]]
    pts_ = [(e["views"][view]["qps"], e["views"][view]["accuracy"][0]) for e in local]
    kept = sorted(frontier(pts_, higher_is_better=True), key=lambda i: pts_[i][0])
    steps = ", ".join(
        f"{esc(local[i]['label'])} ({pct(pts_[i][1])} at {qps_text(pts_[i][0])} q/s)" for i in kept
    )
    best = max(local, key=lambda e: e["views"][view]["accuracy"][0])
    jev = next(e for e in data["landscape"] if e["kind"] == "jev")
    clm = next(e for e in data["landscape"] if e["key"] == CLM.key)["views"][view]
    return (
        f"Of the {len(local)} local models, scored on all 9 benchmarks, the most accurate is "
        f"{esc(best['label'])} at {pct(best['views'][view]['accuracy'][0])}. Jev scores "
        f"{pct(jev['views'][view]['accuracy'][0])} on the same questions (by the argmax of its "
        "probabilities, like every system here; <code>reports/speed_quality</code> scores Jev by "
        f"its returned choice: {pct(jev['choice_accuracy_speed_quality'])}). The accuracy "
        f"frontier, from slowest to fastest: {steps}. CLM-8B, a local dual encoder counted apart "
        f"from those models, scores {pct(clm['accuracy'][0])} at {qps_text(clm['qps'])} q/s."
    )


CSS = """
:root { --fg:#1f2328; --muted:#59636e; --line:#d1d9e0; --soft:#f6f8fa; --accent:#0969da; }
* { box-sizing: border-box; }
body { margin:0; background:#f3f4f6; color:var(--fg);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
  font-size:17px; line-height:1.6; }
main { max-width: 1040px; margin: 32px auto; background:white; padding: 56px 64px 72px;
  border-radius: 14px; box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 8px 24px rgba(0,0,0,.04); }
header .kicker { text-transform:uppercase; letter-spacing:.08em; font-size:12px;
  color:var(--accent); font-weight:600; }
h1 { font-size: 40px; line-height:1.15; margin: 8px 0 12px; letter-spacing:-.01em; }
h2 { font-size: 26px; margin: 56px 0 8px; letter-spacing:-.01em; }
h3 { font-size: 19px; margin: 32px 0 6px; }
.lede { font-size: 20px; color: var(--muted); margin: 0 0 8px; }
.byline { font-size: 14px; color: var(--muted); }
.tldr { background: var(--soft); border: 1px solid var(--line);
  border-left: 4px solid var(--accent); border-radius: 10px; padding: 18px 24px; margin: 28px 0; }
.tldr h2 { margin: 0 0 6px; font-size: 15px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--accent); }
.tldr ul { margin: 0; padding-left: 20px; } .tldr li { margin: 6px 0; }
.cards { display:grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px;
  margin: 18px 0; }
.card { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.card .name { font-weight: 600; font-size: 15px; }
.card .what { font-size: 13px; color: var(--muted); line-height: 1.4; margin: 4px 0 10px; }
.card .big { font-size: 26px; font-weight: 650; font-variant-numeric: tabular-nums; }
.card .sub { font-size: 12.5px; color: var(--muted); }
.card.pending { border-style: dashed; background: #fffbea; }
.swatch { display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:7px;
  border: 2px solid; vertical-align: 0; }
table { border-collapse: collapse; width: 100%; font-size: 14px; margin: 14px 0 6px;
  font-variant-numeric: tabular-nums; }
caption { caption-side: top; text-align: left; font-weight: 600; font-size: 14px;
  padding-bottom: 6px; }
th { text-align: right; font-weight: 600; border-bottom: 2px solid var(--line); padding: 7px 8px;
  vertical-align: bottom; }
td { text-align: right; border-bottom: 1px solid #eaeef2; padding: 7px 8px; vertical-align: top; }
th:first-child, td:first-child { text-align: left; }
table.wide td, table.wide th { font-size: 13px; }
.ci { color: var(--muted); font-size: 12px; white-space: nowrap; }
td.na { color: #9aa3ad; font-style: italic; }
figure { margin: 22px 0 28px; }
figcaption { font-size: 14px; color: var(--muted); margin-top: 4px; }
.split-tag { font-size: 13px; color: var(--muted); margin-bottom: 4px; }
.split-tag b { color: var(--fg); }
.h-split { font-size: 0.7em; font-weight: 500; color: var(--muted); }
table.splits td, table.splits th { text-align: left; }
.plot { width: 100%; }
.callout { border: 1px dashed #d4a72c; background: #fffbea; border-radius: 10px; padding: 12px 16px;
  font-size: 15px; margin: 16px 0; }
.note { font-size: 14px; color: var(--muted); }
.explainer { background: var(--soft); border: 1px solid var(--line); border-radius: 10px;
  padding: 6px 22px 10px; margin: 20px 0; font-size: 15.5px; }
.explainer h3 { margin: 14px 0 4px; font-size: 17px; }
.explainer p { margin: 8px 0; }
.explainer math[display=block] { font-size: 1.15em; margin: 10px 0; }
code { font-size: 14px; background: var(--soft); padding: 1px 5px; border-radius: 5px; }
pre { background: var(--soft); padding: 12px 16px; border-radius: 8px; overflow-x: auto;
  font-size: 13.5px; }
ul.caveats li { margin: 8px 0; }
@media (max-width: 760px) { main { padding: 28px 18px; margin: 0; border-radius: 0; }
  h1 { font-size: 30px; } }
"""


def math_block(body: str) -> str:
    """One displayed MathML equation (MathML Core renders natively in current browsers)."""
    return f"<math display=block>{body}</math>"


def power_of_ten(value: float) -> str:
    """``1e-6`` as MathML ``10^-6``."""
    return f"<msup><mn>10</mn><mn>{round(math.log10(value)):d}</mn></msup>".replace(
        "<mn>-", "<mn>&minus;"
    )


def formulas() -> str:
    """The "Formulas" part of the calibration box: option probabilities from log-probs,
    temperature scaling and its fit, and the three probability metrics, as the code computes
    them (constants imported from the code)."""
    frac = "<mfrac><mrow>{}</mrow><mrow>{}</mrow></mfrac>"
    sum_to_k = (
        "<munderover><mo>&sum;</mo><mrow><mi>{}</mi><mo>=</mo><mn>1</mn></mrow>"
        "<mi>K</mi></munderover>"
    )
    sum_j, sum_i = sum_to_k.format("j"), sum_to_k.format("i")
    ell = "<msub><mi>&ell;</mi><mi>{}</mi></msub>"
    p = "<msub><mi>p</mi><mi>{}</mi></msub>"
    exp = "<msup><mi>e</mi><mrow>{}</mrow></msup>"
    probs = math_block(
        p.format("i")
        + "<mo>=</mo>"
        + frac.format(exp.format(ell.format("i")), sum_j + exp.format(ell.format("j")))
    )
    temp = math_block(
        "<msubsup><mi>p</mi><mi>i</mi><mo>&prime;</mo></msubsup><mo>=</mo>"
        + frac.format(
            f"<msup>{p.format('i')}<mrow><mn>1</mn><mo>/</mo><mi>T</mi></mrow></msup>",
            sum_j + f"<msup>{p.format('j')}<mrow><mn>1</mn><mo>/</mo><mi>T</mi></mrow></msup>",
        )
        + "<mo>=</mo>"
        + frac.format(
            exp.format(ell.format("i") + "<mo>/</mo><mi>T</mi>"),
            sum_j + exp.format(ell.format("j") + "<mo>/</mo><mi>T</mi>"),
        )
    )
    fit = math_block(
        "<msub><mi>T</mi><mi>s</mi></msub><mo>=</mo>"
        "<munder><mrow><mi>arg</mi><mspace width='0.17em'/><mi>min</mi></mrow><mi>T</mi></munder>"
        + frac.format("<mn>1</mn>", "<msub><mi>N</mi><mi>s</mi></msub>")
        + "<munder><mo>&sum;</mo><mrow><mi>q</mi><mo>&isin;</mo><mi>s</mi></mrow></munder>"
        "<mo>&minus;</mo><mi>log</mi><mspace width='0.17em'/>"
        "<msubsup><mi>p</mi><mrow><msub><mi>g</mi><mi>q</mi></msub></mrow><mo>&prime;</mo>"
        "</msubsup><mo>(</mo><mi>T</mi><mo>)</mo>"
    )
    nll = math_block(
        "<mtext>NLL</mtext><mo>=</mo><mo>&minus;</mo><mi>log</mi><mspace width='0.17em'/>"
        "<mi>max</mi><mo>(</mo>" + p.format("g") + f"<mo>,</mo>{power_of_ten(NLL_EPS)}<mo>)</mo>"
    )
    brier = math_block(
        "<mtext>Brier</mtext><mo>=</mo>"
        + sum_i
        + "<msup><mrow><mo>(</mo>"
        + p.format("i")
        + "<mo>&minus;</mo><msub><mi>y</mi><mi>i</mi></msub><mo>)</mo></mrow><mn>2</mn></msup>"
    )
    ece = math_block(
        "<mtext>ECE</mtext><mo>=</mo><munderover><mo>&sum;</mo><mrow><mi>b</mi><mo>=</mo>"
        f"<mn>1</mn></mrow><mn>{N_BINS}</mn></munderover>"
        + frac.format("<msub><mi>n</mi><mi>b</mi></msub>", "<mi>N</mi>")
        + "<mo>|</mo><msub><mtext>acc</mtext><mi>b</mi></msub><mo>&minus;</mo>"
        "<msub><mtext>conf</mtext><mi>b</mi></msub><mo>|</mo>"
    )
    return f"""<h3 id=formulas>Formulas</h3>
<p>The model is asked once, and &ell;<sub>i</sub> is the log-probability of option i's label
token at the answer position (the forms <code>A</code> and <code>&nbsp;A</code> of one label are
added together). The K labels' probabilities are renormalized so they sum to 1:</p>
{probs}
<p>A label missing from the returned top-k gets an upper bound instead of zero: the smallest
returned log-probability or, in the API models' structured read (where only labels can follow),
the probability the listed labels leave over. Jev returns its probabilities directly.</p>
<p>Temperature scaling, written in two equivalent forms:</p>
{temp}
<p>For each signature s, the fitted temperature T<sub>s</sub> minimizes the mean log loss
over its N<sub>s</sub> fitting questions, where g<sub>q</sub> is question q's correct option. T is
searched between {T_MIN:g} and {T_MAX:g}.</p>
{fit}
<p>The metrics per question use p<sub>g</sub>, the probability of the correct option, and
y<sub>i</sub>, which is 1 for the correct option and 0 for the others. NLL is clipped at
10<sup>&minus;{-round(math.log10(NLL_EPS)):d}</sup> so a probability of exactly 0 stays finite;
Brier runs from 0 (perfect) to 2.</p>
{nll}
{brier}
<p>ECE is computed per benchmark. Each question's confidence is its top option's probability,
max<sub>i</sub>&nbsp;p<sub>i</sub>. Questions are sorted by confidence and split into {N_BINS}
groups of equal size; for group b, acc<sub>b</sub> is the share answered correctly and
conf<sub>b</sub> the mean confidence. The reliability figures use fixed-width bins instead, for
display only.</p>
{ece}
<p>Accuracy, NLL, Brier and ECE are computed per benchmark, then averaged over the benchmarks
with equal weight (macro).</p>"""


def calibration_explainer(data: Mapping[str, Any]) -> str:
    """The "How calibration works" box, with the finalist whose raw ECE is worst as example."""
    fins = [f for f in data["finalists"] if "common" in f["views"]]
    worst = max(fins, key=lambda f: f["views"]["common"]["ece_raw"])
    v = worst["views"]["common"]
    return f"""<div class=explainer id=calibration>
<h3>How calibration works</h3>
<p>Calibration changes a system's confidence and leaves the answer it picks unchanged.</p>
<p>The method is temperature scaling. A question has K options, with probabilities
p<sub>1</sub>&hellip;p<sub>K</sub>. Each is replaced by p&prime;<sub>i</sub> &prop;
p<sub>i</sub><sup>1/T</sup> and the results are renormalized to sum to 1. T is one fitted number:
T &gt; 1 softens overconfident answers (a 95% may become 85%), T &lt; 1 sharpens underconfident
ones. The order of the options never changes, so the top answer and the accuracy stay exactly the
same. Only the confidence moves, and with it the Brier score, the NLL (negative log-likelihood,
also called log loss) and the ECE. T is chosen to minimize the log loss on the fitting
questions.</p>
<p>The calibrator, <code>{CALIBRATOR}</code>, fits one temperature per answer format. A
question's <i>signature</i> is its type plus its exact option labels. All A to D multiple-choice
questions share one signature; BoolQ's two options have another; banking77's 77 intents another.
A model can be overconfident on one format and underconfident on another, so each format gets its
own T. A format with fewer than {DEFAULT_MIN_GROUP} fitting questions uses one global T instead.
Each system gets its own temperatures, fitted on its own answers only.</p>
<p>The temperatures are fitted out of fold. A system's questions are split into {FOLDS} folds.
The folds are the same for every system, and the items of one question stay together. Each fold
is calibrated with temperatures fitted on the other {FOLDS - 1}, so no question is scored by a
fit that saw it. The fit still comes from the same benchmarks and split, though. A deployed
calibrator is fitted once, on a separate split.</p>
{formulas()}
<p>For example, {esc(worst["label"])} has the highest raw ECE. On the
{esc(data["views"]["common"]["label"])}, calibration takes its ECE from {v["ece_raw"]:.3f} to
{v["ece_cal"]:.3f} and its Brier score from {v["brier_raw"][0]:.3f} to
{v["brier_cal"][0]:.3f}, while its accuracy stays at {pct(v["accuracy"][0])}.</p>
</div>"""


def cards(data: Mapping[str, Any]) -> str:
    out = []
    for f in data["finalists"]:
        v = f["views"]["common"]
        out.append(
            "<div class=card>"
            f"<div class=name>{swatch(f['color'], f['kind'] != 'local')}{esc(f['label'])}</div>"
            f"<div class=what>{esc(f['detail'])}</div>"
            f"<div class=big>{pct(v['accuracy'][0])}</div>"
            f"<div class=sub>holdout accuracy, {esc(data['views']['common']['short'])}</div>"
            f"<div class=sub>{qps_text(v['qps'])} q/s &middot; calibrated Brier "
            f"{v['brier_cal'][0]:.3f}</div>"
            "</div>"
        )
    for p in data["pending"]:
        progress = ", ".join(f"{k} {v}" for k, v in p["progress"].items())
        out.append(
            "<div class='card pending'>"
            f"<div class=name>{esc(p['label'])}</div>"
            f"<div class=what>{esc(p['detail'])}</div>"
            "<div class=big>pending</div>"
            f"<div class=sub>runs in progress: {esc(progress)}</div>"
            "</div>"
        )
    return f"<div class=cards>{''.join(out)}</div>"


CAPTIONS = {
    "fig-land-acc": "Macro accuracy against questions per second (q/s), one point per model. "
    "Error bars are 95% bootstrap intervals. Hollow markers are API services; their q/s is the "
    "rate of our client's request limiter. The dotted arrow ends at what the service's own "
    "limits would allow without the limiter (<a href='#speed-limits'>speed limits</a>). Local "
    "models ran on " + LOCAL_SETUP + " with 16 questions in flight; other machines will give "
    "very different numbers. The second row of buttons switches the x axis to tokens per "
    "question.",
    "fig-land-brier": "Macro Brier score (multiclass, 0 to 2, lower is better) against q/s. "
    "<i>Calibrated</i>: one temperature per answer format, fitted out of fold (5 folds) on these "
    "questions (see <a href='#calibration'>How calibration works</a>).",
    "fig-final-acc": "Holdout macro accuracy, with 95% intervals, against q/s or tokens per "
    "question. Dotted arrows show API ceilings without our request limiter "
    "(<a href='#speed-limits'>speed limits</a>).",
    "fig-final-brier": "Holdout macro Brier score, raw and calibrated, against q/s or tokens per "
    "question. Dotted arrows as above.",
    "fig-rel-pooled": "Reliability on holdout: all questions of the view pooled into 5-point "
    "confidence bins. The left panel is raw, the right calibrated. Pooling weighs each benchmark "
    "by its size.",
    "fig-rel-dataset": "Reliability per benchmark on holdout, in 10-point confidence bins. The "
    "buttons above the figure switch between raw and calibrated probabilities. Click a legend "
    "entry to hide a system.",
}


FIGURE_SPLIT: dict[str, SplitName] = {
    "fig-land-acc": LANDSCAPE_SPLIT,
    "fig-land-brier": LANDSCAPE_SPLIT,
    "fig-final-acc": FINAL_SPLIT,
    "fig-final-brier": FINAL_SPLIT,
    "fig-rel-pooled": FINAL_SPLIT,
    "fig-rel-dataset": FINAL_SPLIT,
}
"""The split each figure's numbers come from, stated in a tag above the figure."""


def split_tag(data: Mapping[str, Any], split: SplitName) -> str:
    """``screen split · 1,599 questions (6 shared benchmarks), 2,499 (all 9 benchmarks)``."""
    n, views = data["splits"][split], data["views"]
    counts = ", ".join(f"{n[v]:,} ({esc(views[v]['short'])})" for v in VIEW_ORDER)
    return (
        f"<div class=split-tag><a href='#splits'><b>{split}</b> split</a> &middot; "
        f"questions: {counts}</div>"
    )


def figure_block(data: Mapping[str, Any], fig_id: str) -> str:
    return (
        f"<figure>{split_tag(data, FIGURE_SPLIT[fig_id])}"
        f"<div id='{fig_id}' class=plot></div>"
        f"<figcaption>{CAPTIONS[fig_id]}</figcaption></figure>"
    )


def splits_section(data: Mapping[str, Any]) -> str:
    """The "Data splits" section: which split every number comes from, and why."""
    n, views = data["splits"], data["views"]
    n_local = sum(1 for e in data["landscape"] if is_emulator_candidate(e))

    def sizes(split: str) -> str:
        return "<br>".join(f"{n[split][v]:,} ({esc(views[v]['short'])})" for v in VIEW_ORDER)

    rows = [
        (
            "screen",
            f"All {n_local} local candidates, CLM-8B, Jev and the API models",
            "A quick first comparison of every candidate: at most 300 questions per benchmark, "
            "taken from <code>select</code>.",
            "The figures of all candidates.",
        ),
        (
            "select",
            "The finalists, Jev and the API models",
            "Every choice: which models became finalists, the prompt layout, the quantized "
            "builds and the vLLM settings.",
            "Not shown (<code>docs/research/selection.md</code>).",
        ),
        (
            "holdout",
            "The finalists, Jev and the API models",
            "The final numbers. No choice was made on these questions.",
            "The TL;DR, the cards, the finalist tables and figures, calibration and reliability.",
        ),
    ]
    body = "".join(
        f"<tr><td><b>{split}</b></td><td>{sizes(split)}</td><td>{who}</td><td>{use}</td>"
        f"<td>{where}</td></tr>"
        for split, who, use, where in rows
    )
    return f"""<h2 id=splits>Data splits</h2>
<p>Each benchmark was split once, before any model ran, into two halves (<code>select</code> and
<code>holdout</code>) plus a small sample of <code>select</code> (<code>screen</code>). A system
scores differently on different questions, so compare numbers only within one split. Every figure
and table below says which split it shows.</p>
<table class=splits><thead><tr><th>Split</th><th>Questions</th><th>Answered by</th>
<th>Used for</th><th>On this page</th></tr></thead><tbody>{body}</tbody></table>
<p>Calibrated numbers are cross-fitted on
<code>holdout</code>: its questions are split into {FOLDS} folds, and each fold is calibrated with
temperatures fitted on the other {FOLDS - 1}, so no question is scored by a fit that saw it
(<a href='#calibration'>how</a>). The calibrators shipped for use
(<code>calibration/&lt;preset&gt;/registry.json</code>) are fitted once on all of
<code>holdout</code>.</p>"""


def data_caveats(data: Mapping[str, Any]) -> str:
    """Caveat items on how far the numbers can be read: run-to-run noise, GPQA-Diamond's easy
    holdout half, the API read (JSON answers, Luna's short top-5 lists), declared unanswered
    questions and Jev's rounded probabilities. Counts come from the page data; the noise and
    split figures from the correctness audit and the API probe reports."""
    view = "common"
    vname = data["views"][view]["label"]
    fins = {f["key"]: f for f in data["finalists"]}
    others = [
        f["views"][view]["accuracy"][0]
        for f in data["finalists"]
        if f["kind"] != "jev" and view in f["views"]
    ]
    gpqa = BENCHMARK_LABELS[GPQA]
    items = [
        "<li>vLLM is not deterministic on this setup, so runs do not repeat exactly. Sending "
        "byte-identical requests again flipped the top answer on 4 to 7% of hard "
        f"multiple-choice questions ({gpqa}, LEXam, MMLU-Pro; about 0.3% on ARC). The GPTQ "
        f"model's {gpqa} accuracy ranged from 54.5% to 57.6% across runs. In a probe, 2 of 100 "
        "of GPT-6 Luna's top answers also flipped between identical calls. The 95% intervals "
        "cover only the sampling of questions and leave this noise out. The emulators and API "
        f"models ({pct(min(others))} to {pct(max(others))} on the {esc(vname)}) therefore cannot "
        "be ranked when they are within about 1 point of each other.</li>",
        f"<li>{gpqa}'s holdout half is easier than its select half: every system scores 8 to "
        "18 points higher on it (permutation test p = 0.014). The split is a random hash, the "
        "same for every system, so this is luck of the draw. This is why every system scores "
        "about 2 points higher on holdout than on select over the shared benchmarks, and why "
        "Gemma's select-to-holdout jump is bigger.</li>",
    ]
    if any(f["kind"] == "api" for f in data["finalists"]):
        items.append(
            "<li>API models answer in JSON. Local models are made to start their reply with "
            "&ldquo;Answer:&rdquo;, so the next token is the label, but the hosted APIs do not "
            "continue such a prefill. API models therefore must reply "
            "<code>{&quot;answer&quot;: &quot;&lt;label&gt;&quot;}</code> with a label from a "
            "fixed list (Structured Outputs), and the probabilities are read where the label "
            "starts, where only labels can follow. On hard or borderline questions this wrapping "
            "can change the model's pick compared with a free reply.</li>"
        )
    luna = fins.get("luna")
    if luna and view in luna["views"]:
        read = luna["views"][view]["read"]
        n_top = read["n_missing_top"]
        items.append(
            f"<li>{esc(luna['label'])} lists only a few labels: OpenAI returns at most 5 "
            "alternatives and drops unlikely ones, so on "
            f"{pct(read['missing_label_share'], 0)} of its holdout questions some option is "
            "missing. A missing option gets the probability the listed labels leave over, at "
            f"most {pct(1.0 - read['min_label_mass'])}. That is an upper bound, since only labels "
            "can follow there, and its raw NLL depends on it. Its "
            "calibrated numbers barely do (NLL moves by at most 0.02 even with the missing "
            "options at 0), and its accuracy "
            + (
                "not at all (a missing option is never the top answer)."
                if n_top == 0
                else f"on the {n_top:,} questions where a missing option is the top answer."
            )
            + "</li>"
        )
    for f in data["finalists"]:
        if not f["unanswered"]:
            continue
        final = f["unanswered"].get(FINAL_SPLIT, {})
        on_final = ", ".join(
            f"{c['n']} of {c['of']:,} {BENCHMARK_LABELS[b]} questions" for b, c in final.items()
        )
        elsewhere = [
            f"{sum(c['n'] for c in counts.values())} on {split}"
            for split, counts in f["unanswered"].items()
            if split != FINAL_SPLIT
        ]
        items.append(
            f"<li>{esc(f['label'])} left some questions unanswered: "
            + ", ".join([f"{on_final or 'none'} on {FINAL_SPLIT}", *elsewhere])
            + ". Its reply never "
            "reached the answer, even when sent again. It is scored on the questions it "
            "answered, and paired differences use the questions both systems answered.</li>"
        )
    jev = fins.get("jev")
    if jev and "all" in jev["views"]:
        v = jev["views"]["all"]
        items.append(
            "<li>Jev rounds its probabilities to 0.01. On "
            f"{v['n_nll_clipped']:,} of its {v['n']:,} holdout questions the correct option gets "
            f"exactly 0, which NLL clips to 10<sup>&minus;{-round(math.log10(NLL_EPS)):d}</sup> "
            f"({-math.log(NLL_EPS):.1f} nats each). Jev's raw NLL (in "
            "<code>metrics.json</code>) is therefore inflated, and raw-NLL gaps understate its "
            "lead. This "
            "page compares Brier and ECE; calibrated NLL is barely affected.</li>"
        )
    return "\n".join(items)


def speed_limits(data: Mapping[str, Any]) -> str:
    """The "Speed limits" section: the measured q/s of each finalist, what
    capped it, and the ceiling without our client's limiter."""
    view = "common"
    items = []
    for f in data["finalists"]:
        v = f["views"].get(view)
        if v is None or f["kind"] == "local":
            continue
        spec = next(x for x in FINALISTS if x.key == f["key"])
        if spec.max_rpm is None:
            fail(f"{spec.label}: an API finalist needs its limiter rate (max_rpm)")
        c = v["ceiling"]
        others = "; ".join(f"{cap['why']}: {qps_text(cap['qps'])}" for cap in c["caps"][1:])
        wide = f["views"].get("all", {}).get("ceiling")
        items.append(
            f"<li>{esc(f['label'])} ran at {qps_text(v['qps'])} q/s with "
            f"{'/'.join(str(n) for n in v['in_flight'])} in flight, with our limiter set to "
            f"{spec.max_rpm:,} requests/min ({qps_text(spec.max_rpm / 60.0)} q/s). The ceiling "
            f"without the limiter is &asymp;{qps_text(c['qps'])} q/s, bound by "
            f"{esc(c['binding'])}"
            + (f" (other bounds: {esc(others)})" if others else "")
            + (
                f". Over {data['views']['all']['label']} it is &asymp;{qps_text(wide['qps'])} "
                f"q/s, bound by {esc(wide['binding'])}"
                if wide and wide["binding"] != c["binding"]
                else ""
            )
            + f". {c['source']}"
            + (
                f" The bound shown is therefore our own in-flight setting ({c['in_flight']} "
                "here) divided by the measured latency; it grows with that setting and is not a "
                "published service limit."
                if c["in_flight"]
                else ""
            )
            + "</li>"
        )
    local = [f for f in data["finalists"] if f["kind"] == "local" and view in f["views"]]
    little = "; ".join(
        f"{esc(f['label'])} {qps_text(v['qps'])} q/s measured, "
        f"{v['in_flight'][0]} &divide; {v['latency_mean_ms']:,.0f} ms = "
        f"{qps_text(v['in_flight'][0] * 1000.0 / v['latency_mean_ms'])} q/s"
        for f in local
        for v in [f["views"][view]]
    )
    other_split = [f for f in FINALISTS if f.speed[1] != FINAL_SPLIT]
    sources = "".join(
        f"{esc(f.label)}'s speed and cost come from its <code>{f.speed[1]}</code> run. "
        for f in other_split
    )
    return f"""<h3 id=speed-limits>Speed limits</h3>
<p>Local runs kept 16 questions in flight. API runs also went through our client's request
limiter, so their q/s (marked &dagger;) is the limiter's rate. The dotted arrows in the speed
figures end at the q/s each service's own limits would allow without the limiter. The ceilings
below are computed for the {data["views"][view]["label"]}; none of them was measured.</p>
<ul class=caveats>{"".join(items)}</ul>
<p>Every local number comes from one machine, {LOCAL_SETUP}, with presets tuned for that card,
16 questions in flight and no client limiter. So q/s is the number in flight divided by the mean
latency (on the {data["views"][view]["label"]}: {little}).</p>
<p>Another GPU, engine (llama.cpp, MLX, SGLang), quantization or concurrency level can change
q/s several-fold, up or down. The fastest small models reached the limit of 16 in flight before
the GPU's. Local q/s ranks the models on this card; your machine will give other numbers.
Running the same benchmark on a Mac (llama.cpp or MLX) is on the to-do list
(<code>docs/TODO.md</code>) and has not been done.</p>
<p class=note>{sources}Cost: token prices for APIs; for local models, the GPU's electricity:
{DEFAULT_GPU_WATTS:.0f} W at ${DEFAULT_USD_PER_WH * 1000:.2f}/kWh =
${DEFAULT_GPU_WATTS * DEFAULT_USD_PER_WH:.4f} per GPU-hour (hardware not included).</p>"""


def render(data: Mapping[str, Any], figures: Mapping[str, Any]) -> str:
    views = data["views"]
    common_names = ", ".join(BENCHMARK_LABELS[b] for b in views["common"]["benchmarks"])
    skipped = [b for b in DATASETS if b not in views["common"]["benchmarks"]]
    n_final = len(data["finalists"]) + len(data["pending"])
    pending_note = ""
    if data["pending"]:
        names = ", ".join(esc(p["label"]) for p in data["pending"])
        pending_note = (
            f"<div class=callout>{names}: "
            + ("its runs are" if len(data["pending"]) == 1 else "their runs are")
            + " still in progress. Once they are complete, the page regenerates with "
            + ("it" if len(data["pending"]) == 1 else "them")
            + " in every table and figure.</div>"
        )
    tldr = "".join(f"<li>{s}</li>" for s in headline(data))
    body = f"""
<header>
  <div class=kicker>jevemu · summary report</div>
  <h1>Local and API models as stand-ins for Jev</h1>
  <p class=lede>This report tests whether a model you run yourself, or a cheap API model, can
  stand in for Jev by matching its answers and its confidence, and how fast it answers. It
  compares {n_final} systems, Jev included, on 9 benchmarks.</p>
  <div class=byline>Every number is recomputed from the run records by
  <code>scripts/make_summary_report.py</code>. Hover over any point for details.</div>
</header>

<section class=tldr><h2>TL;DR <span class=h-split>(<a href='#splits'>holdout</a> split)</span></h2>
<ul>{tldr}</ul></section>

<p class=note><i>Accuracy</i> is the share of questions where the option with the highest
probability is correct. A <i>macro</i> average weighs each benchmark the same. <i>q/s</i> is
questions answered per second, pooled over every question of a view, so unlike macro accuracy it
depends on the benchmark mix. <i>CI</i> is a 95% confidence interval from a bootstrap;
<i>paired</i> means two systems compared on the same questions. <i>Points</i> are percentage
points. The <i>Brier score</i> is the squared error between the option probabilities and the
correct answer, from 0 (perfect) to 2; lower is better. <i>ECE</i> (expected calibration error)
is the average gap between a system's confidence in its top answer and how often that answer is
right; 0 is perfect. <i>Raw</i> probabilities are as returned; <i>calibrated</i> ones are after
temperature scaling (<a href='#calibration'>how</a>).</p>

{splits_section(data)}

<h2>Systems compared</h2>
<p>Jev (<code>jev-1.13.0</code>) answers multiple-choice and classification questions with a
probability for every option. The emulator in this repository asks an open model the same
question. It reads the probabilities from the model's next-token distribution over the option
labels, in one model call per question. Questions with more than {LETTER_LIMIT} options
(banking77, CLINC150) get two-capital labels (AA, AB, &hellip;), each one token, instead of
letters. Local models use {LOCAL_SETUP}. API models run with reasoning turned off and must reply
<code>{{&quot;answer&quot;: &quot;&lt;label&gt;&quot;}}</code> with a label from a fixed list
(Structured Outputs); the probabilities come from the logprobs (log probabilities) where the
label starts. CLM-8B (<a href='https://github.com/Contrastive-LM/CLM'>Contrastive-LM/CLM</a>),
shown on the <code>screen</code> split only, is a dual encoder: Qwen3-8B embeds the state and
question once and each option's text on its own, and a 20M-parameter head scores the options by
scaled cosine similarity. It answers Jev's wire format, one request per question, on one RTX
3090. In the figures, filled markers are local models (CLM-8B included) and hollow markers are
API services.</p>
{cards(data)}
{pending_note}
<p class=note>Benchmarks: GPQA-Diamond and LEXam (both with an "I don't know" option), MMLU-Pro,
ARC-Challenge, AG News, banking77, CLINC150, BoolQ and SST-5. Yelp was dropped because its star
levels were shown as digits 0 to 4 and read ambiguously.</p>
<p class=note>API models return logprobs for only a few tokens (5 for OpenAI, 20 for
OpenRouter), so they can only score questions whose options fit. Where needed they skip
{esc(", ".join(BENCHMARK_LABELS[b] for b in skipped))}. Comparisons that include them use the
{len(views["common"]["benchmarks"])} benchmarks every system answers ({esc(common_names)}). The
buttons above each figure switch between that view and all 9.</p>

<h2>All candidates on the screen split: accuracy and Brier score against speed</h2>
<p>Every candidate answered the same {data["splits"]["screen"]["all"]:,}-question
<code>screen</code> split with the same settings (16 questions in flight). For every finalist
these numbers are lower than its holdout numbers further down: <code>screen</code> is a
different, smaller set of questions. Jev's q/s here is its holdout rate, since our request
limiter sets it on every split. {landscape_summary(data)}</p>
<p>The dotted line is the Pareto frontier of the local models, CLM-8B left out: the ones nothing
else beats on both speed and score. With the x axis set to tokens, it joins the ones nothing else
beats on both tokens used and score.</p>
<p>Tokens per question count every prompt token sent and every token generated. Each system
counts in its own tokenizer. Jev's count includes the tokens it generates internally.</p>
{figure_block(data, "fig-land-acc")}
{figure_block(data, "fig-land-brier")}

<h2>Finalists on holdout</h2>
<p>The finalists then ran on <code>holdout</code>: the half of every benchmark that no choice
was made on ({data["splits"]["holdout"]["all"]:,} questions over all 9 benchmarks). Differences
against Jev are paired: both systems on the same questions.</p>
{finalists_table(data, "common")}
{finalists_table(data, "all")}
<p class=note>&dagger; Client-limited: the q/s is the rate of our request limiter and does not
measure the service's capacity. The &asymp; value below it is the ceiling without the limiter
(<a href='#speed-limits'>speed limits</a>).</p>
{figure_block(data, "fig-final-acc")}
{figure_block(data, "fig-final-brier")}
{speed_limits(data)}

<h2>Calibration and reliability on holdout</h2>
<p>A reliability diagram groups questions by the probability a system gave its top answer. It
then plots how often that answer was right. A calibrated system sits on the diagonal; points
below the diagonal mean overconfidence.</p>
<p>Bins have equal width, so the points cover every confidence level that occurs. Marker area
shows how many questions a bin holds; bins with fewer than {MIN_BIN_ITEMS} are left out. The top
answer's probability can never be below 1 divided by the number of options: 50% on BoolQ, 25% on
four-option questions.</p>
{calibration_explainer(data)}
{figure_block(data, "fig-rel-pooled")}
<h3>Reliability per benchmark</h3>
{figure_block(data, "fig-rel-dataset")}
{per_benchmark_table(data)}
<p class=note>Accuracy and ECE (raw, then calibrated) per benchmark on holdout. "Not run" marks
benchmarks whose questions have more options than the API returns logprobs for.</p>

<h2>Caveats</h2>
<ul class=caveats>
<li>Speed is not measured like for like. Local models share one consumer GPU and one engine
(vLLM), and your hardware will give different numbers. API numbers are bounded by our client's
request limiter and by network latency. Jev can answer several questions per request, which
these runs did not use.</li>
<li>Calibration is fitted on the evaluation half. The fit is out of fold (5 folds), so no item
is scored by a fit that saw it, but it comes from the same distribution. A deployed calibrator is
fitted once, on a whole split.</li>
<li>Accuracy uses the argmax of the probabilities, the option with the highest
probability. An "I don't know" answer counts as wrong. On a handful of near-ties, Jev's returned
choice differs from its argmax.</li>
{data_caveats(data)}
<li>Local cost covers electricity only: the RTX 3090 draws about 410 W while running, priced at
the New Jersey average. It excludes the GPU and the machine.</li>
</ul>

<h2>Method and reproduction</h2>
<p>Splits are fixed per benchmark: <code>screen</code> ⊂ <code>select</code>, and both are
disjoint from <code>holdout</code>. Macro numbers weigh every benchmark equally. Their 95%
intervals come from a stratified bootstrap ({DEFAULT_BOOTSTRAP_RESAMPLES:,} resamples). ECE uses
10 equal-count bins per benchmark; the reliability diagrams use equal-width bins.</p>
<p>Every number on this page is in the <code>metrics.json</code> file next to it. More detail
is in <code>docs/research/selection.md</code>, <code>docs/research/calibration.md</code>,
<code>reports/speed_quality</code> and <code>reports/jev_vs_qwen</code>.</p>
<pre>uv run python scripts/make_summary_report.py</pre>
"""
    figures_json = json.dumps(figures, separators=(",", ":"), sort_keys=True)
    return f"""<!doctype html>
<html lang=en>
<head>
<meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Local and API models as stand-ins for Jev · jevemu summary report</title>
<script src="{PLOTLY_JS}" charset="utf-8"></script>
<style>{CSS}</style>
</head>
<body>
<main>
{body}
</main>
<script>
const FIGURES = {figures_json};
for (const [id, fig] of Object.entries(FIGURES)) {{
  Plotly.newPlot(id, fig.data, fig.layout, {{responsive: true, displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"]}});
}}
</script>
</body>
</html>
"""


# --- output ------------------------------------------------------------------------------------


def rounded(value: Any) -> Any:
    """``value`` with floats rounded to 6 significant decimals (stable, compact JSON)."""
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 6)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v) for v in value]
    return value


def build_figures(data: Mapping[str, Any]) -> dict[str, Any]:
    makers: dict[str, Callable[[], dict[str, Any]]] = {
        "fig-land-acc": lambda: landscape_figure(data, "accuracy"),
        "fig-land-brier": lambda: landscape_figure(data, "brier"),
        "fig-final-acc": lambda: finalists_figure(data, "accuracy"),
        "fig-final-brier": lambda: finalists_figure(data, "brier"),
        "fig-rel-pooled": lambda: reliability_pooled_figure(data),
        "fig-rel-dataset": lambda: reliability_by_dataset_figure(data),
    }
    return {key: rounded(make()) for key, make in makers.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, default=OUT, help="output directory")
    args = parser.parse_args()
    prices = PriceBook(gpu=GpuTimePrice.from_power(DEFAULT_GPU_WATTS, DEFAULT_USD_PER_WH))
    data = rounded(collect(prices))
    figures = build_figures(data)
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    (out / "index.html").write_text(render(data, figures))
    print(
        f"wrote {args.out}/index.html and metrics.json ({len(data['finalists'])} finalists, "
        f"{len(data['landscape'])} landscape points, {len(data['pending'])} pending)"
    )


if __name__ == "__main__":
    main()
