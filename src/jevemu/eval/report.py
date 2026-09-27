"""Calibration study of recorded runs: cross-fit, score, bootstrap and report.

:func:`run_calibration` takes named run directories of one split (Jev and emulator runs alike),
pairs them on the items every run answered, and for each *system* (a run, and for emulator runs
with diagnostics each requested offline debiasing ``run+batch``, ``run+pride``, ``run+none``;
``run+none`` is left out when it equals the run, i.e. the run had no debiaser) computes every
calibrator *arm* out of fold (:mod:`jevemu.eval.crossfit`), plus ``raw`` (the probabilities as
recorded, renormalized, no floor). Every system gets the same calibrators ("calibrate both or
neither") on the same folds. Metrics and paired intervals come from
:mod:`jevemu.eval.metrics_calib`: per benchmark and macro (equal weight per benchmark), with
Δ(arm - raw) per system, Δ(system - reference) per arm and Δ(debiased system - the same run's
undebiased system) per arm, all on shared resamples. Bootstrap replicates of ECE are biased
upward (resampling adds binning noise), so the percentile interval of an ECE, or of a Δ ECE,
can exclude its own estimate: ``report.md`` shows ECE as point estimates and ``metrics.json``
keeps those intervals only as an indication.

Each run is also scored *as returned*, the way a Jev user sees it (point estimates): NLL, Brier
and top-label ECE of the returned probabilities without renormalization, and the ECE of the
answers' ``confidence`` field read as the probability that the returned answer (Choice: its
``choice``; Score: the modal level) is correct. Noul answers have no ``confidence``.

:meth:`CalibrationReport.write` saves ``report.md``, ``metrics.json`` (estimates, intervals,
reliability bins, fitted calibrators, configuration) and ``predictions.<system>.jsonl`` (per
item: fold, gold, keys and every arm's probabilities).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from jevemu.bench.manifest import RunManifest
from jevemu.calibrate import DEFAULT_EPS
from jevemu.debias import full_cycle
from jevemu.eval.crossfit import (
    DEFAULT_CALIBRATORS,
    DEFAULT_FOLDS,
    DEFAULT_MIN_GROUP,
    CalibratorFit,
    CalibratorSpec,
    CalItem,
    DebiasMethod,
    assign_folds,
    crossfit_predict,
    debias_offline,
    load_run_items,
)
from jevemu.eval.metrics import DEFAULT_BOOTSTRAP_RESAMPLES, DEFAULT_ECE_BINS, ece
from jevemu.eval.metrics_calib import (
    METRICS,
    Interval,
    ItemMetrics,
    ReliabilityBin,
    bootstrap_replicates,
    interval,
    reliability,
)
from jevemu.eval.splits import SplitName

__all__ = ["RAW", "CalibrationReport", "run_calibration"]

FloatArray = NDArray[np.float64]

RAW = "raw"
"""The arm of recorded probabilities, uncalibrated and unfloored."""
DEFAULT_PRIMARY = "temperature@signature"
"""The design's recommended calibrator: the 2x2 grid's "calibrated" column."""
MACRO = "macro"

_METRIC_NAMES = {"accuracy": "Accuracy", "nll": "NLL", "brier": "Brier", "ece": "ECE"}
_DIGITS = {"accuracy": 4, "nll": 3, "brier": 3, "ece": 3}
_DEBIAS_DESCRIPTIONS: dict[str, str] = {
    "none": "the strategy's probabilities before any debiasing",
    "batch": "offline batch calibration, priors fitted out of fold",
    "pride": (
        "offline PriDe, priors fitted out of fold; a question recorded under every cyclic "
        "shift answers with its permutation average"
    ),
}
AS_RETURNED_METRICS: tuple[str, ...] = ("nll", "brier", "ece", "ece_confidence")
"""Metrics of :attr:`CalibrationReport.as_returned` (module docstring)."""


@dataclass
class CalibrationReport:
    """Everything :func:`run_calibration` computed; intervals are built on demand."""

    split: str
    systems: list[str]
    arms: list[str]
    benchmarks: list[str]
    reference: str | None
    primary: str
    config: dict[str, Any]
    n_items: dict[str, int]
    estimates: dict[tuple[str, str, str], dict[str, float]]
    """(system, arm, benchmark) -> metric -> value; benchmark ``"macro"`` included."""
    replicates: dict[tuple[str, str, str], dict[str, FloatArray]]
    reliability: dict[tuple[str, str, str], list[ReliabilityBin]]
    fits: dict[tuple[str, str], list[CalibratorFit]]
    predictions: dict[str, list[dict[str, Any]]] = field(repr=False)
    notes: list[str] = field(default_factory=list)
    as_returned: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    """(run, benchmark) -> :data:`AS_RETURNED_METRICS` (plus ``n_confidence``, the answers with
    a ``confidence``); benchmark ``"macro"`` included, its ``ece_confidence`` over the
    benchmarks that have one."""
    baselines: dict[str, str] = field(default_factory=dict)
    """Debiased system (online or offline) -> the same run's undebiased system."""

    # -- intervals --

    def value(self, system: str, arm: str, benchmark: str, metric: str) -> Interval:
        key = (system, arm, benchmark)
        return interval(self.estimates[key][metric], self.replicates[key][metric])

    def delta(
        self, a: tuple[str, str], b: tuple[str, str], benchmark: str, metric: str
    ) -> Interval:
        """``a - b`` for (system, arm) pairs, paired on shared resamples."""
        ka, kb = (*a, benchmark), (*b, benchmark)
        return interval(
            self.estimates[ka][metric] - self.estimates[kb][metric],
            self.replicates[ka][metric] - self.replicates[kb][metric],
        )

    # -- output --

    def to_json(self) -> dict[str, Any]:
        series: dict[str, Any] = {}
        for system in self.systems:
            for arm in self.arms:
                entry: dict[str, Any] = {}
                for benchmark in [*self.benchmarks, MACRO]:
                    entry[benchmark] = {
                        metric: self.value(system, arm, benchmark, metric).to_json()
                        for metric in METRICS
                    }
                    if arm != RAW:
                        entry[benchmark]["delta_vs_raw"] = {
                            metric: self.delta(
                                (system, arm), (system, RAW), benchmark, metric
                            ).to_json()
                            for metric in METRICS
                        }
                    if self.reference is not None and system != self.reference:
                        entry[benchmark]["delta_vs_reference"] = {
                            metric: self.delta(
                                (system, arm), (self.reference, arm), benchmark, metric
                            ).to_json()
                            for metric in METRICS
                        }
                    baseline = self.baselines.get(system)
                    if baseline is not None:
                        entry[benchmark]["delta_vs_undebiased"] = {
                            metric: self.delta(
                                (system, arm), (baseline, arm), benchmark, metric
                            ).to_json()
                            for metric in METRICS
                        }
                    bins = self.reliability.get((system, arm, benchmark))
                    if bins is not None:
                        entry[benchmark]["reliability"] = [vars(b) for b in bins]
                series.setdefault(system, {})[arm] = entry
        return {
            "split": self.split,
            "systems": self.systems,
            "arms": self.arms,
            "benchmarks": self.benchmarks,
            "reference": self.reference,
            "primary": self.primary,
            "config": self.config,
            "n_items": self.n_items,
            "intervals": "estimate, 95% percentile bootstrap low, high",
            "series": series,
            "calibrators": {
                system: {
                    arm: [vars(fit) for fit in fits]
                    for (s, arm), fits in self.fits.items()
                    if s == system
                }
                for system in self.systems
            },
            "notes": self.notes,
            "as_returned": {
                run: {b: dict(values) for (r, b), values in self.as_returned.items() if r == run}
                for run in dict.fromkeys(r for r, _ in self.as_returned)
            },
        }

    def write(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_json(), indent=1, allow_nan=True)
        (out / "metrics.json").write_text(text + "\n", encoding="utf-8")
        (out / "report.md").write_text(self.markdown(), encoding="utf-8")
        for system, rows in self.predictions.items():
            path = out / f"predictions.{system}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, separators=(",", ":")) + "\n")

    def markdown(self) -> str:
        n_total = sum(self.n_items.values())
        lines = [
            f"# Calibration study ({self.split})",
            "",
            f"Systems: {', '.join(f'`{s}`' for s in self.systems)}. "
            f"{n_total} items in {len(self.benchmarks)} benchmarks, answered by every system. "
            f"{self.config['n_folds']}-fold cross-fitting by question id (seed "
            f"{self.config['seed']}); ε = {self.config['eps']}; groups under "
            f"{self.config['min_group']} training items use the fold's global temperature. "
            f"95% percentile intervals from {self.config['n_resamples']} paired cluster "
            "resamples (macro: stratified by benchmark, equal weights); ECE is a point "
            "estimate (10 equal-mass bins).",
            "",
            *[f"- {note}" for note in self.notes],
            "",
            f"## 2x2 grid (macro, calibrated = `{self.primary}`)",
            "",
            self._grid(),
            "## Every arm (macro)",
            "",
            self._arms_table(),
            "## Best arm per system (macro)",
            "",
            self._best_arms(),
        ]
        if self.as_returned:
            lines += [
                "## As returned (Jev-style, point estimates)",
                "",
                "Returned probabilities without renormalization, and the answers' `confidence` "
                "field read as P(returned answer correct): Choice's `choice`, Score's modal "
                'level; Noul has none, so its macro averages the other benchmarks. "Max Δ" is '
                "the largest absolute macro difference from the renormalized `raw` arm over "
                "NLL, Brier and ECE.",
                "",
                self._as_returned_macro(),
                self._as_returned_per_benchmark(),
            ]
        if self.reference is not None and len(self.systems) > 1:
            pairs = [(s, self.reference) for s in self.systems if s != self.reference]
            lines += [f"## Each system minus `{self.reference}` (macro)", "", self._deltas(pairs)]
        if self.baselines:
            lines += [
                "## Debiasing: each debiased system minus the undebiased emulator (macro)",
                "",
                self._deltas(list(self.baselines.items())),
            ]
        for system in self.systems:
            lines += [
                f"## `{system}` per benchmark: raw → `{self.primary}`",
                "",
                self._per_benchmark(system),
            ]
        lines += ["## Fitted temperatures (mean over folds)", "", self._temperatures()]
        return "\n".join(lines)

    def _fmt(self, value: Interval, metric: str, *, signed: bool = False) -> str:
        """``estimate [low, high]``; ECE and Δ ECE without an interval (module docstring)."""
        digits = _DIGITS[metric]
        spec = f"+.{digits}f" if signed else f".{digits}f"
        if metric == "ece" or math.isnan(value.low):
            return format(value.estimate, spec)
        return (
            f"{format(value.estimate, spec)} [{format(value.low, spec)}, "
            f"{format(value.high, spec)}]"
        )

    def _grid(self) -> str:
        header = ["System"]
        for metric in ("nll", "brier", "ece", "accuracy"):
            name = _METRIC_NAMES[metric]
            header += [f"{name} raw", f"{name} calibrated", f"Δ {name}"]
        rows = []
        for system in self.systems:
            row = [f"`{system}`"]
            for metric in ("nll", "brier", "ece", "accuracy"):
                row += [
                    self._fmt(self.value(system, RAW, MACRO, metric), metric),
                    self._fmt(self.value(system, self.primary, MACRO, metric), metric),
                    self._fmt(
                        self.delta((system, self.primary), (system, RAW), MACRO, metric),
                        metric,
                        signed=True,
                    ),
                ]
            rows.append(row)
        return _table(header, rows)

    def _arms_table(self) -> str:
        header = ["System", "Arm", *(_METRIC_NAMES[m] for m in METRICS), "Δ NLL vs raw"]
        rows = []
        for system in self.systems:
            for arm in self.arms:
                delta = (
                    ""
                    if arm == RAW
                    else self._fmt(
                        self.delta((system, arm), (system, RAW), MACRO, "nll"), "nll", signed=True
                    )
                )
                rows.append(
                    [
                        f"`{system}`",
                        f"`{arm}`",
                        *(self._fmt(self.value(system, arm, MACRO, m), m) for m in METRICS),
                        delta,
                    ]
                )
        return _table(header, rows)

    def _best_arms(self) -> str:
        """Per system, the arm with the best macro value of each metric (first arm on ties)."""
        header = ["System", *(f"Best {_METRIC_NAMES[m]}" for m in METRICS)]
        rows = []
        for system in self.systems:
            row = [f"`{system}`"]
            for metric in METRICS:
                values = {arm: self.estimates[(system, arm, MACRO)][metric] for arm in self.arms}
                pick = max if metric == "accuracy" else min
                best = pick(values, key=values.__getitem__)
                row.append(f"`{best}` {values[best]:.{_DIGITS[metric]}f}")
            rows.append(row)
        return _table(header, rows)

    def _as_returned_macro(self) -> str:
        header = [
            "Run",
            "NLL",
            "Brier",
            "ECE (top probability)",
            "ECE (`confidence`)",
            "Max Δ vs renormalized",
        ]
        rows = []
        for run in dict.fromkeys(r for r, _ in self.as_returned):
            values = self.as_returned[(run, MACRO)]
            renormalized = self.estimates[(run, RAW, MACRO)]
            gap = max(abs(values[m] - renormalized[m]) for m in ("nll", "brier", "ece"))
            rows.append(
                [
                    f"`{run}`",
                    *(f"{values[m]:.4f}" for m in AS_RETURNED_METRICS),
                    f"{gap:.1e}",
                ]
            )
        return _table(header, rows)

    def _as_returned_per_benchmark(self) -> str:
        runs = list(dict.fromkeys(r for r, _ in self.as_returned))
        header = ["Benchmark"]
        for run in runs:
            header += [f"`{run}` ECE (top probability)", f"`{run}` ECE (`confidence`)"]
        rows = []
        for benchmark in [*self.benchmarks, MACRO]:
            row = [f"**{benchmark}**" if benchmark == MACRO else benchmark]
            for run in runs:
                values = self.as_returned[(run, benchmark)]
                confidence = values["ece_confidence"]
                row += [
                    f"{values['ece']:.4f}",
                    "" if math.isnan(confidence) else f"{confidence:.4f}",
                ]
            rows.append(row)
        return _table(header, rows)

    def _deltas(self, pairs: Sequence[tuple[str, str]]) -> str:
        """Δ(system - other) per metric for the raw and primary arms, paired."""
        header = ["System", "Minus", "Arm", *(f"Δ {_METRIC_NAMES[m]}" for m in METRICS)]
        rows = []
        for system, other in pairs:
            for arm in (RAW, self.primary):
                rows.append(
                    [
                        f"`{system}`",
                        f"`{other}`",
                        f"`{arm}`",
                        *(
                            self._fmt(
                                self.delta((system, arm), (other, arm), MACRO, m), m, signed=True
                            )
                            for m in METRICS
                        ),
                    ]
                )
        return _table(header, rows)

    def _per_benchmark(self, system: str) -> str:
        header = ["Benchmark", "n", "Accuracy"]
        for metric in ("nll", "brier", "ece"):
            name = _METRIC_NAMES[metric]
            header += [f"{name} raw", f"{name} cal.", f"Δ {name}"]
        rows = []
        for benchmark in [*self.benchmarks, MACRO]:
            n = sum(self.n_items.values()) if benchmark == MACRO else self.n_items[benchmark]
            label = f"**{benchmark}**" if benchmark == MACRO else benchmark
            row = [
                label,
                str(n),
                self._fmt(self.value(system, RAW, benchmark, "accuracy"), "accuracy"),
            ]
            for metric in ("nll", "brier", "ece"):
                row += [
                    format(
                        self.estimates[(system, RAW, benchmark)][metric], f".{_DIGITS[metric]}f"
                    ),
                    format(
                        self.estimates[(system, self.primary, benchmark)][metric],
                        f".{_DIGITS[metric]}f",
                    ),
                    self._fmt(
                        self.delta((system, self.primary), (system, RAW), benchmark, metric),
                        metric,
                        signed=True,
                    ),
                ]
            rows.append(row)
        return _table(header, rows)

    def _temperatures(self) -> str:
        header = ["System", "Arm", "Group", "Folds", "Mean T", "Range"]
        rows = []
        for (system, arm), fits in self.fits.items():
            if not arm.startswith("temperature@"):
                continue
            by_group: dict[str, list[float]] = {}
            for fit in fits:
                if not fit.fallback:
                    by_group.setdefault(fit.group, []).append(float(fit.params["temperature"]))
            for group, temps in sorted(by_group.items()):
                rows.append(
                    [
                        f"`{system}`",
                        f"`{arm}`",
                        group,
                        str(len(temps)),
                        f"{np.mean(temps):.3f}",
                        f"{min(temps):.3f}-{max(temps):.3f}",
                    ]
                )
        return _table(header, rows)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


# --- the study ---------------------------------------------------------------------------------


def _align(
    runs: Mapping[str, dict[str, list[CalItem]]],
) -> tuple[list[str], dict[str, dict[str, list[CalItem]]], list[str]]:
    """Benchmarks every run has, each run's items restricted to the shared item ids (first
    run's order), and notes on what was dropped."""
    names = list(runs)
    first = runs[names[0]]
    benchmarks = [b for b in first if all(b in runs[n] for n in names)]
    notes = []
    dropped = sorted({b for n in names for b in runs[n]} - set(benchmarks))
    if dropped:
        notes.append(f"Benchmarks not answered by every run, left out: {', '.join(dropped)}.")
    aligned: dict[str, dict[str, list[CalItem]]] = {n: {} for n in names}
    for benchmark in benchmarks:
        shared = set.intersection(*({i.item_id for i in runs[n][benchmark]} for n in names))
        order = [i.item_id for i in first[benchmark] if i.item_id in shared]
        missing = max(len(runs[n][benchmark]) for n in names) - len(order)
        if missing:
            notes.append(f"{benchmark}: {missing} items not answered by every run, left out.")
        for n in names:
            by_id = {i.item_id: i for i in runs[n][benchmark]}
            aligned[n][benchmark] = [by_id[item_id] for item_id in order]
        reference = aligned[names[0]][benchmark]
        for n in names[1:]:
            for a, b in zip(reference, aligned[n][benchmark], strict=True):
                if (a.keys, a.gold) != (b.keys, b.gold):
                    raise ValueError(f"{a.item_id}: runs {names[0]} and {n} disagree on keys/gold")
    return benchmarks, aligned, notes


def run_calibration(
    runs: Mapping[str, str | Path],
    *,
    split: SplitName = "holdout",
    splits_dir: str | Path | None = None,
    benchmarks: Sequence[str] | None = None,
    debias: Sequence[DebiasMethod] = (),
    calibrators: Sequence[str] = DEFAULT_CALIBRATORS,
    primary: str = DEFAULT_PRIMARY,
    reference: str | None = None,
    n_folds: int = DEFAULT_FOLDS,
    seed: int = 0,
    eps: float = DEFAULT_EPS,
    min_group: int = DEFAULT_MIN_GROUP,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    n_bins: int = DEFAULT_ECE_BINS,
) -> CalibrationReport:
    """Run the study (module docstring) on ``runs`` (system name -> run directory)."""
    if not runs:
        raise ValueError("run_calibration needs at least one run")
    specs = [CalibratorSpec.parse(text) for text in calibrators]
    arms = [RAW, *(spec.label for spec in specs)]
    if primary not in arms:
        raise ValueError(f"primary {primary!r} is not one of the arms {arms}")
    if reference is not None and reference not in runs:
        raise ValueError(f"reference {reference!r} is not one of the runs {list(runs)}")
    loaded = {
        name: load_run_items(path, split=split, splits_dir=splits_dir, benchmarks=benchmarks)
        for name, path in runs.items()
    }
    names, aligned, notes = _align(loaded)
    if not names:
        raise ValueError("the runs share no benchmark")
    first = aligned[next(iter(runs))]
    folds = assign_folds((i for b in names for i in first[b]), n_folds=n_folds, seed=seed)

    systems: dict[str, tuple[str, DebiasMethod | None]] = {}
    baselines: dict[str, str] = {}
    for run, path in runs.items():
        systems[run] = (run, None)
        run_items = [i for b in names for i in aligned[run][b]]
        if not debias:
            continue
        if any(i.raw is None for i in run_items):
            notes.append(
                f"`{run}`: black box, no pre-debias scores → debiasing arms skipped; "
                "calibration uses its returned probabilities (renormalized)."
            )
            continue
        manifest = RunManifest.load(Path(path))
        debiaser = None if manifest is None else manifest.system.get("debiaser")
        described = [f"`{run}`: as recorded (debiaser: {f'`{debiaser}`' if debiaser else 'none'})"]
        undebiased: str | None = None
        for debias_method in debias:
            name = f"{run}+{debias_method}"
            if debias_method == "none" and all(
                np.allclose(i.raw, i.probs, rtol=0.0, atol=1e-9)
                for i in run_items
                if i.raw is not None
            ):
                described.append(f"`{name}` left out, identical to `{run}` (no debiasing)")
                undebiased = run
                continue
            if debias_method == "pride" and not any(
                full_cycle(i.presentations, i.free) for i in run_items
            ):
                described.append(
                    f"no `{name}`: no question recorded under all cyclic shifts (run with the "
                    "pride or permutation debiaser)"
                )
                continue
            systems[name] = (run, debias_method)
            described.append(f"`{name}`: {_DEBIAS_DESCRIPTIONS[debias_method]}")
            if debias_method == "none":
                undebiased = name
        notes.append("; ".join(described) + ".")
        if undebiased is not None:
            for system, (source, _) in systems.items():
                if source == run and system != undebiased:
                    baselines[system] = undebiased

    per_item: dict[str, dict[str, dict[str, FloatArray]]] = {}
    fits: dict[tuple[str, str], list[CalibratorFit]] = {}
    for system, (run, method) in systems.items():
        items = [i for b in names for i in aligned[run][b]]
        base = (
            {i.item_id: i.probs for i in items}
            if method is None
            else debias_offline(items, method, folds)
        )
        arm_probs = {RAW: base}
        for spec in specs:
            arm_probs[spec.label], fits[(system, spec.label)] = crossfit_predict(
                items, base, folds, spec, eps=eps, min_group=min_group
            )
        per_item[system] = arm_probs

    estimates: dict[tuple[str, str, str], dict[str, float]] = {}
    replicates: dict[tuple[str, str, str], dict[str, FloatArray]] = {}
    bins: dict[tuple[str, str, str], list[ReliabilityBin]] = {}
    n_items = {}
    for index, benchmark in enumerate(names):
        rows = first[benchmark]
        n_items[benchmark] = len(rows)
        gold = [i.gold for i in rows]
        series = {}
        for system, (run, _) in systems.items():
            for arm in arms:
                metrics = ItemMetrics.from_probs(
                    [per_item[system][arm][i.item_id] for i in aligned[run][benchmark]], gold
                )
                series[(system, arm)] = metrics
                estimates[(system, arm, benchmark)] = metrics.estimate(n_bins=n_bins)
                bins[(system, arm, benchmark)] = reliability(
                    metrics.confidence, metrics.correct, n_bins=n_bins
                )
        reps = bootstrap_replicates(
            [i.cluster for i in rows],
            {f"{s}\x1f{a}": m for (s, a), m in series.items()},
            n_resamples=n_resamples,
            seed=seed + index,
            n_bins=n_bins,
        )
        for system, arm in series:
            replicates[(system, arm, benchmark)] = reps[f"{system}\x1f{arm}"]
    for system in systems:
        for arm in arms:
            estimates[(system, arm, MACRO)] = {
                m: float(np.mean([estimates[(system, arm, b)][m] for b in names])) for m in METRICS
            }
            replicates[(system, arm, MACRO)] = {
                m: np.mean([replicates[(system, arm, b)][m] for b in names], axis=0)
                for m in METRICS
            }

    as_returned: dict[tuple[str, str], dict[str, float]] = {}
    for run in runs:
        for benchmark in names:
            rows = aligned[run][benchmark]
            metrics = ItemMetrics.from_probs([i.returned for i in rows], [i.gold for i in rows])
            values = metrics.estimate(n_bins=n_bins)
            # confidence is about the returned answer: Choice's ``choice``, Score's mode
            rated = [i for i in rows if i.confidence is not None]
            confidence = np.array([i.confidence for i in rated], dtype=np.float64)
            answered = [int(np.argmax(i.returned)) if i.choice is None else i.choice for i in rated]
            correct = np.array([a == i.gold for a, i in zip(answered, rated, strict=True)])
            as_returned[(run, benchmark)] = {
                "nll": values["nll"],
                "brier": values["brier"],
                "ece": values["ece"],
                "ece_confidence": ece(confidence, correct, n_bins=n_bins),
                "n_confidence": float(len(rated)),
            }
        per_benchmark = [as_returned[(run, b)] for b in names]
        with_confidence = [
            v["ece_confidence"] for v in per_benchmark if not math.isnan(v["ece_confidence"])
        ]
        as_returned[(run, MACRO)] = {
            **{m: float(np.mean([v[m] for v in per_benchmark])) for m in ("nll", "brier", "ece")},
            "ece_confidence": float(np.mean(with_confidence)) if with_confidence else math.nan,
            "n_confidence": float(sum(v["n_confidence"] for v in per_benchmark)),
        }

    predictions: dict[str, list[dict[str, Any]]] = {}
    for system, (run, _) in systems.items():
        predictions[system] = [
            {
                "item_id": i.item_id,
                "benchmark": i.benchmark,
                "question_id": i.question_id,
                "fold": folds[i.item_id],
                "gold": i.gold,
                "keys": list(i.keys),
                "arms": {
                    arm: [round(float(p), 6) for p in per_item[system][arm][i.item_id]]
                    for arm in arms
                },
            }
            for b in names
            for i in aligned[run][b]
        ]

    return CalibrationReport(
        split=split,
        systems=list(systems),
        arms=arms,
        benchmarks=names,
        reference=reference,
        primary=primary,
        config={
            "runs": {name: str(path) for name, path in runs.items()},
            "n_folds": n_folds,
            "seed": seed,
            "eps": eps,
            "min_group": min_group,
            "n_resamples": n_resamples,
            "n_bins": n_bins,
            "calibrators": list(calibrators),
            "debias": list(debias),
        },
        n_items=n_items,
        estimates=estimates,
        replicates=replicates,
        reliability=bins,
        fits=fits,
        predictions=predictions,
        notes=notes,
        as_returned=as_returned,
        baselines=baselines,
    )
