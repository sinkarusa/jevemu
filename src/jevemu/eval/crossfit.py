"""Cross-fitted calibration of recorded runs.

Inputs are run directories written by :func:`~jevemu.bench.runner.run_split` (one records file
per ``<benchmark>.<split>``) plus the frozen split files they answered (``splits_dir``, by
default the run directory's sibling ``splits/``), which give each item's question (and so its
answer keys), ``question_id`` and stratum.

- **Items** (:func:`load_run_items`). One :class:`CalItem` per successful record: the answer's
  probabilities in the question's key order, renormalized (Jev rounds to 0.01) and as returned,
  its ``confidence`` field, the gold key's index, the question signature and the debiasing
  positions. Emulator records also carry the strategy's raw probabilities and any permuted
  presentations from ``x_jevemu``.
- **Folds** (:func:`assign_folds`). ``n_folds`` folds per benchmark by question id, stratified
  by stratum: within each stratum (strata in name order) question ids are ordered by
  ``sha256(f"{seed}\\x1f{benchmark}\\x1f{question_id}")`` and dealt round-robin, the deal
  continuing from one stratum to the next, so every stratum and the benchmark spread evenly
  and every item of a question id shares its fold. Folds depend only on the split, so every
  system gets the same folds.
- **Offline debiasing** (:func:`debias_offline`, emulator runs only): ``none`` (the strategy's
  raw probabilities), ``batch`` and ``pride``. Priors are fitted out of fold like calibrators:
  for fold ``f`` from the other folds' questions (``pride``: those recorded with permuted
  presentations), then divided out of fold ``f``. A ``pride`` question with its own
  presentations is answered with their average.
- **Calibration** (:func:`crossfit_predict`). A :class:`CalibratorSpec` ``name@scope`` fits
  calibrator ``name`` per group: ``global`` (one for all items), ``benchmark`` or
  ``signature`` (question signature, pooled across benchmarks, as the calibrator registry
  keys it). For fold ``f`` it is fitted on the other folds and predicts fold ``f``, so no item
  is ever calibrated by a model that saw it. Inputs are ``probs_to_logits(p, eps)`` (the
  design's ε = 1e-4 floor for both systems); rows of different K in one group are padded with
  ``-inf``. A group with fewer than ``min_group`` training items falls back to the fold's
  global temperature.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from jevemu.bench.manifest import run_key
from jevemu.bench.runner import SPLITS_DIR, load_records
from jevemu.bench.spec import Gold
from jevemu.calibrate import (
    CALIBRATORS,
    DEFAULT_EPS,
    CalibrationKey,
    Calibrator,
    CalibratorRegistry,
    TemperatureCalibrator,
    probs_to_logits,
    question_signature,
)
from jevemu.debias import (
    LabelPriors,
    Presentation,
    apply_prior,
    average_presentations,
    fit_batch_priors,
    fit_pride_priors,
    free_positions,
    full_cycle,
    prior_key,
)
from jevemu.eval.splits import (
    DATASETS,
    DROPPED_DATASETS,
    FrozenSplit,
    SplitName,
    question_id_of,
    stratum_of,
)
from jevemu.render import answer_keys
from jevemu.types import (
    Answer,
    ChoiceAnswer,
    EmulatorDiagnostics,
    NoulAnswer,
    Question,
    ScoreAnswer,
)

__all__ = [
    "DEBIAS_METHODS",
    "DEFAULT_CALIBRATORS",
    "DEFAULT_FOLDS",
    "DEFAULT_MIN_GROUP",
    "CalItem",
    "CalibratorFit",
    "CalibratorSpec",
    "DebiasMethod",
    "Scope",
    "apply_label_priors",
    "assign_folds",
    "crossfit_predict",
    "debias_offline",
    "fit_registry",
    "load_run_items",
]

FloatArray = NDArray[np.float64]

Scope = Literal["global", "benchmark", "signature"]
SCOPES: tuple[Scope, ...] = ("global", "benchmark", "signature")
DebiasMethod = Literal["none", "batch", "pride"]
DEBIAS_METHODS: tuple[DebiasMethod, ...] = ("none", "batch", "pride")

DEFAULT_FOLDS = 5
DEFAULT_MIN_GROUP = 30
"""Signature/benchmark groups with fewer training items use the fold's global temperature."""

DEFAULT_CALIBRATORS: tuple[str, ...] = (
    "identity@global",
    "temperature@global",
    "temperature@benchmark",
    "temperature@signature",
    "vector@signature",
    "platt@signature",
    "isotonic@signature",
    "histogram@signature",
)
"""The calibrator arms :func:`crossfit_predict` runs by default (every registered calibrator)."""


# --- items -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CalItem:
    """One answered item, ready for cross-fitting."""

    item_id: str
    benchmark: str
    question_id: str
    stratum: str
    signature: str
    """:func:`~jevemu.calibrate.question_signature` of the question."""
    prior_key: str
    """:func:`~jevemu.debias.prior_key` (signature plus fixed "I don't know" position)."""
    free: tuple[bool, ...]
    keys: tuple[str, ...]
    gold: int
    """Index of the gold key in ``keys``."""
    probs: FloatArray
    """The answer's probabilities in key order, renormalized."""
    returned: FloatArray
    """The answer's probabilities in key order as returned (Jev's do not sum to exactly 1)."""
    raw: FloatArray | None = None
    """Emulator only: the strategy's probabilities before debiasing and calibration."""
    presentations: tuple[Presentation, ...] = ()
    """Emulator only: permuted presentations recorded by a permuting debiaser."""
    confidence: float | None = None
    """The answer's ``confidence`` field (Choice and Score; ``None`` for Noul)."""
    choice: int | None = None
    """Choice answers: index of the returned ``choice`` (Jev's can differ from the argmax of
    its rounded probabilities on near ties)."""

    @property
    def cluster(self) -> str:
        """Bootstrap cluster: every item of a question id resamples together."""
        return f"{self.benchmark}\x1f{self.question_id}"


def _normalized(values: Sequence[float]) -> FloatArray:
    array = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError("a probability vector has no mass")
    out: FloatArray = array / total
    return out


def _answer_probs(answer: Answer, keys: Sequence[str]) -> list[float]:
    if isinstance(answer, NoulAnswer):
        return [answer.noul, 1.0 - answer.noul]
    return [answer.probabilities.get(key, 0.0) for key in keys]


def _gold_index(question: Question, keys: Sequence[str], gold: Gold) -> int:
    if question.type == "noul":
        if not isinstance(gold, bool):
            raise TypeError(f"noul gold must be a bool, got {gold!r}")
        return 0 if gold else 1
    if question.type == "score":
        if isinstance(gold, bool) or not isinstance(gold, int):
            raise TypeError(f"score gold must be a level index, got {gold!r}")
        return gold
    return list(keys).index(str(gold))


def _presentations(
    diagnostics: EmulatorDiagnostics, keys: Sequence[str]
) -> tuple[Presentation, ...]:
    position = {key: i for i, key in enumerate(keys)}
    out = []
    for score in diagnostics.permutation_scores:
        order = tuple(position[key] for key in score.order)
        with np.errstate(divide="ignore"):
            logprobs = np.log([score.probabilities[key] for key in score.order])
        out.append(Presentation(order, tuple(float(v) for v in logprobs)))
    return tuple(out)


def load_run_items(
    run_dir: str | Path,
    *,
    split: SplitName = "holdout",
    splits_dir: str | Path | None = None,
    benchmarks: Iterable[str] | None = None,
) -> dict[str, list[CalItem]]:
    """Benchmark -> its successfully answered items (split order), for every
    ``<benchmark>.<split>.jsonl`` in ``run_dir`` but :data:`DROPPED_DATASETS` (or only
    ``benchmarks``)."""
    run_path = Path(run_dir)
    split_path = Path(splits_dir) if splits_dir is not None else run_path.parent / SPLITS_DIR
    suffix = f".{split}.jsonl"
    found = sorted(file.name[: -len(suffix)] for file in run_path.glob(f"*{suffix}"))
    wanted = (
        [b for b in found if b not in DROPPED_DATASETS]
        if benchmarks is None
        else [b for b in benchmarks if b in found]
    )
    out: dict[str, list[CalItem]] = {}
    for benchmark in _benchmark_order(wanted):
        frozen = FrozenSplit.from_jsonl(split_path / f"{run_key(benchmark, split)}.jsonl")
        records = load_records(run_path / f"{run_key(benchmark, split)}.jsonl")
        items = []
        for bench_item in frozen.items:
            record = records.get(bench_item.item_id)
            if record is None or record.answer is None:
                continue
            question = bench_item.question
            keys = answer_keys(question)
            free = free_positions(question)
            diagnostics = record.diagnostics
            raw = None
            if diagnostics is not None:
                raw = _normalized([diagnostics.raw_probabilities.get(k, 0.0) for k in keys])
            answer = record.answer
            returned = _answer_probs(answer, keys)
            items.append(
                CalItem(
                    item_id=bench_item.item_id,
                    benchmark=benchmark,
                    question_id=question_id_of(bench_item),
                    stratum=stratum_of(bench_item),
                    signature=question_signature(question),
                    prior_key=prior_key(question, free),
                    free=free,
                    keys=keys,
                    gold=_gold_index(question, keys, record.gold),
                    probs=_normalized(returned),
                    returned=np.asarray(returned, dtype=np.float64),
                    raw=raw,
                    presentations=() if diagnostics is None else _presentations(diagnostics, keys),
                    confidence=(
                        answer.confidence
                        if isinstance(answer, (ChoiceAnswer, ScoreAnswer))
                        else None
                    ),
                    choice=keys.index(answer.choice) if isinstance(answer, ChoiceAnswer) else None,
                )
            )
        out[benchmark] = items
    return out


def _benchmark_order(names: Iterable[str]) -> list[str]:
    names = list(names)
    known = [name for name in DATASETS if name in names]
    return known + sorted(set(names) - set(known))


# --- folds -------------------------------------------------------------------------------------


def assign_folds(
    items: Iterable[CalItem], *, n_folds: int = DEFAULT_FOLDS, seed: int = 0
) -> dict[str, int]:
    """Item id -> fold in ``[0, n_folds)`` (module docstring)."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    by_benchmark: defaultdict[str, dict[str, str]] = defaultdict(dict)
    members: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for item in items:
        strata = by_benchmark[item.benchmark]
        if strata.setdefault(item.question_id, item.stratum) != item.stratum:
            raise ValueError(f"{item.benchmark}: question {item.question_id!r} spans two strata")
        members[(item.benchmark, item.question_id)].append(item.item_id)
    folds: dict[str, int] = {}
    for benchmark, strata in by_benchmark.items():
        by_stratum: defaultdict[str, list[str]] = defaultdict(list)
        for question_id, stratum in strata.items():
            by_stratum[stratum].append(question_id)
        dealt = 0
        for stratum in sorted(by_stratum):
            ranked = sorted(by_stratum[stratum], key=lambda q: _fold_hash(seed, benchmark, q))
            for question_id in ranked:
                for item_id in members[(benchmark, question_id)]:
                    folds[item_id] = dealt % n_folds
                dealt += 1
    return folds


def _fold_hash(seed: int, benchmark: str, question_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\x1f{benchmark}\x1f{question_id}".encode()).digest()


# --- offline debiasing -------------------------------------------------------------------------


def _log(values: FloatArray) -> FloatArray:
    with np.errstate(divide="ignore"):
        out: FloatArray = np.log(values)
    return out


def _raw_by_id(items: Sequence[CalItem]) -> dict[str, FloatArray]:
    missing = [item.item_id for item in items if item.raw is None]
    if missing:
        raise ValueError(
            f"{len(missing)} items have no raw probabilities (run the emulator with "
            f"diagnostics), e.g. {missing[0]!r}"
        )
    return {item.item_id: item.raw for item in items if item.raw is not None}


def _divided(probs: FloatArray, item: CalItem, priors: LabelPriors) -> FloatArray:
    """``probs`` divided by the prior of ``item``'s prior key (unchanged without one)."""
    prior = priors.priors.get(item.prior_key)
    return probs if prior is None else np.exp(apply_prior(_log(probs), prior, item.free))


def apply_label_priors(items: Sequence[CalItem], priors: LabelPriors) -> dict[str, FloatArray]:
    """Item id -> its raw probabilities divided by its prior key's prior (unchanged without
    one): the answer of ``BatchDebiaser(priors)`` or ``PriDeDebiaser(priors=priors)`` to the
    recorded question, the input a deployed calibrator sees behind that debiaser."""
    raw = _raw_by_id(items)
    return {item.item_id: _divided(raw[item.item_id], item, priors) for item in items}


def debias_offline(
    items: Sequence[CalItem],
    method: DebiasMethod,
    folds: Mapping[str, int],
    *,
    min_batch: int = DEFAULT_MIN_GROUP,
) -> dict[str, FloatArray]:
    """Item id -> probabilities after ``method`` (module docstring), priors fitted out of fold.

    Needs every item's raw probabilities (``x_jevemu`` diagnostics). A question whose prior
    key has no prior in the training folds (``batch``: fewer than ``min_batch`` training
    questions) keeps its raw probabilities.
    """
    raw = _raw_by_id(items)
    if method == "none":
        return raw
    out: dict[str, FloatArray] = {}
    for fold in sorted(set(folds[item.item_id] for item in items)):
        train = [item for item in items if folds[item.item_id] != fold]
        test = [item for item in items if folds[item.item_id] == fold]
        if method == "batch":
            priors = fit_batch_priors(
                ((item.prior_key, raw[item.item_id].tolist(), item.free) for item in train),
                min_count=min_batch,
            )
        elif method == "pride":
            priors = fit_pride_priors(
                (item.prior_key, item.presentations, item.free)
                for item in train
                if full_cycle(item.presentations, item.free)
            )
        else:
            raise ValueError(f"unknown debias method {method!r}; known: {DEBIAS_METHODS}")
        for item in test:
            if method == "pride" and full_cycle(item.presentations, item.free):
                out[item.item_id] = np.exp(average_presentations(item.presentations))
            else:
                out[item.item_id] = _divided(raw[item.item_id], item, priors)
    return out


# --- calibration -------------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibratorSpec:
    """Calibrator ``name`` fitted per ``scope`` group, written ``"name@scope"``."""

    name: str
    scope: Scope

    @classmethod
    def parse(cls, text: str) -> CalibratorSpec:
        name, _, scope_text = text.partition("@")
        if name not in CALIBRATORS:
            raise ValueError(f"unknown calibrator {name!r}; known: {sorted(CALIBRATORS)}")
        for scope in SCOPES:
            if scope == (scope_text or "global"):
                return cls(name, scope)
        raise ValueError(f"unknown scope {scope_text!r}; known: {SCOPES}")

    @property
    def label(self) -> str:
        return f"{self.name}@{self.scope}"

    def group(self, item: CalItem) -> str:
        if self.scope == "global":
            return "all"
        return item.benchmark if self.scope == "benchmark" else item.signature


@dataclass(frozen=True)
class CalibratorFit:
    """One fitted calibrator: its fold, group, training size and parameters."""

    fold: int
    group: str
    n_train: int
    fallback: bool
    """The group had too few training items and used the fold's global temperature."""
    params: dict[str, Any] = field(default_factory=dict)


def _padded_logits(
    items: Sequence[CalItem], probs: Mapping[str, FloatArray], eps: float
) -> FloatArray:
    width = max(len(item.keys) for item in items)
    z = np.full((len(items), width), -math.inf)
    for row, item in enumerate(items):
        values = probs[item.item_id]
        z[row, : values.shape[0]] = probs_to_logits(values, eps)
    return z


def _labels(items: Sequence[CalItem]) -> NDArray[np.int64]:
    return np.array([item.gold for item in items], dtype=np.int64)


def _unpad(items: Sequence[CalItem], calibrated: FloatArray) -> dict[str, FloatArray]:
    out = {}
    for row, item in enumerate(items):
        values = calibrated[row, : len(item.keys)]
        out[item.item_id] = values / values.sum()
    return out


def crossfit_predict(
    items: Sequence[CalItem],
    probs: Mapping[str, FloatArray],
    folds: Mapping[str, int],
    spec: CalibratorSpec,
    *,
    eps: float = DEFAULT_EPS,
    min_group: int = DEFAULT_MIN_GROUP,
    factory: Callable[[str], Calibrator] | None = None,
) -> tuple[dict[str, FloatArray], list[CalibratorFit]]:
    """Out-of-fold calibrated probabilities for every item, and every fitted calibrator.

    ``probs`` maps item id to its (uncalibrated) probabilities in key order. ``factory``
    builds an unfitted calibrator from its name (default: the registered class).
    """
    make = factory or (lambda name: CALIBRATORS[name]())
    out: dict[str, FloatArray] = {}
    fits: list[CalibratorFit] = []
    for fold in sorted(set(folds[item.item_id] for item in items)):
        train = [item for item in items if folds[item.item_id] != fold]
        test = [item for item in items if folds[item.item_id] == fold]
        if not train:
            raise ValueError(f"fold {fold} has no training items")
        global_t: TemperatureCalibrator | None = None
        train_groups: defaultdict[str, list[CalItem]] = defaultdict(list)
        for item in train:
            train_groups[spec.group(item)].append(item)
        test_groups: defaultdict[str, list[CalItem]] = defaultdict(list)
        for item in test:
            test_groups[spec.group(item)].append(item)
        for group, members in sorted(test_groups.items()):
            fit_rows = train_groups.get(group, [])
            calibrator: Calibrator
            if len(fit_rows) >= min_group:
                both = [*fit_rows, *members]
                z = _padded_logits(both, probs, eps)
                calibrator = make(spec.name).fit(z[: len(fit_rows)], _labels(fit_rows))
                calibrated = calibrator.transform(z[len(fit_rows) :])
                fits.append(CalibratorFit(fold, group, len(fit_rows), False, calibrator.to_json()))
            else:
                if global_t is None:
                    global_t = TemperatureCalibrator().fit(
                        _padded_logits(train, probs, eps), _labels(train)
                    )
                calibrator = global_t
                calibrated = global_t.transform(_padded_logits(members, probs, eps))
                fits.append(CalibratorFit(fold, group, len(fit_rows), True, global_t.to_json()))
            out.update(_unpad(members, calibrated))
    return out, fits


# --- deployment --------------------------------------------------------------------------------


def fit_registry(
    items: Sequence[CalItem],
    probs: Mapping[str, FloatArray],
    *,
    backend: str,
    model: str,
    template_id: str,
    name: str = "temperature",
    eps: float = DEFAULT_EPS,
    min_group: int = DEFAULT_MIN_GROUP,
) -> CalibratorRegistry:
    """A calibrator registry fitted on all of ``items`` (no folds: for deployment, after the
    cross-fitted study has chosen ``name``): ``name`` per question signature with at least
    ``min_group`` items, and a temperature over every item as the (model, template) fallback."""
    if not items:
        raise ValueError("fit_registry needs at least one item")
    registry = CalibratorRegistry()
    fallback = TemperatureCalibrator().fit(_padded_logits(items, probs, eps), _labels(items))
    registry.register_fallback(model, template_id, fallback)
    groups: defaultdict[str, list[CalItem]] = defaultdict(list)
    for item in items:
        groups[item.signature].append(item)
    for signature, members in sorted(groups.items()):
        if len(members) >= min_group:
            calibrator = CALIBRATORS[name]().fit(
                _padded_logits(members, probs, eps), _labels(members)
            )
            registry.register(CalibrationKey(backend, model, template_id, signature), calibrator)
    return registry
