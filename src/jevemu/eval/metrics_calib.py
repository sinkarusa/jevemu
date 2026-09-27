"""Calibration metrics of probability vectors and a paired cluster bootstrap.

Per item (probabilities in key order, gold key index): predicted key = first argmax;
``correct``; ``confidence`` = the predicted key's probability; NLL ``-log max(p_gold, 1e-6)``;
multiclass Brier ``sum_k (p_k - [k = gold])^2``. Per benchmark: accuracy, mean NLL, mean
Brier and top-label ECE with 10 equal-mass bins (:func:`jevemu.eval.metrics.ece`), plus the
reliability diagram's bins.

**Bootstrap.** :func:`bootstrap_replicates` resamples one benchmark's question clusters with
replacement (every item of a question id moves together) ``n_resamples`` times with a fixed
seed and evaluates every metric of every series on each resample. All series of a benchmark
(systems x arms, aligned on the same items) share the resamples, so the replicates of any
difference between them are paired. ECE on a resample is exactly :func:`~jevemu.eval.metrics.ece`
of the expanded sample (items weighted by how often their cluster was drawn; items with equal
confidence may be binned in another order). Macro replicates average the benchmarks'
replicates, which resampled independently: a stratified bootstrap. Intervals are percentile
intervals of the replicates.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jevemu.eval.metrics import DEFAULT_BOOTSTRAP_RESAMPLES, DEFAULT_ECE_BINS, NLL_EPS, ece

__all__ = [
    "METRICS",
    "Interval",
    "ItemMetrics",
    "ReliabilityBin",
    "bootstrap_replicates",
    "interval",
    "reliability",
    "weighted_ece",
]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

METRICS: tuple[str, ...] = ("accuracy", "nll", "brier", "ece")
_CELLS = 2_000_000
"""Bound on replicate-by-item cells per bootstrap chunk (memory ~16 MB per float array)."""


@dataclass(frozen=True)
class ItemMetrics:
    """Per-item correctness, confidence, NLL and Brier of one series on one benchmark."""

    correct: BoolArray
    confidence: FloatArray
    nll: FloatArray
    brier: FloatArray

    @classmethod
    def from_probs(
        cls, probs: Sequence[FloatArray], gold: Sequence[int], *, eps: float = NLL_EPS
    ) -> ItemMetrics:
        if len(probs) != len(gold):
            raise ValueError(f"{len(probs)} probability rows for {len(gold)} gold labels")
        n = len(probs)
        width = max((row.shape[0] for row in probs), default=1)
        padded = np.zeros((n, width), dtype=np.float64)
        for i, row in enumerate(probs):
            padded[i, : row.shape[0]] = row
        labels = np.asarray(gold, dtype=np.int64)
        rows = np.arange(n)
        predicted = padded.argmax(axis=1)
        p_gold = padded[rows, labels]
        onehot = np.zeros_like(padded)
        onehot[rows, labels] = 1.0
        return cls(
            correct=predicted == labels,
            confidence=padded[rows, predicted],
            nll=-np.log(np.clip(p_gold, eps, 1.0)),
            brier=((padded - onehot) ** 2).sum(axis=1),
        )

    def __len__(self) -> int:
        return int(self.correct.shape[0])

    def estimate(self, *, n_bins: int = DEFAULT_ECE_BINS) -> dict[str, float]:
        return {
            "accuracy": float(np.mean(self.correct)),
            "nll": float(np.mean(self.nll)),
            "brier": float(np.mean(self.brier)),
            "ece": ece(self.confidence, self.correct, n_bins=n_bins),
        }


@dataclass(frozen=True)
class ReliabilityBin:
    n: int
    confidence_low: float
    confidence_high: float
    confidence: float
    """Mean confidence in the bin."""
    accuracy: float


def reliability(
    confidence: FloatArray, correct: BoolArray, *, n_bins: int = DEFAULT_ECE_BINS
) -> list[ReliabilityBin]:
    """The equal-mass bins :func:`~jevemu.eval.metrics.ece` uses, in confidence order."""
    order = np.argsort(confidence, kind="stable")
    bins = []
    for group in np.array_split(order, n_bins):
        if len(group):
            conf = confidence[group]
            bins.append(
                ReliabilityBin(
                    len(group),
                    float(conf.min()),
                    float(conf.max()),
                    float(conf.mean()),
                    float(np.mean(correct[group])),
                )
            )
    return bins


def weighted_ece(
    confidence: FloatArray, correct: FloatArray, weights: FloatArray, *, n_bins: int
) -> FloatArray:
    """Equal-mass ECE of each row of integer ``weights`` (``(R, n)``): the ECE of the sample
    holding item ``i`` ``weights[r, i]`` times. ``confidence``/``correct`` must already be in
    ascending confidence order; bins split the expanded sample like ``np.array_split``."""
    total = weights.sum(axis=1)
    if (total <= 0).any():
        raise ValueError("every replicate needs a positive total weight")
    cum_w = np.concatenate([np.zeros((weights.shape[0], 1)), np.cumsum(weights, axis=1)], axis=1)
    cum_conf = np.concatenate(
        [np.zeros((weights.shape[0], 1)), np.cumsum(weights * confidence, axis=1)], axis=1
    )
    cum_hit = np.concatenate(
        [np.zeros((weights.shape[0], 1)), np.cumsum(weights * correct, axis=1)], axis=1
    )
    n_total = np.rint(total).astype(np.int64)
    base, extra = np.divmod(n_total, n_bins)
    b = np.arange(n_bins + 1)
    edges = (base[:, None] * b[None, :] + np.minimum(b[None, :], extra[:, None])).astype(
        np.float64
    )  # (R, B + 1) expanded positions
    # Items whose cumulative end is <= the edge lie wholly before it; item k straddles it.
    k = (cum_w[:, None, 1:] <= edges[:, :, None]).sum(axis=2)  # (R, B + 1)
    rows = np.arange(weights.shape[0])[:, None]
    conf_ext = np.append(confidence, 0.0)
    hit_ext = np.append(correct, 0.0)
    offset = edges - cum_w[rows, k]
    s_conf = cum_conf[rows, k] + offset * conf_ext[k]
    s_hit = cum_hit[rows, k] + offset * hit_ext[k]
    gaps = np.abs(np.diff(s_hit, axis=1) - np.diff(s_conf, axis=1))
    out: FloatArray = gaps.sum(axis=1) / total
    return out


def bootstrap_replicates(
    clusters: Sequence[Hashable],
    series: Mapping[str, ItemMetrics],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 0,
    n_bins: int = DEFAULT_ECE_BINS,
) -> dict[str, dict[str, FloatArray]]:
    """Series -> metric -> ``(n_resamples,)`` replicates on shared cluster resamples."""
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    n = len(clusters)
    if n == 0:
        raise ValueError("bootstrap_replicates needs at least one item")
    for name, metrics in series.items():
        if len(metrics) != n:
            raise ValueError(f"series {name!r} has {len(metrics)} items, clusters {n}")
    codes: dict[Hashable, int] = {}
    cluster_of = np.fromiter((codes.setdefault(c, len(codes)) for c in clusters), np.int64, n)
    m = len(codes)
    orders = {name: np.argsort(s.confidence, kind="stable") for name, s in series.items()}
    out = {name: {metric: np.empty(n_resamples) for metric in METRICS} for name in series}
    rng = np.random.default_rng(seed)
    step = max(1, _CELLS // max(n, m))
    uniform = np.full(m, 1.0 / m)
    for start in range(0, n_resamples, step):
        size = min(step, n_resamples - start)
        counts = rng.multinomial(m, uniform, size=size).astype(np.float64)
        weights = counts[:, cluster_of]  # (size, n)
        total = weights.sum(axis=1)
        chunk = slice(start, start + size)
        for name, s in series.items():
            order = orders[name]
            target = out[name]
            target["accuracy"][chunk] = weights @ s.correct.astype(np.float64) / total
            target["nll"][chunk] = weights @ s.nll / total
            target["brier"][chunk] = weights @ s.brier / total
            target["ece"][chunk] = weighted_ece(
                s.confidence[order],
                s.correct[order].astype(np.float64),
                weights[:, order],
                n_bins=n_bins,
            )
    return out


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def to_json(self) -> list[float]:
        return [self.estimate, self.low, self.high]


def interval(estimate: float, replicates: FloatArray, *, level: float = 0.95) -> Interval:
    """Percentile interval of ``replicates`` around the full-sample ``estimate``."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    alpha = (1.0 - level) / 2.0
    if replicates.shape[0] == 0 or not np.isfinite(replicates).all():
        return Interval(estimate, math.nan, math.nan)
    low, high = np.quantile(replicates, [alpha, 1.0 - alpha])
    return Interval(estimate, float(low), float(high))
