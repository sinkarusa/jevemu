"""Evaluation metrics and paired significance tests (bootstrap and McNemar), as pure
numpy/scipy functions shared by every comparison report.

Conventions:

- ``probabilities`` is an ``(n, K)`` array, one row per item, columns in answer-key order; rows
  are used as given (callers pass renormalized distributions). ``gold`` is the ``(n,)`` column
  index of the correct key.
- Decomposable metrics (NLL, Brier, KL, total variation) return **per-item** arrays: the
  corpus value is their mean, and the paired bootstrap resamples them.
- Logarithms are natural (nats).

Definitions (``docs/implementation/design.md``, "Paired comparison harness"):

- Accuracy: fraction of items whose predicted key is the gold key (evaluate-idk's
  ``trad_score``; an abstention is not correct).
- NLL: ``-log max(p_gold, eps)``. The clip (default ``1e-6``, the smallest value of the design's
  ε sweep {1e-6, 1e-4, 1e-3}) keeps an exact zero, such as a Jev probability rounded to 0.00,
  finite.
- Brier: multiclass, ``sum_k (p_k - [k == gold])^2`` per item (range [0, 2]; not divided by K).
- ECE: top-label expected calibration error, ``sum_b (n_b / n) |acc_b - conf_b|`` with
  ``conf`` the probability of the predicted key. The design's default is 10 **equal-mass** bins
  (items sorted by confidence, split into 10 contiguous groups whose sizes differ by at most
  one, so tied confidences may straddle a boundary); ``binning="equal_width"`` uses
  ``[0, 0.1), …, [0.9, 1.0]`` instead.
- KL(p || q) and total variation ``0.5 * sum_k |p_k - q_k|`` between paired distributions. KL
  smooths both rows as ``(x + eps) / (1 + K * eps)`` (default ``eps = 1e-12``), so a zero
  probability (a ``-inf`` logprob) gives a large finite value instead of ``inf``: each
  log-ratio is capped near ``ln(1 / eps) ≈ 27.6`` nats, and ratios of probabilities well above
  ``eps`` are unchanged to ~1e-10.
- Paired bootstrap: 10,000 resamples of items (or of question clusters, so every permutation
  of a question stays together) with a fixed seed; 95% interval on the difference of means;
  BCa when there are at least 500 clusters, else percentile.
- McNemar: exact two-sided binomial test on the discordant pairs of per-item correctness.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from scipy.special import ndtr, ndtri

__all__ = [
    "BCA_MIN_CLUSTERS",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_ECE_BINS",
    "KL_EPS",
    "NLL_EPS",
    "BootstrapCI",
    "BootstrapMethod",
    "EceBinning",
    "McNemarResult",
    "accuracy",
    "brier",
    "ece",
    "kl_divergence",
    "mcnemar_exact",
    "nll",
    "paired_bootstrap_ci",
    "total_variation",
]

NLL_EPS = 1e-6
KL_EPS = 1e-12
DEFAULT_ECE_BINS = 10
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
BCA_MIN_CLUSTERS = 500
"""The design uses BCa intervals from this many clusters up, percentile intervals below."""

EceBinning = Literal["equal_mass", "equal_width"]
BootstrapMethod = Literal["percentile", "bca"]

FloatArray = NDArray[np.float64]


def _vector(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array


def _rows(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must be an (n, K) array, got shape {array.shape}")
    return array


def _same_length(*arrays: NDArray[np.generic]) -> int:
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"paired inputs must have equal length, got {sorted(lengths)}")
    return lengths.pop()


# --- per-item scores ---------------------------------------------------------------------------


def accuracy(predicted: ArrayLike, gold: ArrayLike) -> float:
    """Fraction of items with ``predicted == gold`` (keys or indices); ``nan`` when empty."""
    pred, true = np.asarray(predicted), np.asarray(gold)
    if pred.ndim != 1 or true.ndim != 1:
        raise ValueError("predicted and gold must be one-dimensional")
    if _same_length(pred, true) == 0:
        return math.nan
    return float(np.mean(pred == true))


def nll(p_gold: ArrayLike, *, eps: float = NLL_EPS) -> FloatArray:
    """Per-item negative log-likelihood of the gold key, ``-log max(p_gold, eps)``."""
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must lie in (0, 1), got {eps}")
    p = _vector(p_gold, "p_gold")
    return np.asarray(-np.log(np.clip(p, eps, 1.0)), dtype=np.float64)


def brier(probabilities: ArrayLike, gold: ArrayLike) -> FloatArray:
    """Per-item multiclass Brier score ``sum_k (p_k - [k == gold])^2``."""
    probs = _rows(probabilities, "probabilities")
    index = np.asarray(gold)
    if index.ndim != 1 or not np.issubdtype(index.dtype, np.integer):
        raise ValueError("gold must be a one-dimensional array of column indices")
    n, k = probs.shape
    _same_length(probs, index)
    if n and (index.min() < 0 or index.max() >= k):
        raise ValueError(f"gold indices must lie in [0, {k})")
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), index] = 1.0
    return np.asarray(np.sum((probs - onehot) ** 2, axis=1), dtype=np.float64)


# --- calibration -------------------------------------------------------------------------------


def ece(
    confidence: ArrayLike,
    correct: ArrayLike,
    *,
    n_bins: int = DEFAULT_ECE_BINS,
    binning: EceBinning = "equal_mass",
) -> float:
    """Top-label ECE: ``sum_b (n_b / n) |mean(correct_b) - mean(confidence_b)|``.

    ``confidence`` is the probability of each item's predicted key and ``correct`` whether that
    key is gold. Binning as in the module docstring; empty bins contribute nothing. ``nan``
    when there are no items.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    conf = _vector(confidence, "confidence")
    hit = _vector(correct, "correct")
    n = _same_length(conf, hit)
    if n == 0:
        return math.nan
    if binning == "equal_mass":
        order = np.argsort(conf, kind="stable")
        groups = [g for g in np.array_split(order, n_bins) if len(g)]
    elif binning == "equal_width":
        # Bin i holds [i / B, (i + 1) / B); the last bin also holds 1.0.
        ids = np.minimum((np.clip(conf, 0.0, 1.0) * n_bins).astype(np.int64), n_bins - 1)
        groups = [np.flatnonzero(ids == b) for b in range(n_bins)]
        groups = [g for g in groups if len(g)]
    else:
        raise ValueError(f"unknown binning {binning!r}")
    total = math.fsum(len(g) * abs(float(hit[g].mean() - conf[g].mean())) for g in groups)
    return total / n


# --- paired distributions ----------------------------------------------------------------------


def kl_divergence(p: ArrayLike, q: ArrayLike, *, eps: float = KL_EPS) -> FloatArray:
    """Per-row KL(p || q) in nats after additive ``eps`` smoothing of both rows."""
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must lie in (0, 1), got {eps}")
    left, right = _rows(p, "p"), _rows(q, "q")
    if left.shape != right.shape:
        raise ValueError(f"p and q must have the same shape, got {left.shape} and {right.shape}")
    k = left.shape[1]
    ps = (np.clip(left, 0.0, None) + eps) / (1.0 + k * eps)
    qs = (np.clip(right, 0.0, None) + eps) / (1.0 + k * eps)
    kl = np.sum(ps * (np.log(ps) - np.log(qs)), axis=1)
    # Rounding can leave a tiny negative value for identical rows.
    return np.asarray(np.maximum(kl, 0.0), dtype=np.float64)


def total_variation(p: ArrayLike, q: ArrayLike) -> FloatArray:
    """Per-row total variation distance ``0.5 * sum_k |p_k - q_k|``."""
    left, right = _rows(p, "p"), _rows(q, "q")
    if left.shape != right.shape:
        raise ValueError(f"p and q must have the same shape, got {left.shape} and {right.shape}")
    return np.asarray(0.5 * np.sum(np.abs(left - right), axis=1), dtype=np.float64)


# --- paired tests ------------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapCI:
    """``estimate`` = mean(a) - mean(b) on the full sample, with its bootstrap interval."""

    estimate: float
    low: float
    high: float
    confidence_level: float
    n_resamples: int
    method: BootstrapMethod
    n_clusters: int

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval is evidence of a difference (the design's reporting rule)."""
        return self.low > 0.0 or self.high < 0.0


def paired_bootstrap_ci(
    a: ArrayLike,
    b: ArrayLike,
    *,
    clusters: Sequence[Hashable] | None = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = 0.95,
    seed: int = 0,
    method: BootstrapMethod | None = None,
) -> BootstrapCI:
    """Bootstrap interval for ``mean(a) - mean(b)`` over paired per-item values.

    Items are resampled with replacement as pairs, or as whole ``clusters`` (one label per
    item, e.g. the question id, so all permutations of a question move together); the
    statistic is always the item-level mean difference of the resample. ``method`` defaults to
    BCa with at least :data:`BCA_MIN_CLUSTERS` clusters and percentile below. The same
    ``seed`` gives the same interval. A sample with no spread (every resample equal) returns
    the point estimate as both ends.
    """
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must lie in (0, 1), got {confidence_level}")
    left, right = _vector(a, "a"), _vector(b, "b")
    n = _same_length(left, right)
    if n == 0:
        raise ValueError("paired_bootstrap_ci needs at least one item")
    diff = left - right
    if clusters is None:
        sums, sizes = diff, np.ones(n)
    else:
        if len(clusters) != n:
            raise ValueError(f"clusters has {len(clusters)} labels for {n} items")
        codes: dict[Hashable, int] = {}
        ids = np.fromiter((codes.setdefault(c, len(codes)) for c in clusters), np.int64, n)
        sums = np.bincount(ids, weights=diff).astype(np.float64)
        sizes = np.bincount(ids).astype(np.float64)
    m = len(sums)
    chosen = method or ("bca" if m >= BCA_MIN_CLUSTERS else "percentile")
    total, count = float(sums.sum()), float(sizes.sum())
    estimate = total / count
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples)
    step = max(1, 2_000_000 // m)  # bounds the index matrix to ~16 MB
    for start in range(0, n_resamples, step):
        idx = rng.integers(0, m, size=(min(step, n_resamples - start), m))
        boot[start : start + len(idx)] = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    if np.ptp(boot) == 0.0:
        low = high = estimate
    elif chosen == "percentile":
        low, high = (float(x) for x in np.quantile(boot, [alpha, 1.0 - alpha]))
    elif chosen == "bca":
        low, high = _bca(boot, estimate, sums, sizes, alpha)
    else:
        raise ValueError(f"unknown bootstrap method {chosen!r}")
    return BootstrapCI(estimate, low, high, confidence_level, n_resamples, chosen, m)


def _bca(
    boot: FloatArray, estimate: float, sums: FloatArray, sizes: FloatArray, alpha: float
) -> tuple[float, float]:
    """Bias-corrected and accelerated percentiles (Efron 1987), jackknife over clusters.

    The bias-correction fraction is clipped to ``[0.5/B, 1 - 0.5/B]`` so a lopsided bootstrap
    distribution gives an extreme but finite percentile.
    """
    b = len(boot)
    fraction = float(np.mean(boot < estimate))
    z0 = float(ndtri(min(max(fraction, 0.5 / b), 1.0 - 0.5 / b)))
    jack = (sums.sum() - sums) / (sizes.sum() - sizes) if len(sums) > 1 else np.array([estimate])
    dev = jack.mean() - jack
    denom = 6.0 * float(np.sum(dev**2)) ** 1.5
    accel = float(np.sum(dev**3)) / denom if denom > 0.0 else 0.0
    levels = []
    for z in (float(ndtri(alpha)), float(ndtri(1.0 - alpha))):
        levels.append(float(ndtr(z0 + (z0 + z) / (1.0 - accel * (z0 + z)))))
    low, high = np.quantile(boot, levels)
    return float(low), float(high)


@dataclass(frozen=True)
class McNemarResult:
    """Discordant pairs and the exact two-sided p-value."""

    only_a: int
    """Items ``a`` got right and ``b`` got wrong."""
    only_b: int
    """Items ``b`` got right and ``a`` got wrong."""
    p_value: float


def mcnemar_exact(correct_a: ArrayLike, correct_b: ArrayLike) -> McNemarResult:
    """Exact McNemar test: ``min(1, 2 * P(X <= min(only_a, only_b)))`` for
    ``X ~ Binomial(only_a + only_b, 1/2)``; ``p_value = 1`` without discordant pairs."""
    left = np.asarray(correct_a, dtype=bool)
    right = np.asarray(correct_b, dtype=bool)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("correct_a and correct_b must be one-dimensional")
    _same_length(left, right)
    only_a = int(np.sum(left & ~right))
    only_b = int(np.sum(~left & right))
    discordant = only_a + only_b
    if discordant == 0:
        return McNemarResult(only_a, only_b, 1.0)
    p = float(stats.binomtest(min(only_a, only_b), discordant, 0.5).pvalue)
    return McNemarResult(only_a, only_b, min(1.0, p))
