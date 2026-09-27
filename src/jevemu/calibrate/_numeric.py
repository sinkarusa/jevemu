"""Numerics shared by the calibrators: input checks, stable softmax, top-label redistribution.

Calibrators treat ``logp`` rows as logits: rows need not be normalized (``probs_to_logits``
output is not), ``-inf`` means zero probability, and every row needs one finite entry.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from jevemu.calibrate.base import FloatArray, IntArray

DEFAULT_EPS = 1e-4
"""Probability floor of :func:`probs_to_logits`."""

LOGIT_CLIP = 30.0
"""Bound on log-odds features, so zero probabilities stay finite (``sigmoid(30) ~ 1 - 1e-13``)."""

_P_TOL = 1e-9


def probs_to_logits(p: ArrayLike, eps: float = DEFAULT_EPS) -> FloatArray:
    """``log(clip(p, eps, 1))``: finite logits from probabilities that may hold exact zeros.

    Jev reports probabilities to two decimals and returns exact zeros, so its answers need a
    floor before any log-space calibration; the emulator uses the same ``eps`` so both systems
    are calibrated on the same scale. Any shape is accepted and preserved.
    """
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must lie in (0, 1), got {eps}")
    arr = np.asarray(p, dtype=np.float64)
    if np.isnan(arr).any() or (arr < 0.0).any() or (arr > 1.0 + _P_TOL).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return np.log(np.clip(arr, eps, 1.0))


def as_logp(logp: ArrayLike, *, n_classes: int | None = None) -> FloatArray:
    """Validate an ``(N, K)`` logit array (``K >= 2``, no NaN/+inf, a finite entry per row)."""
    z = np.asarray(logp, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] < 2:
        raise ValueError(f"logp must have shape (N, K) with K >= 2, got {z.shape}")
    if n_classes is not None and z.shape[1] != n_classes:
        raise ValueError(f"calibrator was fitted for K={n_classes} keys, got K={z.shape[1]}")
    if np.isnan(z).any() or (z == np.inf).any():
        raise ValueError("logp must not contain NaN or +inf")
    if not np.isfinite(z).any(axis=1).all():
        raise ValueError("every logp row needs at least one finite entry")
    return z


def as_labels(y: ArrayLike, logp: FloatArray) -> IntArray:
    """Validate correct-key indices ``y`` (shape ``(N,)``, integers in ``[0, K)``, ``N >= 1``)."""
    arr = np.asarray(y)
    n, k = logp.shape
    if arr.shape != (n,):
        raise ValueError(f"y must have shape ({n},), got {arr.shape}")
    if n == 0:
        raise ValueError("cannot fit a calibrator on zero rows")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"y must hold integer key indices, got dtype {arr.dtype}")
    if (arr < 0).any() or (arr >= k).any():
        raise ValueError(f"y must lie in [0, {k})")
    return arr.astype(np.int64)


def centered(z: FloatArray) -> FloatArray:
    """Shift each row so its maximum is 0 (softmax-invariant; keeps ``exp`` in ``[0, 1]``)."""
    shifted: FloatArray = z - z.max(axis=1, keepdims=True)
    return shifted


def finite_targets(z: FloatArray, y: IntArray) -> FloatArray:
    """``z[i, y[i]]``; raises if a correct key has zero probability (NLL is then infinite)."""
    zy: FloatArray = z[np.arange(z.shape[0]), y]
    bad = np.flatnonzero(~np.isfinite(zy))
    if bad.size:
        raise ValueError(
            f"row {int(bad[0])}: the correct key has -inf logp, so NLL is infinite for every "
            "parameter value; floor probabilities with probs_to_logits first"
        )
    return zy


def log_softmax(z: FloatArray) -> FloatArray:
    shifted = centered(z)
    out: FloatArray = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return out


def softmax(z: FloatArray) -> FloatArray:
    return np.exp(log_softmax(z))


def top_label(probs: FloatArray) -> tuple[IntArray, FloatArray]:
    """Index (first on ties) and probability of each row's most likely key."""
    top = probs.argmax(axis=1).astype(np.int64)
    return top, probs[np.arange(probs.shape[0]), top]


def top_label_logit(probs: FloatArray, top: IntArray) -> FloatArray:
    """``log(p_top / (1 - p_top))``, with ``1 - p_top`` summed over the other keys, clipped."""
    rows = np.arange(probs.shape[0])
    others = probs.copy()
    others[rows, top] = 0.0
    with np.errstate(divide="ignore"):
        logit: FloatArray = np.log(probs[rows, top]) - np.log(others.sum(axis=1))
    return np.clip(logit, -LOGIT_CLIP, LOGIT_CLIP)


def with_top_label(probs: FloatArray, top: IntArray, confidence: FloatArray) -> FloatArray:
    """Set each row's top-key probability to ``confidence``; rescale the others to fit.

    The other keys keep their proportions and share ``1 - confidence``; if they all had zero
    probability they share it uniformly. The top key is never demoted: ``confidence`` is floored
    at the smallest value that keeps it a (possibly tied) maximum, which is
    ``second / (second + rest)`` (``1 / K`` when ``rest == 0``). Rows sum to 1.
    """
    n, k = probs.shape
    rows = np.arange(n)
    others = probs.copy()
    others[rows, top] = 0.0
    rest = others.sum(axis=1)
    second = others.max(axis=1)
    has_rest = rest > 0.0
    floor = np.divide(second, second + rest, out=np.full(n, 1.0 / k), where=has_rest)
    conf = np.clip(np.maximum(confidence, floor), 0.0, 1.0)
    share = np.divide(
        others, rest[:, None], out=np.full((n, k), 1.0 / (k - 1)), where=has_rest[:, None]
    )
    out: FloatArray = share * (1.0 - conf)[:, None]
    out[rows, top] = conf
    return out


def check_name(d: dict[str, Any], name: str) -> None:
    if d.get("name") != name:
        raise ValueError(f"expected a {name!r} calibrator, got name={d.get('name')!r}")


def float_list(a: FloatArray) -> list[float]:
    return [float(v) for v in a]


def finite_array(values: Any, what: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise ValueError(f"{what} must be a flat list of finite numbers")
    return arr
