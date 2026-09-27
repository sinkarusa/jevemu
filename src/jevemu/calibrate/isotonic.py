"""Top-label isotonic regression (Zadrozny & Elkan, 2002) on ``p_max``."""

from __future__ import annotations

from typing import Any

import numpy as np

from jevemu.calibrate._numeric import (
    as_labels,
    as_logp,
    check_name,
    finite_array,
    float_list,
    softmax,
    top_label,
    with_top_label,
)
from jevemu.calibrate.base import FloatArray, IntArray


def _pool_adjacent_violators(values: FloatArray, weights: FloatArray) -> FloatArray:
    """Weighted least-squares non-decreasing fit to ``values`` (already ordered by ``x``)."""
    block_value: list[float] = []
    block_weight: list[float] = []
    block_size: list[int] = []
    for v, w in zip(values.tolist(), weights.tolist(), strict=True):
        size = 1
        while block_value and block_value[-1] > v:
            pv, pw = block_value.pop(), block_weight.pop()
            v = (pv * pw + v * w) / (pw + w)
            w += pw
            size += block_size.pop()
        block_value.append(v)
        block_weight.append(w)
        block_size.append(size)
    return np.repeat(np.asarray(block_value, dtype=np.float64), block_size)


class IsotonicCalibrator:
    """Replace ``p_max`` by a non-decreasing fit of top-key correctness against ``p_max``.

    ``fit`` runs pool-adjacent-violators on (``p_max``, top key correct) pairs, ties in
    ``p_max`` pooled first. ``transform`` interpolates linearly between the fitted knots, is
    constant beyond the first and last knot, and rescales the other keys proportionally so
    rows sum to 1 (the top key stays a maximum, see ``with_top_label``). The map is
    non-decreasing in ``p_max``.
    """

    name = "isotonic"

    def __init__(self) -> None:
        self.n_classes: int | None = None
        self.x: FloatArray | None = None
        """Knot confidences, strictly increasing."""
        self.y: FloatArray | None = None
        """Calibrated confidence at each knot, non-decreasing."""

    def fit(self, logp: FloatArray, y: IntArray) -> IsotonicCalibrator:
        z = as_logp(logp)
        labels = as_labels(y, z)
        top, confidence = top_label(softmax(z))
        correct = (top == labels).astype(np.float64)
        xs, inverse, counts = np.unique(confidence, return_inverse=True, return_counts=True)
        weights = counts.astype(np.float64)
        fitted = _pool_adjacent_violators(np.bincount(inverse, weights=correct) / weights, weights)
        # Keep only the ends of each constant run: interpolation is unchanged, the JSON small.
        same_as_prev = np.r_[False, fitted[1:] == fitted[:-1]]
        same_as_next = np.r_[fitted[:-1] == fitted[1:], False]
        keep = ~(same_as_prev & same_as_next)
        self.n_classes, self.x, self.y = z.shape[1], xs[keep], fitted[keep]
        return self

    def transform(self, logp: FloatArray) -> FloatArray:
        n_classes, x, y = self._fitted()
        probs = softmax(as_logp(logp, n_classes=n_classes))
        top, confidence = top_label(probs)
        return with_top_label(probs, top, np.interp(confidence, x, y))

    def to_json(self) -> dict[str, Any]:
        n_classes, x, y = self._fitted()
        return {"name": self.name, "n_classes": n_classes, "x": float_list(x), "y": float_list(y)}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> IsotonicCalibrator:
        check_name(d, cls.name)
        x, y = finite_array(d["x"], "x"), finite_array(d["y"], "y")
        n_classes = int(d["n_classes"])
        if n_classes < 2 or x.shape != y.shape or x.shape[0] == 0:
            raise ValueError("isotonic needs n_classes >= 2 and equal-length, non-empty x and y")
        if (np.diff(x) <= 0).any() or (np.diff(y) < 0).any():
            raise ValueError("isotonic x must increase strictly and y must not decrease")
        calibrator = cls()
        calibrator.n_classes, calibrator.x, calibrator.y = n_classes, x, y
        return calibrator

    def _fitted(self) -> tuple[int, FloatArray, FloatArray]:
        if self.n_classes is None or self.x is None or self.y is None:
            raise RuntimeError("IsotonicCalibrator is not fitted")
        return self.n_classes, self.x, self.y
