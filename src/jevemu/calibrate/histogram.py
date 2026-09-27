"""Top-label histogram binning (Zadrozny & Elkan, 2001) with equal-mass bins on ``p_max``."""

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

DEFAULT_BINS = 10


class HistogramCalibrator:
    """Replace ``p_max`` by the top-key accuracy of its bin; bins hold equal numbers of rows.

    Bin ``j`` is ``(edges[j-1], edges[j]]`` (open-ended at both extremes). Edges are the
    ``j / n_bins`` quantiles of the fitted ``p_max`` values (each a fitted value), so a run of
    tied confidences never straddles two bins; ties merge bins instead, and every bin is
    non-empty. The other keys are rescaled proportionally so rows sum to 1 (the top key stays
    a maximum, see ``with_top_label``). The map is piecewise constant and need not be monotone.
    """

    name = "histogram"

    def __init__(self, n_bins: int = DEFAULT_BINS) -> None:
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        self.n_bins = n_bins
        self.n_classes: int | None = None
        self.edges: FloatArray | None = None
        """Upper edges of all bins but the last, strictly increasing."""
        self.values: FloatArray | None = None
        """Calibrated confidence per bin (``len(edges) + 1`` entries)."""

    def fit(self, logp: FloatArray, y: IntArray) -> HistogramCalibrator:
        z = as_logp(logp)
        labels = as_labels(y, z)
        top, confidence = top_label(softmax(z))
        correct = (top == labels).astype(np.float64)
        quantiles = np.arange(1, self.n_bins) / self.n_bins
        edges = np.unique(np.quantile(confidence, quantiles, method="inverted_cdf"))
        edges = edges[edges < confidence.max()]
        bins = np.searchsorted(edges, confidence, side="left")
        counts = np.bincount(bins, minlength=edges.shape[0] + 1)
        values = np.bincount(bins, weights=correct, minlength=edges.shape[0] + 1) / counts
        self.n_classes, self.edges, self.values = z.shape[1], edges, values
        return self

    def transform(self, logp: FloatArray) -> FloatArray:
        n_classes, edges, values = self._fitted()
        probs = softmax(as_logp(logp, n_classes=n_classes))
        top, confidence = top_label(probs)
        bins = np.searchsorted(edges, confidence, side="left")
        return with_top_label(probs, top, values[bins])

    def to_json(self) -> dict[str, Any]:
        n_classes, edges, values = self._fitted()
        return {
            "name": self.name,
            "n_bins": self.n_bins,
            "n_classes": n_classes,
            "edges": float_list(edges),
            "values": float_list(values),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> HistogramCalibrator:
        check_name(d, cls.name)
        edges, values = finite_array(d["edges"], "edges"), finite_array(d["values"], "values")
        n_classes = int(d["n_classes"])
        if n_classes < 2 or values.shape[0] != edges.shape[0] + 1:
            raise ValueError("histogram needs n_classes >= 2 and len(values) == len(edges) + 1")
        if (np.diff(edges) <= 0).any():
            raise ValueError("histogram edges must increase strictly")
        calibrator = cls(int(d["n_bins"]))
        calibrator.n_classes, calibrator.edges, calibrator.values = n_classes, edges, values
        return calibrator

    def _fitted(self) -> tuple[int, FloatArray, FloatArray]:
        if self.n_classes is None or self.edges is None or self.values is None:
            raise RuntimeError("HistogramCalibrator is not fitted")
        return self.n_classes, self.edges, self.values
