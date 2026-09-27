"""Platt scaling (Platt, 1999): logistic regression on one log-odds feature."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize

from jevemu.calibrate._numeric import (
    LOGIT_CLIP,
    as_labels,
    as_logp,
    check_name,
    softmax,
    top_label,
    top_label_logit,
    with_top_label,
)
from jevemu.calibrate.base import FloatArray, IntArray


def _sigmoid(u: FloatArray) -> FloatArray:
    e = np.exp(-np.abs(u))
    return np.where(u >= 0.0, 1.0 / (1.0 + e), e / (1.0 + e))


class PlattCalibrator:
    """``sigmoid(a * s + b)`` for a log-odds feature ``s``; the mode follows ``K``.

    - **Binary** (``K == 2``: Noul, two-option Choice): ``s = logp[:, 0] - logp[:, 1]`` and the
      output is ``[q, 1 - q]`` with ``q`` the calibrated probability of the *first* key, trained
      on ``y == 0``. For Noul (keys ``("true", "false")``) ``q`` is the calibrated P(yes). Unlike
      top-label calibration this can correct a yes/no bias, so it may move either key above 0.5.
    - **Top-label** (``K > 2``): ``s = logit(p_max)`` and ``q`` estimates P(the top key is
      correct); ``q`` replaces ``p_max`` and the other keys are rescaled proportionally (the top
      key stays a maximum, see ``with_top_label``).

    ``s`` is clipped to +-30 so zero probabilities stay finite. ``fit`` uses Platt's smoothed
    targets ``(N+ + 1) / (N+ + 2)`` and ``1 / (N- + 2)``, which keep ``a`` and ``b`` finite on
    separable data, and minimizes their cross-entropy with L-BFGS-B from ``a = 1, b = 0``.
    """

    name = "platt"

    def __init__(self) -> None:
        self.n_classes: int | None = None
        self.a = 1.0
        self.b = 0.0

    def fit(self, logp: FloatArray, y: IntArray) -> PlattCalibrator:
        z = as_logp(logp)
        labels = as_labels(y, z)
        k = z.shape[1]
        if k == 2:
            s = self._binary_feature(z)
            positive = labels == 0
        else:
            probs = softmax(z)
            top, _ = top_label(probs)
            s = top_label_logit(probs, top)
            positive = top == labels
        n_pos = int(positive.sum())
        n_neg = positive.shape[0] - n_pos
        target = np.where(positive, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))

        def objective(ab: FloatArray) -> tuple[float, FloatArray]:
            u = ab[0] * s + ab[1]
            loss = float(np.mean(np.logaddexp(0.0, u) - target * u))
            g = _sigmoid(u) - target
            return loss, np.array([float(np.mean(g * s)), float(np.mean(g))])

        result = minimize(
            objective, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B", options={"gtol": 1e-10}
        )
        self.a, self.b = float(result.x[0]), float(result.x[1])
        self.n_classes = k
        return self

    def transform(self, logp: FloatArray) -> FloatArray:
        if self.n_classes is None:
            raise RuntimeError("PlattCalibrator is not fitted")
        z = as_logp(logp, n_classes=self.n_classes)
        if self.n_classes == 2:
            q = _sigmoid(self.a * self._binary_feature(z) + self.b)
            return np.stack([q, 1.0 - q], axis=1)
        probs = softmax(z)
        top, _ = top_label(probs)
        q = _sigmoid(self.a * top_label_logit(probs, top) + self.b)
        return with_top_label(probs, top, q)

    def to_json(self) -> dict[str, Any]:
        if self.n_classes is None:
            raise RuntimeError("PlattCalibrator is not fitted")
        return {"name": self.name, "n_classes": self.n_classes, "a": self.a, "b": self.b}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> PlattCalibrator:
        check_name(d, cls.name)
        n_classes, a, b = int(d["n_classes"]), float(d["a"]), float(d["b"])
        if n_classes < 2 or not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError(f"invalid platt parameters: n_classes={n_classes}, a={a}, b={b}")
        calibrator = cls()
        calibrator.n_classes, calibrator.a, calibrator.b = n_classes, a, b
        return calibrator

    @staticmethod
    def _binary_feature(z: FloatArray) -> FloatArray:
        return np.clip(z[:, 0] - z[:, 1], -LOGIT_CLIP, LOGIT_CLIP)
