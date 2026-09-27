"""Temperature plus a per-key bias, for questions with a fixed key set."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize

from jevemu.calibrate._numeric import (
    as_labels,
    as_logp,
    centered,
    check_name,
    finite_array,
    finite_targets,
    float_list,
    softmax,
)
from jevemu.calibrate.base import FloatArray, IntArray
from jevemu.calibrate.temperature import T_MAX, T_MIN

DEFAULT_L2 = 1e-3


class VectorCalibrator:
    """``softmax(logp / T + b)`` with one bias ``b_k`` per key position.

    The bias absorbs a label prior (e.g. a model that over-picks ``A``), so it only means
    something for one key set: fit per question signature. ``fit`` minimizes mean NLL plus
    ``l2 / 2 * |b|^2`` over ``(log T, b)`` with L-BFGS-B. The penalty keeps ``b`` finite for keys
    that are never correct and fixes softmax's shift freedom; the stored ``b`` is centered to
    mean 0. Zero-probability keys (``-inf``) stay at zero.
    """

    name = "vector"

    def __init__(self, *, l2: float = DEFAULT_L2) -> None:
        if not (math.isfinite(l2) and l2 >= 0.0):
            raise ValueError(f"l2 must be finite and >= 0, got {l2}")
        self.l2 = float(l2)
        self.temperature = 1.0
        self.bias: FloatArray | None = None

    def fit(self, logp: FloatArray, y: IntArray) -> VectorCalibrator:
        z = centered(as_logp(logp))
        labels = as_labels(y, z)
        zy = finite_targets(z, labels)
        n, k = z.shape
        z_finite = np.where(np.isfinite(z), z, 0.0)
        onehot = np.zeros((n, k))
        onehot[np.arange(n), labels] = 1.0

        def objective(theta: FloatArray) -> tuple[float, FloatArray]:
            t = math.exp(float(theta[0]))
            b = theta[1:]
            s = z / t + b
            m = s.max(axis=1)
            e = np.exp(s - m[:, None])
            total = e.sum(axis=1)
            p = e / total[:, None]
            nll = float(np.mean(m + np.log(total) - (zy / t + b[labels])))
            nll += 0.5 * self.l2 * float(b @ b)
            d_log_t = float(np.mean(zy - (p * z_finite).sum(axis=1))) / t
            d_b = (p - onehot).mean(axis=0) + self.l2 * b
            return nll, np.concatenate([[d_log_t], d_b])

        result = minimize(
            objective,
            np.zeros(k + 1),
            jac=True,
            method="L-BFGS-B",
            bounds=[(math.log(T_MIN), math.log(T_MAX))] + [(None, None)] * k,
            options={"gtol": 1e-10},
        )
        bias = np.asarray(result.x[1:], dtype=np.float64)
        self.temperature = math.exp(float(result.x[0]))
        self.bias = bias - bias.mean()
        return self

    def transform(self, logp: FloatArray) -> FloatArray:
        bias = self._fitted_bias()
        return softmax(as_logp(logp, n_classes=bias.shape[0]) / self.temperature + bias)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "l2": self.l2,
            "temperature": self.temperature,
            "bias": float_list(self._fitted_bias()),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> VectorCalibrator:
        check_name(d, cls.name)
        calibrator = cls(l2=float(d["l2"]))
        calibrator._restore(float(d["temperature"]), d["bias"])
        return calibrator

    def _restore(self, temperature: float, bias: Sequence[float]) -> None:
        if not (math.isfinite(temperature) and temperature > 0.0):
            raise ValueError(f"temperature must be finite and > 0, got {temperature}")
        arr = finite_array(bias, "bias")
        if arr.shape[0] < 2:
            raise ValueError("bias needs one entry per key (K >= 2)")
        self.temperature = temperature
        self.bias = arr

    def _fitted_bias(self) -> FloatArray:
        if self.bias is None:
            raise RuntimeError("VectorCalibrator is not fitted")
        return self.bias
