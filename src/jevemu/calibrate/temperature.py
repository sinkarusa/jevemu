"""Temperature scaling (Guo et al., 2017): one ``T`` for all keys, fitted by NLL."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize

from jevemu.calibrate._numeric import (
    as_labels,
    as_logp,
    centered,
    check_name,
    finite_targets,
    softmax,
)
from jevemu.calibrate.base import FloatArray, IntArray

T_MIN = 1e-3
T_MAX = 1e3
"""Search bounds for ``T``: separable data drives the NLL optimum to ``T -> 0``."""


class TemperatureCalibrator:
    """``softmax(logp / T)``. ``T > 1`` softens overconfident rows; ``T = 1`` is the identity.

    ``fit`` minimizes the mean NLL of the correct keys over ``log T`` with L-BFGS-B, so
    ``T > 0`` by construction; the objective is unimodal in ``log T``. Zero-probability keys
    (``-inf``) stay at zero. Temperature does not depend on ``K``, so one fitted ``T`` applies
    to any question (the registry's per-(model, template) fallback).
    """

    name = "temperature"

    def __init__(self, temperature: float = 1.0) -> None:
        if not (math.isfinite(temperature) and temperature > 0.0):
            raise ValueError(f"temperature must be finite and > 0, got {temperature}")
        self.temperature = float(temperature)

    def fit(self, logp: FloatArray, y: IntArray) -> TemperatureCalibrator:
        z = centered(as_logp(logp))
        zy = finite_targets(z, as_labels(y, z))
        z_finite = np.where(np.isfinite(z), z, 0.0)  # p * z with p = 0 where z = -inf

        def nll_and_grad(theta: FloatArray) -> tuple[float, FloatArray]:
            t = math.exp(float(theta[0]))
            e = np.exp(z / t)  # row max is exp(0) = 1, so no overflow for any t
            total = e.sum(axis=1)
            p = e / total[:, None]
            nll = float(np.mean(np.log(total) - zy / t))
            d_log_t = float(np.mean(zy - (p * z_finite).sum(axis=1))) / t
            return nll, np.array([d_log_t])

        result = minimize(
            nll_and_grad,
            np.zeros(1),
            jac=True,
            method="L-BFGS-B",
            bounds=[(math.log(T_MIN), math.log(T_MAX))],
            options={"gtol": 1e-10},
        )
        self.temperature = math.exp(float(result.x[0]))
        return self

    def transform(self, logp: FloatArray) -> FloatArray:
        return softmax(as_logp(logp) / self.temperature)

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "temperature": self.temperature}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> TemperatureCalibrator:
        check_name(d, cls.name)
        return cls(float(d["temperature"]))
