"""Calibrator contract: post-hoc maps from log-probabilities to calibrated probabilities.

Fitted only on held-out data. ``logp`` rows are log-probabilities over one question's keys
(shape ``(N, K)``); ``y`` holds the index of the correct key per row.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class Calibrator(Protocol):
    name: str

    def fit(self, logp: FloatArray, y: IntArray) -> Calibrator: ...

    def transform(self, logp: FloatArray) -> FloatArray:
        """Calibrated probabilities, shape ``(N, K)``, rows summing to 1."""
        ...

    def to_json(self) -> dict[str, Any]: ...

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Calibrator: ...
