"""No-op calibrator: renormalizes each row and changes nothing else."""

from __future__ import annotations

from typing import Any

from jevemu.calibrate._numeric import as_labels, as_logp, check_name, softmax
from jevemu.calibrate.base import FloatArray, IntArray


class IdentityCalibrator:
    """``softmax(logp)``: exact for normalized rows; the registry's last-resort fallback."""

    name = "identity"

    def fit(self, logp: FloatArray, y: IntArray) -> IdentityCalibrator:
        as_labels(y, as_logp(logp))
        return self

    def transform(self, logp: FloatArray) -> FloatArray:
        return softmax(as_logp(logp))

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> IdentityCalibrator:
        check_name(d, cls.name)
        return cls()
