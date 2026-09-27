"""Post-hoc calibrators, the calibrator registry, and the probability floor for logits."""

from jevemu.calibrate._numeric import DEFAULT_EPS, probs_to_logits
from jevemu.calibrate.base import Calibrator, FloatArray, IntArray
from jevemu.calibrate.histogram import HistogramCalibrator
from jevemu.calibrate.identity import IdentityCalibrator
from jevemu.calibrate.isotonic import IsotonicCalibrator
from jevemu.calibrate.platt import PlattCalibrator
from jevemu.calibrate.registry import (
    CALIBRATORS,
    CalibrationKey,
    CalibrationSource,
    CalibratorRegistry,
    ResolvedCalibrator,
    calibrator_from_json,
    question_signature,
)
from jevemu.calibrate.temperature import TemperatureCalibrator
from jevemu.calibrate.vector import VectorCalibrator

__all__ = [
    "CALIBRATORS",
    "DEFAULT_EPS",
    "CalibrationKey",
    "CalibrationSource",
    "Calibrator",
    "CalibratorRegistry",
    "FloatArray",
    "HistogramCalibrator",
    "IdentityCalibrator",
    "IntArray",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "ResolvedCalibrator",
    "TemperatureCalibrator",
    "VectorCalibrator",
    "calibrator_from_json",
    "probs_to_logits",
    "question_signature",
]
