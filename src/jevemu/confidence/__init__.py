"""Pluggable confidence functions for Choice and Score answers (fitting: ``.fit``)."""

from jevemu.confidence.functions import (
    CONFIDENCE_FUNCTIONS,
    DEFAULT_CONFIDENCE,
    ConfidenceFn,
    margin,
    max_prob,
    mode_distance,
    one_minus_norm_entropy,
    peak_linear,
    uniform_spread,
)

__all__ = [
    "CONFIDENCE_FUNCTIONS",
    "DEFAULT_CONFIDENCE",
    "ConfidenceFn",
    "margin",
    "max_prob",
    "mode_distance",
    "one_minus_norm_entropy",
    "peak_linear",
    "uniform_spread",
]
