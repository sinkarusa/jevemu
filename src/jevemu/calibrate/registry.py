"""Fitted calibrators keyed by (backend, model, template_id, question signature).

Lookup order: the exact key, then a per-(model, template_id) temperature, then identity.
Registries save to and load from one JSON file.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jevemu.calibrate.base import Calibrator
from jevemu.calibrate.histogram import HistogramCalibrator
from jevemu.calibrate.identity import IdentityCalibrator
from jevemu.calibrate.isotonic import IsotonicCalibrator
from jevemu.calibrate.platt import PlattCalibrator
from jevemu.calibrate.temperature import TemperatureCalibrator
from jevemu.calibrate.vector import VectorCalibrator
from jevemu.types import ChoiceQuestion, Question, ScoreQuestion

CALIBRATORS: dict[str, type[Calibrator]] = {
    cls.name: cls
    for cls in (
        IdentityCalibrator,
        TemperatureCalibrator,
        VectorCalibrator,
        PlattCalibrator,
        IsotonicCalibrator,
        HistogramCalibrator,
    )
}
"""Calibrator classes by ``name``."""

REGISTRY_FORMAT = "jevemu.calibrator_registry"
REGISTRY_VERSION = 1

CalibrationSource = Literal["exact", "temperature_fallback", "identity"]


def calibrator_from_json(d: dict[str, Any]) -> Calibrator:
    """Rebuild any calibrator from its ``to_json`` output, dispatching on ``d["name"]``."""
    name = d.get("name")
    if not isinstance(name, str) or name not in CALIBRATORS:
        raise ValueError(f"unknown calibrator {name!r}; known: {sorted(CALIBRATORS)}")
    return CALIBRATORS[name].from_json(d)


def question_signature(question: Question) -> str:
    """Identify a question's answer space: ``"<type>:<K>:<16 hex chars>"``.

    The hex part is the start of the SHA-256 of the canonical JSON (sorted object keys, compact
    separators, UTF-8) of what defines the answer positions:

    - ``choice``: the option keys in request order. Option descriptions are per-item text and
      are excluded, so e.g. every ``A``-``E`` bank question shares one signature.
    - ``score``: the level descriptions in order (the keys are always ``"0"``..``"K-1"``).
    - ``noul``: the fixed keys ``["true", "false"]``, so all Noul questions share one.

    Instructions and state never enter. Reordering options changes the signature, because a
    calibrator's parameters are tied to key positions.
    """
    answer_space: list[Any]
    if isinstance(question, (ChoiceQuestion, ScoreQuestion)):
        answer_space = list(question.criteria)  # choice: option keys; score: level descriptions
    else:
        answer_space = ["true", "false"]
    canonical = json.dumps(answer_space, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{question.type}:{len(answer_space)}:{digest}"


@dataclass(frozen=True, order=True)
class CalibrationKey:
    backend: str
    """``BackendInfo.backend`` (e.g. ``"vllm_http"``), or ``"jev"`` for the real API."""
    model: str
    template_id: str
    """``RenderedPrompt.template_id``: layout plus template hash."""
    signature: str
    """:func:`question_signature` of the question."""

    @classmethod
    def for_question(
        cls, *, backend: str, model: str, template_id: str, question: Question
    ) -> CalibrationKey:
        return cls(backend, model, template_id, question_signature(question))


@dataclass(frozen=True)
class ResolvedCalibrator:
    calibrator: Calibrator
    source: CalibrationSource
    """Which lookup level matched: the exact key, the model/template temperature, or none."""


class CalibratorRegistry:
    """Fitted calibrators with a per-(model, template_id) temperature fallback.

    Registering under an existing key replaces the earlier calibrator.
    """

    def __init__(self) -> None:
        self._exact: dict[CalibrationKey, Calibrator] = {}
        self._fallback: dict[tuple[str, str], TemperatureCalibrator] = {}

    def register(self, key: CalibrationKey, calibrator: Calibrator) -> None:
        self._exact[key] = calibrator

    def register_fallback(
        self, model: str, template_id: str, calibrator: TemperatureCalibrator
    ) -> None:
        """Temperature used for any question of ``model`` + ``template_id`` without an exact
        entry, on any backend."""
        if not isinstance(calibrator, TemperatureCalibrator):
            raise TypeError(f"fallback must be a TemperatureCalibrator, got {type(calibrator)}")
        self._fallback[(model, template_id)] = calibrator

    def resolve(self, key: CalibrationKey) -> ResolvedCalibrator:
        exact = self._exact.get(key)
        if exact is not None:
            return ResolvedCalibrator(exact, "exact")
        fallback = self._fallback.get((key.model, key.template_id))
        if fallback is not None:
            return ResolvedCalibrator(fallback, "temperature_fallback")
        return ResolvedCalibrator(IdentityCalibrator(), "identity")

    def to_json(self) -> dict[str, Any]:
        return {
            "format": REGISTRY_FORMAT,
            "version": REGISTRY_VERSION,
            "calibrators": [
                {
                    "backend": key.backend,
                    "model": key.model,
                    "template_id": key.template_id,
                    "signature": key.signature,
                    "calibrator": self._exact[key].to_json(),
                }
                for key in sorted(self._exact)
            ],
            "fallbacks": [
                {"model": model, "template_id": template_id, "calibrator": cal.to_json()}
                for (model, template_id), cal in sorted(self._fallback.items())
            ],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> CalibratorRegistry:
        if d.get("format") != REGISTRY_FORMAT or d.get("version") != REGISTRY_VERSION:
            raise ValueError(
                f"not a {REGISTRY_FORMAT} v{REGISTRY_VERSION} document: "
                f"format={d.get('format')!r}, version={d.get('version')!r}"
            )
        registry = cls()
        for entry in d["calibrators"]:
            key = CalibrationKey(
                entry["backend"], entry["model"], entry["template_id"], entry["signature"]
            )
            registry.register(key, calibrator_from_json(entry["calibrator"]))
        for entry in d["fallbacks"]:
            registry.register_fallback(
                entry["model"],
                entry["template_id"],
                TemperatureCalibrator.from_json(entry["calibrator"]),
            )
        return registry

    def save(self, path: str | os.PathLike[str]) -> None:
        text = json.dumps(self.to_json(), indent=2, allow_nan=False)
        Path(path).write_text(text + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> CalibratorRegistry:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
