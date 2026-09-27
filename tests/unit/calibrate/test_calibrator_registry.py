from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jevemu.calibrate import (
    CALIBRATORS,
    CalibrationKey,
    Calibrator,
    CalibratorRegistry,
    IdentityCalibrator,
    IsotonicCalibrator,
    TemperatureCalibrator,
    VectorCalibrator,
    calibrator_from_json,
    question_signature,
)
from jevemu.types import ChoiceQuestion, NoulQuestion, ScoreQuestion

GPQA = ChoiceQuestion(
    instructions="Which answer is correct?",
    criteria={"A": "1 eV", "B": "2 eV", "C": "3 eV", "D": "4 eV", "E": "I don't know"},
)
KEY = CalibrationKey.for_question(
    backend="vllm_http", model="Qwen/Qwen3-8B", template_id="state_first-abc", question=GPQA
)
LOGP = np.log(np.array([[0.6, 0.2, 0.1, 0.05, 0.05], [0.3, 0.3, 0.2, 0.1, 0.1]]))


def fitted_vector() -> VectorCalibrator:
    rng = np.random.default_rng(0)
    logp = np.log(rng.dirichlet(np.ones(5), size=200))
    return VectorCalibrator().fit(logp, rng.integers(0, 5, size=200))


def fitted_isotonic() -> IsotonicCalibrator:
    rng = np.random.default_rng(1)
    logp = np.log(rng.dirichlet(np.ones(5), size=200))
    return IsotonicCalibrator().fit(logp, rng.integers(0, 5, size=200))


# --- lookup order ---------------------------------------------------------------------------


def test_resolve_prefers_exact_then_model_template_temperature_then_identity() -> None:
    registry = CalibratorRegistry()
    assert registry.resolve(KEY).source == "identity"

    fallback = TemperatureCalibrator(2.0)
    registry.register_fallback(KEY.model, KEY.template_id, fallback)
    resolved = registry.resolve(KEY)
    assert (resolved.source, resolved.calibrator) == ("temperature_fallback", fallback)

    exact = fitted_vector()
    registry.register(KEY, exact)
    resolved = registry.resolve(KEY)
    assert (resolved.source, resolved.calibrator) == ("exact", exact)


@pytest.mark.parametrize(
    ("field", "value", "source"),
    [
        # The fallback is per (model, template): another backend or question still gets it.
        ("backend", "vllm_offline", "temperature_fallback"),
        ("signature", "choice:4:0000000000000000", "temperature_fallback"),
        ("template_id", "question_first-abc", "identity"),
        ("model", "Qwen/Qwen3-0.6B", "identity"),
    ],
)
def test_exact_entries_need_every_key_field(field: str, value: str, source: str) -> None:
    registry = CalibratorRegistry()
    registry.register(KEY, fitted_vector())
    registry.register_fallback(KEY.model, KEY.template_id, TemperatureCalibrator(2.0))
    other = dataclasses.replace(KEY, **{field: value})
    assert registry.resolve(other).source == source


def test_identity_fallback_leaves_probabilities_unchanged() -> None:
    resolved = CalibratorRegistry().resolve(KEY)
    np.testing.assert_allclose(resolved.calibrator.transform(LOGP), np.exp(LOGP), atol=1e-15)


def test_fallback_must_be_a_temperature() -> None:
    not_a_temperature: Any = IdentityCalibrator()
    with pytest.raises(TypeError, match="TemperatureCalibrator"):
        CalibratorRegistry().register_fallback("m", "t", not_a_temperature)


# --- persistence ----------------------------------------------------------------------------


def populated() -> CalibratorRegistry:
    registry = CalibratorRegistry()
    registry.register(KEY, fitted_vector())
    noul_key = CalibrationKey.for_question(
        backend="jev",
        model="jev-1.13.0",
        template_id="jev",
        question=NoulQuestion(instructions="Is it urgent?"),
    )
    registry.register(noul_key, TemperatureCalibrator(1.7))
    registry.register(
        CalibrationKey("vllm_http", "Qwen/Qwen3-8B", "question_first-abc", KEY.signature),
        fitted_isotonic(),
    )
    registry.register_fallback(KEY.model, KEY.template_id, TemperatureCalibrator(1.3))
    return registry


def test_save_load_round_trip_is_exact(tmp_path: Path) -> None:
    registry = populated()
    path = tmp_path / "calibrators.json"
    registry.save(path)
    loaded = CalibratorRegistry.load(path)
    assert loaded.to_json() == registry.to_json()
    for key in (KEY, CalibrationKey("other", KEY.model, KEY.template_id, "x")):
        original, restored = registry.resolve(key), loaded.resolve(key)
        assert restored.source == original.source
        assert np.array_equal(
            restored.calibrator.transform(LOGP), original.calibrator.transform(LOGP)
        )


def test_saved_file_is_independent_of_registration_order(tmp_path: Path) -> None:
    forward, backward = CalibratorRegistry(), CalibratorRegistry()
    keys = [CalibrationKey("b", "m", "t", f"choice:2:{i:016x}") for i in range(3)]
    for key in keys:
        forward.register(key, TemperatureCalibrator(1.5))
    for key in reversed(keys):
        backward.register(key, TemperatureCalibrator(1.5))
    forward.save(tmp_path / "f.json")
    backward.save(tmp_path / "b.json")
    assert (tmp_path / "f.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_load_rejects_a_non_temperature_fallback() -> None:
    document = populated().to_json()
    document["fallbacks"][0]["calibrator"] = {"name": "identity"}
    with pytest.raises(ValueError, match="temperature"):
        CalibratorRegistry.from_json(document)


def test_load_rejects_an_unknown_format_version() -> None:
    document = {**populated().to_json(), "version": 999}
    with pytest.raises(ValueError, match="version=999"):
        CalibratorRegistry.from_json(document)


def fitted(cls: type[Calibrator]) -> Calibrator:
    rng = np.random.default_rng(2)
    return cls().fit(np.log(rng.dirichlet(np.ones(5), size=100)), rng.integers(0, 5, size=100))


@pytest.mark.parametrize("name", sorted(CALIBRATORS))
def test_calibrator_from_json_dispatches_on_name(name: str) -> None:
    calibrator = fitted(CALIBRATORS[name])
    restored = calibrator_from_json(json.loads(json.dumps(calibrator.to_json())))
    assert type(restored) is CALIBRATORS[name]
    assert np.array_equal(restored.transform(LOGP), calibrator.transform(LOGP))


def test_calibrator_from_json_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown calibrator 'bogus'"):
        calibrator_from_json({"name": "bogus"})


# --- question signature ---------------------------------------------------------------------


def choice(criteria: dict[str, str | None], instructions: str = "Pick one.") -> ChoiceQuestion:
    return ChoiceQuestion(instructions=instructions, criteria=criteria)


def test_signature_format_names_type_and_key_count() -> None:
    assert re.fullmatch(r"choice:5:[0-9a-f]{16}", question_signature(GPQA))
    score = ScoreQuestion(instructions="Rate.", criteria=["low", "mid", "high"])
    assert re.fullmatch(r"score:3:[0-9a-f]{16}", question_signature(score))


def test_choice_signature_ignores_instructions_and_option_text() -> None:
    other_item = choice(
        {"A": "5 eV", "B": None, "C": "7 eV", "D": "8 eV", "E": "I don't know"},
        instructions="A different question with the same answer letters.",
    )
    assert question_signature(other_item) == question_signature(GPQA)


@pytest.mark.parametrize(
    "criteria",
    [
        {"B": None, "A": None, "C": None, "D": None, "E": None},  # same keys, other order
        {"A": None, "B": None, "C": None, "D": None},  # one key fewer
        {"a": None, "b": None, "c": None, "d": None, "e": None},  # other keys
    ],
)
def test_choice_signature_tracks_option_keys_and_their_order(
    criteria: dict[str, str | None],
) -> None:
    assert question_signature(choice(criteria)) != question_signature(GPQA)


def test_score_signature_tracks_level_descriptions() -> None:
    base = ScoreQuestion(instructions="Rate.", criteria=["low", "mid", "high"])
    reworded = ScoreQuestion(instructions="Rate again.", criteria=["low", "mid", "high"])
    other_scale = ScoreQuestion(instructions="Rate.", criteria=["none", "some", "all"])
    structured = ScoreQuestion(
        instructions="Rate.", criteria=[{"what": "low"}, {"what": "mid"}, {"what": "high"}]
    )
    assert question_signature(reworded) == question_signature(base)
    assert question_signature(other_scale) != question_signature(base)
    assert question_signature(structured) != question_signature(base)


def test_structured_level_signature_ignores_object_key_order() -> None:
    a = ScoreQuestion(instructions="Rate.", criteria=[{"what": "low", "examples": ["x"]}, "high"])
    b = ScoreQuestion(instructions="Rate.", criteria=[{"examples": ["x"], "what": "low"}, "high"])
    assert question_signature(a) == question_signature(b)


def test_question_types_never_share_a_signature() -> None:
    as_choice = choice({"0": None, "1": None})
    as_score = ScoreQuestion(instructions="Rate.", criteria=["0", "1"])
    as_noul = NoulQuestion(instructions="Yes?")
    as_true_false = choice({"true": None, "false": None})
    signatures = {question_signature(q) for q in (as_choice, as_score, as_noul, as_true_false)}
    assert len(signatures) == 4


def test_all_noul_questions_share_one_signature() -> None:
    bare = NoulQuestion(instructions="Is it urgent?")
    with_criteria = NoulQuestion(
        instructions="Is it a duplicate?", criteria={"true": "same record", "false": "different"}
    )
    assert question_signature(bare) == question_signature(with_criteria)
