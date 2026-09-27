from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jevemu.confidence import (
    CONFIDENCE_FUNCTIONS,
    ConfidenceFn,
    margin,
    max_prob,
    one_minus_norm_entropy,
    peak_linear,
)

FIXTURE_DIR = Path(__file__).parents[2] / "golden" / "fixtures" / "jev_docs"
ROUNDING = 0.006
"""Jev reports confidence to 0.01 and apparently computes it from unrounded probabilities, so
the formula on displayed probabilities may be off by up to 0.005 (INDEX.md, confidence.md)."""

KNOWN_MISMATCHES: dict[tuple[str, str], float] = {
    # 0.88/0.12/0.0 -> formula 0.82, reported 0.81: off by 0.01, beyond rounding.
    ("api__03__response.json", "department"): 0.82,
}
"""Fixture answers where ``peak_linear`` does not reproduce Jev, with the formula's value."""

WIDGET_MISMATCHES = [
    # docs.typesafe.ai/primitives/score ScoreExplorer widget data (INDEX.md "Skipped"): answer
    # objects without a request, so they are not fixtures. Score confidence may use another rule.
    pytest.param([0.0, 0.14, 0.86, 0.0, 0.0], 0.89, 0.825, id="outfit_formality_K5"),
    pytest.param([0.0, 0.0, 0.48, 0.52], 0.52, 0.36, id="candidate_fit_K4"),
]


def doc_answers() -> list[tuple[str, str, list[float], float]]:
    """(fixture, question id, probabilities, reported confidence) of every Choice/Score answer."""
    answers = []
    for path in sorted(FIXTURE_DIR.glob("*__response.json")):
        for qid, answer in json.loads(path.read_text())["answers"].items():
            if "confidence" in answer:
                probs = list(answer["probabilities"].values())
                answers.append((path.name, qid, probs, answer["confidence"]))
    return answers


DOC_ANSWERS = doc_answers()


def doc_params(*, mismatched: bool) -> list[object]:
    return [
        pytest.param(*a, id=f"{a[0].removesuffix('__response.json')}:{a[1]}")
        for a in DOC_ANSWERS
        if (a[:2] in KNOWN_MISMATCHES) == mismatched
    ]


def test_every_doc_answer_with_a_confidence_is_checked() -> None:
    assert len(DOC_ANSWERS) == 16
    assert set(KNOWN_MISMATCHES) <= {(name, qid) for name, qid, _, _ in DOC_ANSWERS}


@pytest.mark.parametrize(("fixture", "qid", "probs", "reported"), doc_params(mismatched=False))
def test_peak_linear_matches_jev_doc_examples_within_rounding(
    fixture: str, qid: str, probs: list[float], reported: float
) -> None:
    assert peak_linear(probs) == pytest.approx(reported, abs=ROUNDING)


@pytest.mark.parametrize(("fixture", "qid", "probs", "reported"), doc_params(mismatched=True))
def test_peak_linear_known_mismatches_with_jev_doc_examples(
    fixture: str, qid: str, probs: list[float], reported: float
) -> None:
    formula = peak_linear(probs)
    assert formula == pytest.approx(KNOWN_MISMATCHES[(fixture, qid)], abs=1e-9)
    assert abs(formula - reported) > ROUNDING


@pytest.mark.parametrize(("probs", "reported", "formula"), WIDGET_MISMATCHES)
def test_peak_linear_known_mismatches_with_score_widget_examples(
    probs: list[float], reported: float, formula: float
) -> None:
    assert peak_linear(probs) == pytest.approx(formula, abs=1e-9)
    assert abs(peak_linear(probs) - reported) > ROUNDING


# --- behavior at the boundaries --------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 5, 255])
def test_uniform_is_zero_and_one_hot_is_one(k: int) -> None:
    uniform = [1.0 / k] * k
    one_hot = [1.0] + [0.0] * (k - 1)
    for fn in (peak_linear, margin, one_minus_norm_entropy):
        assert fn(uniform) == pytest.approx(0.0, abs=1e-12)
    for fn in CONFIDENCE_FUNCTIONS.values():
        assert fn(one_hot) == 1.0
    assert max_prob(uniform) == pytest.approx(1.0 / k)


def test_peak_linear_clips_rounded_probabilities_below_one_over_k() -> None:
    # Two-decimal rounding can leave p_max < 1/K: the raw formula gives -0.005.
    assert peak_linear([0.33, 0.33, 0.33]) == 0.0


def test_margin_is_the_gap_between_the_two_largest() -> None:
    assert margin([0.2, 0.5, 0.3]) == pytest.approx(0.2)
    assert margin([0.45, 0.1, 0.45]) == 0.0


def test_entropy_confidence_scales_with_key_count() -> None:
    # The same two-way split is less decisive among 2 keys than among 4.
    assert one_minus_norm_entropy([0.5, 0.5]) == pytest.approx(0.0, abs=1e-12)
    assert one_minus_norm_entropy([0.5, 0.5, 0.0, 0.0]) == pytest.approx(0.5)


@pytest.mark.parametrize("fn", CONFIDENCE_FUNCTIONS.values(), ids=CONFIDENCE_FUNCTIONS.keys())
@pytest.mark.parametrize("bad", [[1.0], [], [0.5, -0.1, 0.6], [math.nan, 0.5]])
def test_invalid_probability_vectors_are_rejected(fn: ConfidenceFn, bad: list[float]) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        fn(bad)


@st.composite
def distributions(draw: st.DrawFn) -> list[float]:
    weights = draw(st.lists(st.floats(0.0, 1.0), min_size=2, max_size=12))
    total = sum(weights)
    if total == 0.0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


@given(distributions(), st.randoms())
def test_confidence_is_in_unit_interval_and_ignores_key_order(
    probs: list[float], rnd: random.Random
) -> None:
    # Jev's Choice probabilities come back in shuffled key order.
    shuffled = list(probs)
    rnd.shuffle(shuffled)
    for fn in CONFIDENCE_FUNCTIONS.values():
        value = fn(probs)
        assert 0.0 <= value <= 1.0
        assert fn(shuffled) == pytest.approx(value, abs=1e-12)
