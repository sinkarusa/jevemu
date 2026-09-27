from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jevemu.backends.base import SeqScore, TokenLogprob
from jevemu.render import choice_scheme
from jevemu.scoring.aggregate import (
    logsumexp,
    low_mass_warning,
    merge_variants,
    observed_mass,
    observed_surfaces,
    renormalize,
    sequence_logprob,
)

SCHEME = choice_scheme(["x", "y", "z"])
SURFACES = {s: SCHEME.label_of_surface(s) or "" for s in SCHEME.surfaces}


def top(*entries: tuple[str, float]) -> tuple[TokenLogprob, ...]:
    return tuple(TokenLogprob(token, None, logprob) for token, logprob in entries)


def test_variants_merge_by_summing_probability() -> None:
    seen = observed_surfaces(
        top((" A", math.log(0.5)), ("A", math.log(0.1)), (" B", math.log(0.2))), SURFACES
    )
    merged = merge_variants(seen, SURFACES)
    assert math.exp(merged["A"]) == pytest.approx(0.6)
    assert math.exp(merged["B"]) == pytest.approx(0.2)
    assert "C" not in merged  # absent, never zero


def test_floored_entries_are_not_observations() -> None:
    seen = observed_surfaces(
        top((" A", math.log(0.9)), (" B", -math.inf), ("C", -math.inf)), SURFACES
    )
    assert set(seen) == {" A"}


def test_duplicate_token_texts_are_distinct_tokens() -> None:
    seen = observed_surfaces(top((" A", math.log(0.2)), (" A", math.log(0.3))), SURFACES)
    assert math.exp(seen[" A"]) == pytest.approx(0.5)


def test_renormalize_keeps_zero_and_rejects_no_mass() -> None:
    out = renormalize([math.log(0.2), -math.inf, math.log(0.6)])
    assert [math.exp(v) for v in out] == pytest.approx([0.25, 0.0, 0.75])
    with pytest.raises(ValueError, match="no valid label"):
        renormalize([-math.inf, -math.inf])
    with pytest.raises(ValueError, match="NaN"):
        renormalize([0.0, math.nan])


def test_unbounded_scores_share_all_the_mass() -> None:
    out = renormalize([math.inf, 3.0, math.inf])
    assert [math.exp(v) for v in out] == pytest.approx([0.5, 0.0, 0.5])


def test_logsumexp_is_stable_and_rejects_non_logprobs() -> None:
    assert logsumexp([1000.0, 1000.0]) == pytest.approx(1000.0 + math.log(2))
    assert logsumexp([]) == -math.inf
    assert logsumexp([-math.inf, -math.inf]) == -math.inf
    with pytest.raises(ValueError, match="not a logprob"):
        logsumexp([0.0, math.nan])


def test_observed_mass_sums_and_caps_overlapping_echo_mass() -> None:
    assert observed_mass([math.log(0.2), math.log(0.1)]) == pytest.approx(0.3)
    assert observed_mass([]) == 0.0
    assert observed_mass([math.log(0.8), math.log(0.7)]) == 1.0


@pytest.mark.parametrize(
    ("mass", "mode", "warns"),
    [
        (0.49, "raw_logprobs", True),
        (0.5, "raw_logprobs", False),
        (0.1, "processed_logprobs", False),
    ],
)
def test_low_mass_warns_only_below_half_under_raw_logprobs(
    mass: float, mode: str, warns: bool
) -> None:
    warning = low_mass_warning(mass, mode)  # type: ignore[arg-type]
    assert (warning is not None) == warns


def test_sequence_logprob_floors_to_zero_and_rejects_empty_scores() -> None:
    floored = SeqScore(" x", (" ", "x"), (math.log(0.5), -math.inf))
    assert sequence_logprob(floored) == -math.inf
    with pytest.raises(ValueError, match="no tokens"):
        sequence_logprob(SeqScore(" x", (), ()))
    with pytest.raises(ValueError, match="nan"):
        sequence_logprob(SeqScore(" x", ("x",), (math.nan,)))


# --- properties -------------------------------------------------------------------------------

_logprob = st.floats(min_value=-30.0, max_value=0.0, allow_nan=False)
_label_surfaces = st.dictionaries(st.sampled_from(sorted(SURFACES)), _logprob, min_size=1)
_junk = st.dictionaries(st.text(alphabet="!?*#xyz\n", min_size=1, max_size=3), _logprob, max_size=8)


def _probabilities(entries: dict[str, float]) -> dict[str, float]:
    merged = merge_variants(observed_surfaces(top(*entries.items()), SURFACES), SURFACES)
    labels = sorted(merged)
    return dict(
        zip(labels, (math.exp(v) for v in renormalize([merged[k] for k in labels])), strict=True)
    )


@given(_label_surfaces, _junk)
def test_probabilities_are_non_negative_and_sum_to_one(
    labels: dict[str, float], junk: dict[str, float]
) -> None:
    probabilities = _probabilities({**junk, **labels})
    assert all(p >= 0.0 for p in probabilities.values())
    assert math.fsum(probabilities.values()) == pytest.approx(1.0)


@given(_label_surfaces, _junk, st.lists(st.sampled_from(sorted(SURFACES))))
def test_invalid_and_floored_tokens_do_not_change_the_result(
    labels: dict[str, float], junk: dict[str, float], floored: list[str]
) -> None:
    extra = {**junk, **{s: -math.inf for s in floored if s not in labels}}
    assert _probabilities({**extra, **labels}) == pytest.approx(_probabilities(labels))
