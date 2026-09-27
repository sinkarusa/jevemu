from __future__ import annotations

import string

import pytest

from jevemu.backends.fake import FakeBackend
from jevemu.render import (
    LabelCapacityError,
    LabelScheme,
    choice_scheme,
    scheme_for,
    score_scheme,
    verify_single_token,
)
from jevemu.types import ChoiceQuestion, NoulQuestion, ScoreQuestion


def _choice(n_options: int) -> ChoiceQuestion:
    return ChoiceQuestion(
        instructions="Pick one.", criteria={f"option {i}": None for i in range(n_options)}
    )


def _score(n_levels: int) -> ScoreQuestion:
    return ScoreQuestion(instructions="Rate it.", criteria=[f"level {i}" for i in range(n_levels)])


@pytest.mark.parametrize(
    ("n_options", "labels"),
    [
        (2, ("A", "B")),
        (26, tuple(string.ascii_uppercase)),
        (27, (*string.ascii_uppercase, "a")),
        (52, tuple(string.ascii_uppercase + string.ascii_lowercase)),
    ],
)
def test_choice_letters_run_upper_then_lower(n_options: int, labels: tuple[str, ...]) -> None:
    question = _choice(n_options)
    scheme = scheme_for(question)
    assert scheme.name == "letters"
    assert scheme.labels == labels
    assert scheme.keys == tuple(question.criteria)


def test_choice_above_52_options_raises_capacity_error() -> None:
    with pytest.raises(LabelCapacityError) as excinfo:
        scheme_for(_choice(53))
    assert (excinfo.value.scheme, excinfo.value.needed, excinfo.value.capacity) == (
        "letters",
        53,
        52,
    )


@pytest.mark.parametrize("n_levels", [2, 10])
def test_score_digits_label_level_indices(n_levels: int) -> None:
    scheme = scheme_for(_score(n_levels))
    assert scheme.name == "digits"
    assert scheme.labels == tuple(str(i) for i in range(n_levels))
    assert scheme.keys == tuple(str(i) for i in range(n_levels))


def test_score_above_10_levels_raises_capacity_error() -> None:
    with pytest.raises(LabelCapacityError) as excinfo:
        score_scheme(11)
    assert (excinfo.value.scheme, excinfo.value.needed, excinfo.value.capacity) == (
        "digits",
        11,
        10,
    )


def test_noul_yes_means_true() -> None:
    scheme = scheme_for(NoulQuestion(instructions="Is it urgent?"))
    assert scheme.name == "yes_no"
    assert scheme.labels == ("Yes", "No")
    assert scheme.key_of("Yes") == "true"
    assert scheme.key_of("No") == "false"


def test_labels_and_keys_map_both_ways_in_option_order() -> None:
    options = [f"opt-{i}" for i in range(52)]
    scheme = choice_scheme(options)
    for label, key in zip(scheme.labels, options, strict=True):
        assert scheme.key_of(label) == key
        assert scheme.label_of(key) == label
    with pytest.raises(KeyError):
        scheme.key_of("?")
    with pytest.raises(KeyError):
        scheme.label_of("not an option")


def test_surfaces_merge_bare_and_space_prefixed_only() -> None:
    scheme = choice_scheme(["x", "y", "z"])
    assert scheme.variants("B") == ("B", " B")
    assert scheme.surfaces == ("A", " A", "B", " B", "C", " C")
    assert scheme.label_of_surface(" C") == "C"
    assert scheme.label_of_surface("C") == "C"
    for other in ("  C", "C ", "\nC", "c", " c", "D", ""):
        assert scheme.label_of_surface(other) is None
    with pytest.raises(KeyError):
        scheme.variants("D")


def test_lowercase_labels_are_distinct_from_uppercase() -> None:
    scheme = choice_scheme([f"opt-{i}" for i in range(30)])
    assert scheme.label_of_surface(" a") == "a"
    assert scheme.key_of("a") == "opt-26"
    assert scheme.key_of("A") == "opt-0"


@pytest.mark.parametrize(
    ("labels", "keys"),
    [
        (("A", "B"), ("x",)),
        (("A", "B"), ("x", "x")),
        (("A", " A"), ("x", "y")),
    ],
    ids=["length-mismatch", "duplicate-keys", "colliding-surfaces"],
)
def test_inconsistent_scheme_rejected(labels: tuple[str, ...], keys: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - each case has its own message
        LabelScheme("letters", labels, keys)


_VOCAB = ["A", " A", "B", " ", "Yes", " Yes", "No"]


async def test_verify_single_token_checks_every_surface() -> None:
    backend = FakeBackend(vocab=_VOCAB)
    assert await verify_single_token(backend, choice_scheme(["x", "y"])) == {
        "A": True,
        " A": True,
        "B": True,
        " B": False,
    }
    assert await verify_single_token(backend, ["Yes", "No"]) == {
        "Yes": True,
        " Yes": True,
        "No": True,
        " No": False,
    }


class _BosBackend(FakeBackend):
    """Tokenizer that prepends a BOS id to every input, like vLLM's default /tokenize."""

    async def tokenize(self, text: str) -> list[int]:
        return [0, *await super().tokenize(text)]


async def test_verify_single_token_discounts_special_tokens() -> None:
    backend = _BosBackend(vocab=_VOCAB)
    assert await verify_single_token(backend, ["A", "B"]) == {
        "A": True,
        " A": True,
        "B": True,
        " B": False,
    }


async def test_verify_single_token_rejects_a_bare_string() -> None:
    with pytest.raises(TypeError):
        await verify_single_token(FakeBackend(vocab=_VOCAB), "AB")
