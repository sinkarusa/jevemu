"""Renderer tests: byte-exact snapshots and shared-prefix stability.

Snapshots live in ``snapshots/<case>__<layout>.txt``. After an intended template change,
regenerate them with ``JEVEMU_UPDATE_SNAPSHOTS=1 uv run pytest tests/unit/render`` and review
the diff: every changed prompt invalidates calibrators keyed by its ``template_id``.
"""

from __future__ import annotations

import os
from os.path import commonprefix
from pathlib import Path

import jinja2
import pytest

from jevemu.render import (
    LAYOUTS,
    LabelScheme,
    Layout,
    PromptRenderer,
    scheme_for,
    sequence_scheme_for,
)
from jevemu.render import renderer as renderer_module
from jevemu.types import (
    ChoiceQuestion,
    JSONish,
    NoulQuestion,
    Question,
    ScoreQuestion,
    SystemOneRequest,
)

SNAPSHOTS = Path(__file__).parent / "snapshots"
JEV_DOCS = Path(__file__).parents[2] / "golden" / "fixtures" / "jev_docs"
UPDATE = os.environ.get("JEVEMU_UPDATE_SNAPSHOTS") == "1"

# Real TypeSafe doc examples: (fixture stem, question id).
DOC_CASES: dict[str, tuple[str, str]] = {
    "choice_text": ("primitives_choice__01", "department"),
    "choice_null_descriptions": ("primitives_advanced__01", "customer_name"),
    "choice_structured": ("primitives_advanced__03", "department"),
    "score_text": ("primitives_score__01", "bug_severity"),
    "score_structured": ("primitives_advanced__04", "pr_scope"),
    "noul_bare": ("primitives_noul__01", "is_human_escalation"),
    "noul_criteria": ("primitives_noul__01", "is_repeat_contact"),
    "noul_structured": ("primitives_advanced__05", "requests_credentials"),
}

# Nulls wherever the docs allow them, array state, caller key order, non-ASCII.
SYNTHETIC_CASES: dict[str, tuple[JSONish, Question]] = {
    "score_nulls": (
        ["Order 1182 arrived two days late.", "The box was crushed; the mug inside is fine."],
        ScoreQuestion(
            instructions=None,
            criteria=["No damage", None, {"what": "Item unusable", "examples": ["shattered"]}],
        ),
    ),
    "noul_nulls": (
        {"zeta_last_name": "Müller", "alpha_note": "Café order, paid in €"},
        NoulQuestion(
            instructions=None, criteria={"false": "No complaint in the note", "true": None}
        ),
    ),
}


def _doc_case(stem: str, question_id: str) -> tuple[JSONish, Question]:
    request = SystemOneRequest.model_validate_json(
        (JEV_DOCS / f"{stem}__request.json").read_bytes()
    )
    return request.state, request.questions[question_id]


CASES: dict[str, tuple[JSONish, Question]] = {
    **{name: _doc_case(*source) for name, source in DOC_CASES.items()},
    **SYNTHETIC_CASES,
}


def _serialize(prompt_messages: list[tuple[str, str]], template_id: str) -> bytes:
    parts = [f"template_id: {template_id}"]
    parts += [f"<<< {role} >>>\n{content}" for role, content in prompt_messages]
    return ("\n".join(parts) + "\n").encode()


def _check_snapshot(name: str, actual: bytes) -> None:
    path = SNAPSHOTS / f"{name}.txt"
    if UPDATE:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(actual)
    assert path.read_bytes() == actual


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("case", sorted(CASES))
def test_snapshot(case: str, layout: Layout) -> None:
    state, question = CASES[case]
    renderer = PromptRenderer(layout)
    prompt = renderer.render(state, question, scheme_for(question))
    actual = _serialize([(m.role, m.content) for m in prompt.messages], prompt.template_id)
    _check_snapshot(f"{case}__{layout}", actual)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("case", sorted(c for c in CASES if c.startswith("choice_")))
def test_key_scheme_snapshot(case: str, layout: Layout) -> None:
    """Echo and trie list Choice options by key, without letters."""
    state, question = CASES[case]
    prompt = PromptRenderer(layout).render(state, question, sequence_scheme_for(question))
    actual = _serialize([(m.role, m.content) for m in prompt.messages], prompt.template_id)
    _check_snapshot(f"{case}__keys__{layout}", actual)


def _department() -> ChoiceQuestion:
    _, question = CASES["choice_text"]
    assert isinstance(question, ChoiceQuestion)
    return question


def test_question_first_system_message_is_shared_across_states() -> None:
    renderer = PromptRenderer("question_first")
    question = _department()
    states: list[JSONish] = [
        "Where is my package?",
        {"ticket": {"subject": "Refund", "body": "I was charged twice."}},
        ["line one", "line two"],
    ]
    prompts = [renderer.render(state, question, scheme_for(question)) for state in states]
    assert {p.messages[0] for p in prompts} == {prompts[0].messages[0]}
    assert prompts[0].messages[0].role == "system"
    assert len({p.messages[1].content for p in prompts}) == len(states)


def test_state_first_user_prefix_holds_the_whole_state_across_questions() -> None:
    renderer = PromptRenderer("state_first")
    state = {"ticket": {"body": "Tracking says label created."}, "tail": "END-OF-STATE"}
    questions: list[Question] = [
        _department(),
        ScoreQuestion(instructions="How upset?", criteria=["Calm", "Annoyed", "Furious"]),
        NoulQuestion(instructions=None),
        NoulQuestion(instructions="Was the customer charged?"),
    ]
    prompts = [renderer.render(state, q, scheme_for(q)) for q in questions]
    assert {tuple(m.role for m in p.messages) for p in prompts} == {("user", "assistant")}
    shared = commonprefix([p.messages[0].content for p in prompts])
    assert "END-OF-STATE" in shared


def test_options_follow_the_scheme_order() -> None:
    question = ChoiceQuestion(
        instructions="Pick.", criteria={"red": None, "green": "grass", "blue": None}
    )
    rotated = LabelScheme("letters", ("A", "B", "C"), ("blue", "red", "green"))
    content = PromptRenderer().render("s", question, rotated).messages[0].content
    assert "A) blue\nB) red\nC) green: grass\n" in content


@pytest.mark.parametrize(
    ("question", "scheme"),
    [
        (_department(), LabelScheme("letters", ("A", "B"), ("returns", "shipping"))),
        (_department(), LabelScheme("letters", ("A", "B", "C"), ("returns", "shipping", "x"))),
        (
            ScoreQuestion(instructions="Rate.", criteria=["low", "high"]),
            LabelScheme("digits", ("0", "1", "2"), ("0", "1", "2")),
        ),
        (NoulQuestion(instructions="Yes?"), LabelScheme("letters", ("A", "B"), ("x", "y"))),
    ],
    ids=["missing-option", "unknown-option", "extra-level", "noul-with-choice-keys"],
)
def test_scheme_must_cover_exactly_the_questions_options(
    question: Question, scheme: LabelScheme
) -> None:
    with pytest.raises(ValueError, match="do not match"):
        PromptRenderer().render("s", question, scheme)


def test_unknown_layout_rejected() -> None:
    with pytest.raises(ValueError, match="unknown layout"):
        PromptRenderer("answer_first")  # type: ignore[arg-type]


def _ids_with_sources(monkeypatch: pytest.MonkeyPatch, sources: dict[str, str]) -> dict[str, str]:
    env = jinja2.Environment(loader=jinja2.DictLoader(sources))
    env.filters["jsonish"] = renderer_module.render_value
    monkeypatch.setattr(renderer_module, "_environment", lambda: env)
    return {layout: PromptRenderer(layout).template_id for layout in LAYOUTS}


def test_template_id_tracks_imported_template_edits(monkeypatch: pytest.MonkeyPatch) -> None:
    real = renderer_module._environment()
    assert real.loader is not None
    sources = {
        name: real.loader.get_source(real, name)[0]
        for name in ("question_first.j2", "state_first.j2", "_blocks.j2")
    }
    shipped = {layout: PromptRenderer(layout).template_id for layout in LAYOUTS}

    assert _ids_with_sources(monkeypatch, sources) == shipped
    edited = {**sources, "_blocks.j2": sources["_blocks.j2"].replace("only.", "only!")}
    for layout, template_id in _ids_with_sources(monkeypatch, edited).items():
        assert template_id != shipped[layout]
        assert template_id.startswith(f"{layout}-")
