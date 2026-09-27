"""Questions and strategies of the scoring golden set (shared by the recorder and the replay).

Not a test module. Re-record after any change to these cases, the prompt templates or the
strategies' requests (see ``record_scoring_fixtures.py``).
"""

from __future__ import annotations

from pathlib import Path

from jevemu.render import Layout
from jevemu.scoring import (
    AutoStrategy,
    ConstrainedStrategy,
    EchoStrategy,
    FirstTokenStrategy,
    ScoringStrategy,
    TrieStrategy,
)
from jevemu.types import ChoiceQuestion, JSONish, NoulQuestion, Question, ScoreQuestion

HERE = Path(__file__).parent
FIXTURE = HERE / "scoring_vllm.jsonl"
EXPECTED = HERE / "expected.json"

COUNTRIES = (  # noqa: SIM905 - 60 option keys, readable as one block
    "Albania Argentina Australia Austria Belgium Bolivia Brazil Bulgaria Canada Chile China "
    "Colombia Croatia Cuba Denmark Ecuador Egypt Estonia Finland France Germany Ghana Greece "
    "Hungary Iceland India Indonesia Iran Iraq Ireland Israel Italy Jamaica Japan Kenya Latvia "
    "Lebanon Libya Malta Mexico Morocco Nepal Netherlands Nigeria Norway Pakistan Peru Poland "
    "Portugal Romania Russia Senegal Serbia Spain Sweden Switzerland Thailand Turkey Ukraine "
    "Vietnam"
).split()


def strategies() -> dict[str, ScoringStrategy]:
    return {
        s.name: s
        for s in (
            FirstTokenStrategy(),
            ConstrainedStrategy(),
            TrieStrategy(),
            EchoStrategy("sum"),
            EchoStrategy("mean"),
            EchoStrategy("pmi"),
            AutoStrategy(),
        )
    }


# (layout, state, question, strategy names); ``None`` means every strategy.
CASES: dict[str, tuple[Layout, JSONish, Question, tuple[str, ...] | None]] = {
    "choice_easy": (
        "question_first",
        "I bought a banana at the market.",
        ChoiceQuestion(
            instructions="What kind of item is this?",
            criteria={"fruit": None, "tool": None, "vehicle": None},
        ),
        None,
    ),
    "choice_prefix_keys": (
        "state_first",
        {"source_text": "Invoice #4471 issued March 3, 2026 to Beaver Dam Logistics for $12,840."},
        ChoiceQuestion(
            instructions="Who was the invoice issued to?",
            criteria={
                "Beaver Logistics": None,
                "Dam Logistics": None,
                "Beaver Dam Logistics": None,
                "Beaver": None,
                "Dam": None,
            },
        ),
        None,
    ),
    "score_damage": (
        "question_first",
        "The package arrived completely destroyed and the item is unusable.",
        ScoreQuestion(
            instructions="How severe is the damage?",
            criteria=["none", "minor", "moderate", "severe", "total loss"],
        ),
        None,
    ),
    "noul_human": (
        "state_first",
        "Customer: I want to talk to a human right now!",
        NoulQuestion(
            instructions="Does the customer ask for a human agent?",
            criteria={"true": "Explicitly asks for a person", "false": None},
        ),
        None,
    ),
    "choice_60_countries": (
        "question_first",
        "The Eiffel Tower is in its capital.",
        ChoiceQuestion(
            instructions="Which country is this about?", criteria=dict.fromkeys(COUNTRIES)
        ),
        ("trie", "echo_sum", "auto"),
    ),
}


def pairs() -> list[tuple[str, str]]:
    """Every (case, strategy name) in the golden set."""
    names = list(strategies())
    return [(case, name) for case, spec in CASES.items() for name in (spec[3] or names)]
