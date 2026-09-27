"""``jevemu.compat.typesafe``: code from the TypeSafe SDK docs runs with only its import changed.

The example bodies below are copied from docs.typesafe.ai/sdk/python (quickstart, usage guide,
client reference); the only change is the import line of this module. A default emulator on a
scripted FakeBackend stands in for the API, so ``TypeSafeClient()`` needs no arguments.
"""

from __future__ import annotations

import asyncio
import math
import string
from collections.abc import Iterator, Sequence

import pytest
from pydantic import BaseModel

from jevemu.backends.base import RenderedPrompt
from jevemu.backends.fake import FakeBackend
from jevemu.compat import typesafe
from jevemu.compat.typesafe import (
    AsyncTypeSafeClient,
    Choice,
    Noul,
    NoulAnswer,
    Score,
    SystemOneResponse,
    TypeSafeAPITimeoutError,
    TypeSafeClient,
    TypeSafeError,
)
from jevemu.emulator import Emulator
from jevemu.render import CHOICE_LABELS
from jevemu.scoring import ScoreResult

VOCAB = [
    *CHOICE_LABELS,
    *(" " + label for label in CHOICE_LABELS),
    "Yes",
    " Yes",
    "No",
    " No",
    *string.printable,
]


def _table(prompt: RenderedPrompt, allowed: Sequence[str] | None) -> dict[str, float]:
    table = {" " + label: -0.5 * i for i, label in enumerate(CHOICE_LABELS)}
    table.update({digit: -0.4 * int(digit) for digit in string.digits})
    table.update({" Yes": math.log(0.9), " No": math.log(0.1)})
    return table


@pytest.fixture(autouse=True)
def default_emulator() -> Iterator[Emulator]:
    emulator = Emulator(FakeBackend(next_token=_table, vocab=VOCAB), round_to=0.01)
    typesafe.set_default_emulator(emulator)
    yield emulator
    typesafe.set_default_emulator(None)


def test_quickstart_async() -> None:
    async def main() -> None:
        async with AsyncTypeSafeClient() as client:
            response = await client.system_one(
                state={"document": "I was charged twice. Please fix this ASAP."},
                questions={
                    "billing": Noul(instructions="Is this ticket about billing?"),
                    "tone": Choice(
                        instructions="What is the customer's tone?",
                        criteria={"calm": None, "frustrated": None, "angry": None},
                    ),
                    "urgency": Score(
                        instructions="How urgent is this ticket?",
                        criteria=["can wait", "this week", "today"],
                    ),
                },
            )

        print(response.nouls["billing"].noul)
        print(response.choices["tone"].choice)
        print(response.scores["urgency"].score)

        assert response.nouls["billing"].noul == 0.9
        assert response.choices["tone"].choice == "calm"
        urgency = response.scores["urgency"].probabilities
        assert urgency[0] > urgency[1]  # integer level keys, as in the SDK

    asyncio.run(main())


def test_quickstart_sync() -> None:
    with TypeSafeClient() as client:
        response = client.system_one(
            state={"document": "I was charged twice. Please fix this ASAP."},
            questions={
                "billing": Noul(instructions="Is this ticket about billing?"),
                "tone": Choice(
                    instructions="What is the customer's tone?",
                    criteria={"calm": None, "frustrated": None, "angry": None},
                ),
                "urgency": Score(
                    instructions="How urgent is this ticket?",
                    criteria=["can wait", "this week", "today"],
                ),
            },
        )

    print(response.nouls["billing"].noul)
    print(response.choices["tone"].choice)
    print(response.scores["urgency"].score)

    assert set(response.answers) == {"billing", "tone", "urgency"}
    assert response.scores["urgency"].legend == {0: "can wait", 1: "this week", 2: "today"}
    assert response.model == "jevemu/fake-model"


def test_usage_guide_positional_arguments() -> None:
    client = TypeSafeClient()
    state = "I was charged twice. Please help ASAP."
    questions = {
        "billing": Noul(instructions="Is this about billing?"),
        "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": None}),
        "urgency": Score(instructions="How urgent is this?", criteria=["low", "medium", "high"]),
    }
    result = client.system_one(state, questions)
    print(
        result.nouls["billing"].noul,
        result.choices["tone"].choice,
        result.scores["urgency"].score,
    )
    client.close()


def test_questions_as_dictionaries() -> None:
    with TypeSafeClient() as client:
        result = client.system_one(
            state={"message": "I was charged twice. Please help."},
            questions={
                "billing": {"type": "noul", "instructions": "Is this about billing?"},
                "tone": {
                    "type": "choice",
                    "instructions": "What is the tone?",
                    "criteria": {"calm": None, "angry": None},
                },
            },
        )
        assert 0 <= result.nouls["billing"].noul <= 1
        assert result.choices["tone"].choice in {"calm", "angry"}


def test_typed_response_models() -> None:
    class BillingResponse(SystemOneResponse):
        billing: NoulAnswer

    with TypeSafeClient() as client:
        result = client.system_one(
            "I was charged twice.",
            {"billing": Noul(instructions="Is this about billing?")},
            response_model=BillingResponse,
        )
        assert 0 <= result.billing.noul <= 1
        assert result.billing == result.nouls["billing"]
        print(result.request_id)
        assert result.request_id.startswith("jevemu-")

    class BillingAnswers(BaseModel):
        billing: NoulAnswer

    class CustomResponse(BaseModel):
        answers: BillingAnswers

    custom = TypeSafeClient().system_one(
        "I was charged twice.",
        {"billing": Noul(instructions="Is this about billing?")},
        response_model=CustomResponse,
    )
    assert 0 <= custom.answers.billing.noul <= 1


def test_models_list_names_the_emulated_model() -> None:
    print(TypeSafeClient().models.list())

    async def main() -> None:
        async with AsyncTypeSafeClient() as client:
            models = await client.models.list()
        assert [m.name for m in models.models] == ["jevemu/fake-model"]

    asyncio.run(main())


def test_invalid_questions_raise_typesafe_error() -> None:
    with TypeSafeClient() as client:
        with pytest.raises(TypeSafeError):
            client.system_one("text", {})
        with pytest.raises(TypeSafeError):
            client.system_one("text", {"q": {"type": "score", "instructions": "?", "criteria": []}})


class _NeverAnswers:
    name = "never"

    def supports(self, capabilities: object, question: object) -> bool:
        return True

    async def score(self, *args: object) -> ScoreResult:
        await asyncio.sleep(10.0)
        raise AssertionError("the timeout should have cancelled this")


def test_timeout_raises_typesafe_timeout_error() -> None:
    slow = Emulator(FakeBackend(next_token=_table, vocab=VOCAB), strategy=_NeverAnswers())
    with TypeSafeClient(emulator=slow) as client, pytest.raises(TypeSafeAPITimeoutError) as info:
        client.system_one("text", {"q": Noul(instructions="?")}, timeout=0.05)
    assert info.value.timeout == 0.05
    assert isinstance(info.value, TimeoutError)
