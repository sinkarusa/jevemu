"""Every scoring strategy against a live vLLM server (``scripts/serve_vllm.sh``).

Skipped unless ``JEVEMU_VLLM_URL`` is set; ``JEVEMU_VLLM_MODEL`` (default Qwen/Qwen3-0.6B) and
``JEVEMU_VLLM_MAX_LOGPROBS`` (default 576) must match the server launch. Capabilities come from
the capability probe (cached per server identity).
"""

from __future__ import annotations

import math
import os
import uuid
from collections.abc import AsyncIterator

import pytest

from jevemu.backends.probe import ServerFlags, probe_capabilities
from jevemu.backends.vllm_http import VLLMHTTPBackend
from jevemu.render import PromptRenderer, sequence_scheme_for
from jevemu.scoring import (
    AutoStrategy,
    ConstrainedStrategy,
    EchoStrategy,
    FirstTokenStrategy,
    ScoringStrategy,
    TrieStrategy,
)
from jevemu.types import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

URL = os.environ.get("JEVEMU_VLLM_URL", "")
MODEL = os.environ.get("JEVEMU_VLLM_MODEL", "Qwen/Qwen3-0.6B")
MAX_LOGPROBS = int(os.environ.get("JEVEMU_VLLM_MAX_LOGPROBS", "576"))

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not URL, reason="set JEVEMU_VLLM_URL to run live vLLM tests"),
]

STRATEGIES: list[ScoringStrategy] = [
    FirstTokenStrategy(),
    ConstrainedStrategy(),
    TrieStrategy(),
    EchoStrategy("sum"),
    EchoStrategy("mean"),
    EchoStrategy("pmi"),
    AutoStrategy(),
]

EASY = ChoiceQuestion(
    instructions="What kind of item is this?",
    criteria={"fruit": None, "tool": None, "vehicle": None},
)
QUESTIONS: dict[str, tuple[str, Question]] = {
    "choice": ("I bought a banana at the market.", EASY),
    "score": (
        "The login page throws a 500 error for all users since the last deploy.",
        ScoreQuestion(
            instructions="Rate the bug severity.",
            criteria=["cosmetic", "minor", "major", "critical"],
        ),
    ),
    "noul": (
        "Customer: this is the third time I write about the same refund.",
        NoulQuestion(instructions="Is this a repeat contact?"),
    ),
}
COUNTRY_NAMES = (  # noqa: SIM905 - 60 option keys, readable as one block
    "Albania Argentina Australia Austria Belgium Bolivia Brazil Bulgaria Canada Chile China "
    "Colombia Croatia Cuba Denmark Ecuador Egypt Estonia Finland France Germany Ghana Greece "
    "Hungary Iceland India Indonesia Iran Iraq Ireland Israel Italy Jamaica Japan Kenya Latvia "
    "Lebanon Libya Malta Mexico Morocco Nepal Netherlands Nigeria Norway Pakistan Peru Poland "
    "Portugal Romania Russia Senegal Serbia Spain Sweden Switzerland Thailand Turkey Ukraine "
    "Vietnam"
).split()
COUNTRIES = ChoiceQuestion(
    instructions="Which country is this about?", criteria=dict.fromkeys(COUNTRY_NAMES)
)


@pytest.fixture
async def backend() -> AsyncIterator[VLLMHTTPBackend]:
    async with VLLMHTTPBackend(URL, MODEL) as live:
        live.capabilities = await probe_capabilities(live, ServerFlags(max_logprobs=MAX_LOGPROBS))
        yield live


def renderer() -> PromptRenderer:
    return PromptRenderer("question_first")


def fresh(state: str) -> str:
    """A state no earlier request has cached, so results do not depend on test order."""
    return f"{state}\n(ref {uuid.uuid4().hex[:8]})"


def assert_valid(probabilities: dict[str, float], question: Question) -> None:
    assert tuple(probabilities) == sequence_scheme_for(question).keys
    assert all(p >= 0.0 and not math.isnan(p) for p in probabilities.values())
    assert math.fsum(probabilities.values()) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("kind", sorted(QUESTIONS))
@pytest.mark.parametrize("strategy", STRATEGIES, ids=[s.name for s in STRATEGIES])
async def test_every_strategy_returns_a_valid_distribution(
    backend: VLLMHTTPBackend, strategy: ScoringStrategy, kind: str
) -> None:
    state, question = QUESTIONS[kind]
    result = await strategy.score(backend, renderer(), fresh(state), question)
    assert_valid(result.probabilities, question)
    assert 0.0 <= result.observed_mass <= 1.0
    assert result.n_backend_calls >= 1
    assert result.prompt_tokens > 0


@pytest.mark.parametrize(
    "strategy", [TrieStrategy(), EchoStrategy(), AutoStrategy()], ids=["trie", "echo", "auto"]
)
async def test_sixty_options_score_by_option_text(
    backend: VLLMHTTPBackend, strategy: ScoringStrategy
) -> None:
    result = await strategy.score(
        backend, renderer(), fresh("The Eiffel Tower is in its capital."), COUNTRIES
    )
    assert_valid(result.probabilities, COUNTRIES)
    assert max(result.probabilities, key=result.probabilities.__getitem__) == "France"
    if isinstance(strategy, AutoStrategy):
        assert result.strategy == "echo_sum"
        assert any(w.startswith("auto: ") for w in result.warnings)


async def test_constrained_and_first_token_agree_on_an_easy_question(
    backend: VLLMHTTPBackend,
) -> None:
    state = fresh("I bought a banana at the market.")
    free = await FirstTokenStrategy().score(backend, renderer(), state, EASY)
    forced = await ConstrainedStrategy().score(backend, renderer(), state, EASY)
    argmax = [max(r.probabilities, key=r.probabilities.__getitem__) for r in (free, forced)]
    assert argmax == ["fruit", "fruit"]
    # The mask is reflected and the model writes a label anyway: same distribution up to
    # bf16 prefix-cache noise.
    assert free.probabilities["fruit"] == pytest.approx(forced.probabilities["fruit"], abs=0.05)
