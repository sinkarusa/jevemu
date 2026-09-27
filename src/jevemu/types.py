"""Pydantic models mirroring the Jev System One wire format (``POST /v1/systemone``).

Requests are validated strictly (unknown fields rejected) so a payload that passes here is
one Jev would accept. Responses tolerate unknown fields (kept in ``model_extra``) so a newer
Jev response still parses. Emulator-only diagnostics live under ``x_jevemu``, which Jev
clients ignore.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

DEFAULT_MODEL = "jev-1.13.0"

CHOICE_MIN_OPTIONS = 2
CHOICE_MAX_OPTIONS = 255
SCORE_MIN_LEVELS = 2
SCORE_MAX_LEVELS = 10

JSONish = str | dict[str, Any] | list[Any]
"""A Jev text-or-structure field: plain string, JSON object, or JSON array."""

Description = JSONish | None
"""Instructions, option descriptions and level descriptions also accept null (docs:
primitives/advanced "Where you can use JSON")."""

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Response(BaseModel):
    model_config = ConfigDict(extra="allow")


# --- questions -------------------------------------------------------------------------------


class NoulQuestion(_Request):
    """Yes/no question; the answer is P(yes)."""

    type: Literal["noul"] = "noul"
    instructions: Description
    criteria: dict[Literal["true", "false"], Description] | None = None


class ChoiceQuestion(_Request):
    """Pick one option from ``criteria`` (option -> description or null), order preserved."""

    type: Literal["choice"] = "choice"
    instructions: Description
    criteria: Annotated[
        dict[Annotated[str, Field(min_length=1)], Description],
        Field(min_length=CHOICE_MIN_OPTIONS, max_length=CHOICE_MAX_OPTIONS),
    ]


class ScoreQuestion(_Request):
    """Rate against ordered levels; level ``i`` has value ``i``."""

    type: Literal["score"] = "score"
    instructions: Description
    criteria: Annotated[
        list[Description], Field(min_length=SCORE_MIN_LEVELS, max_length=SCORE_MAX_LEVELS)
    ]


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class SystemOneRequest(_Request):
    """One state evaluated against independent, caller-named questions."""

    state: JSONish
    model: str = DEFAULT_MODEL
    questions: Annotated[dict[str, Question], Field(min_length=1)]


# --- answers ---------------------------------------------------------------------------------


class NoulAnswer(_Response):
    type: Literal["noul"] = "noul"
    noul: Probability


class ChoiceAnswer(_Response):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, Probability]
    confidence: Probability

    @model_validator(mode="after")
    def _choice_is_an_option(self) -> ChoiceAnswer:
        if self.choice not in self.probabilities:
            raise ValueError(f"choice {self.choice!r} is not a key of probabilities")
        return self


class ScoreAnswer(_Response):
    type: Literal["score"] = "score"
    score: float
    legend: Annotated[
        dict[str, Description], Field(min_length=SCORE_MIN_LEVELS, max_length=SCORE_MAX_LEVELS)
    ]
    probabilities: dict[str, Probability]
    confidence: Probability

    @model_validator(mode="after")
    def _levels_consistent(self) -> ScoreAnswer:
        levels = [str(i) for i in range(len(self.legend))]
        if list(self.legend) != levels:
            raise ValueError(f"legend keys must be '0'..'{len(levels) - 1}' in order")
        if set(self.probabilities) != set(levels):
            raise ValueError("probabilities keys must match legend keys")
        if not 0.0 <= self.score <= len(levels) - 1:
            raise ValueError(f"score {self.score} outside [0, {len(levels) - 1}]")
        return self


Answer = Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]


CACHED_INPUT_FIELD = "cached_input_tokens"
"""Extra ``Usage`` field: input tokens served from the provider's prompt cache."""
CACHE_WRITE_FIELD = "cache_write_tokens"
"""Extra ``Usage`` field: input tokens written to the provider's prompt cache."""


class Usage(_Response):
    """Tokens of one response. ``input_tokens`` counts every prompt token; paid-API emulator
    backends add the extra fields :data:`CACHED_INPUT_FIELD` and :data:`CACHE_WRITE_FIELD`."""

    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]


class PermutationScore(BaseModel):
    """One presentation of a Choice question scored by a debiaser (``x_jevemu`` only)."""

    model_config = ConfigDict(extra="forbid")

    order: list[str]
    """The original option keys in the order they were presented (position 0 first)."""
    probabilities: dict[str, Probability]
    """The strategy's probabilities for that presentation, keyed by original option key."""


class ApiUsage(BaseModel):
    """Paid-API accounting of one question's backend calls (``x_jevemu`` only)."""

    model_config = ConfigDict(extra="forbid")

    calls: Annotated[int, Field(ge=0)]
    response_cache_hits: Annotated[int, Field(ge=0)]
    """Calls served from the local response cache (no spend)."""
    latency_ms: Annotated[float, Field(ge=0.0)]
    """Summed HTTP round trips of the calls (recorded ones for cache hits)."""
    cost_usd: Annotated[float, Field(ge=0.0)]
    """Spend incurred by these calls: 0 for cache hits."""
    models: list[str]
    """Distinct model ids the responses reported."""
    system_fingerprints: list[str]
    """Distinct ``system_fingerprint`` values the responses reported."""
    providers: list[str] = Field(default_factory=list)
    """Distinct serving providers the responses reported (routing APIs: OpenRouter)."""


class EmulatorDiagnostics(BaseModel):
    """Per-question emulator internals, reported under ``x_jevemu``."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    backend_model: str
    vllm_version: str
    strategy: str
    label_scheme: str
    permutations: Annotated[int, Field(ge=1)] = 1
    """Presentations of the question that were scored (1 without a permuting debiaser)."""
    observed_mass: Probability
    """Sum of valid-label probability before renormalization."""
    missing_labels: list[str] = Field(default_factory=list)
    raw_probabilities: dict[str, Probability]
    """Probabilities before debiasing and calibration (the question's own option order)."""
    debiaser: str | None = None
    """``Debiaser.id`` (e.g. ``"pride(alpha=0.05,...)"``); ``None`` without a debiaser."""
    debias_calls: Annotated[int, Field(ge=0)] = 0
    """Backend calls the debiaser added; included in ``n_backend_calls``."""
    debias_prior: dict[str, Probability] | None = None
    """The label prior the debiaser divided out, over the options it applies to."""
    permutation_scores: list[PermutationScore] = Field(default_factory=list)
    """Every presentation a permuting debiaser scored, the question's own order first."""
    calibrator: str | None = None
    n_backend_calls: Annotated[int, Field(ge=0)]
    latency_ms: Annotated[float, Field(ge=0.0)]
    cached_tokens: Annotated[int, Field(ge=0)] | None = None
    warnings: list[str] = Field(default_factory=list)
    api: ApiUsage | None = None
    """Paid-API backends only; omitted from the serialized form when ``None``."""

    @model_serializer(mode="wrap")
    def _omit_absent_api(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if data.get("api") is None:
            data.pop("api", None)
        return data


class SystemOneResponse(_Response):
    model: str
    answers: dict[str, Answer]
    usage: Usage
    x_jevemu: dict[str, EmulatorDiagnostics] | None = None


class SystemOneClient(Protocol):
    """Anything that answers Jev requests: the real Jev client or the emulator."""

    async def system_one(self, request: SystemOneRequest) -> SystemOneResponse: ...
