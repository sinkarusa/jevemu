"""Backend contract: raw token logprobs plus measured capabilities.

Backends do no probability maths. They return raw token logprobs; renormalization, variant
merging, debiasing and calibration happen client-side in ``jevemu.scoring`` and friends.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant"]

PRICE_ID_FLAG = "price_id"
"""``BackendInfo.flags`` key a paid-API backend sets to its ``jevemu.eval.costs`` price id."""


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class RenderedPrompt:
    """A chat prompt ending in the assistant prefill (e.g. ``"Answer:"``) or, for backends
    without ``assistant_prefill``, in the user turn.

    With a prefill, backends must continue the final assistant message, not open a new turn.
    Without one, the answer is the first token of a new assistant turn.
    """

    messages: tuple[ChatMessage, ...]
    template_id: str
    """Template name plus content hash; keys calibrators and caches."""

    def __post_init__(self) -> None:
        if not self.messages or self.messages[-1].role not in ("assistant", "user"):
            raise ValueError("RenderedPrompt must end with an assistant prefill or a user message")

    @property
    def has_prefill(self) -> bool:
        return self.messages[-1].role == "assistant"

    @property
    def prefill(self) -> str:
        """The assistant prefill; ``ValueError`` for a prompt that ends in the user turn."""
        if not self.has_prefill:
            raise ValueError("this prompt has no assistant prefill (it ends in the user turn)")
        return self.messages[-1].content


@dataclass(frozen=True)
class TokenLogprob:
    token: str
    """Decoded token text, including any leading space (``" A"``)."""
    token_id: int | None
    logprob: float
    """Natural-log probability; ``-inf`` for floored or impossible tokens."""


@dataclass(frozen=True)
class ApiCall:
    """Accounting of one billed API request (``None`` on ``NextTokenDist`` for local servers)."""

    latency_ms: float
    """HTTP round trip of the attempt that produced the response; for a response-cache hit,
    the one recorded when it was first fetched."""
    cost_usd: float
    """Spend incurred by this call: 0 for a response-cache hit."""
    response_cached: bool
    """Served from the local response cache (no network call)."""
    completion_tokens: int
    """Billed output tokens, reasoning tokens included."""
    cache_write_tokens: int | None
    """Prompt tokens written to the provider's prompt cache; ``None`` if not reported."""
    model: str
    """Model id the response reports."""
    system_fingerprint: str | None
    provider: str | None = None
    """Serving provider behind a routing API (OpenRouter); ``None`` if not reported."""


@dataclass(frozen=True)
class NextTokenDist:
    """Distribution over the first generated token after the prefill (or the user turn)."""

    top: tuple[TokenLogprob, ...]
    """Top-k alternatives, sorted by descending logprob, length <= requested ``top_k``."""
    sampled: TokenLogprob
    """The token actually generated (greedy)."""
    constrained: bool
    """Whether an ``allowed`` constraint was sent with the request."""
    prompt_tokens: int
    cached_tokens: int | None = None
    api: ApiCall | None = None
    """Billing and latency of a paid API call; ``None`` for local servers."""


@dataclass(frozen=True)
class SeqScore:
    """Logprob of a continuation given a prefix (echo / prompt-logprob scoring)."""

    continuation: str
    tokens: tuple[str, ...]
    token_logprobs: tuple[float, ...]
    """One per continuation token; vLLM's -9999 floor is mapped to ``-inf``."""
    prompt_tokens: int | None = None
    """Tokens the backend processed for this request (prefix + continuation), when it reports
    them (vLLM's ``usage.prompt_tokens``); ``None`` when unknown."""

    @property
    def total(self) -> float:
        return math.fsum(self.token_logprobs)


@dataclass(frozen=True)
class BackendInfo:
    backend: str
    """Adapter name, e.g. ``"vllm_http"``."""
    model: str
    model_revision: str | None
    engine_version: str
    """e.g. the vLLM version string."""
    flags: Mapping[str, str] = field(default_factory=dict)
    """Server flags relevant to scoring (max-logprobs, logprobs-mode, dtype, ...)."""
    price: Mapping[str, Any] | None = None
    """Per-token price snapshot of a paid API whose price is not built into
    ``jevemu.eval.costs.TOKEN_PRICES`` (``TokenPrice`` fields; OpenRouter)."""


@dataclass(frozen=True)
class Capabilities:
    top_logprobs_max: int
    """From ``--max-logprobs``."""
    structured_choice: bool
    """``next_token_logprobs`` honors ``allowed``: vLLM's ``structured_outputs.choice`` over the
    surfaces, or on a hosted chat API a Structured Outputs JSON ``enum`` over the labels read
    where the value starts."""
    logprobs_mode: Literal["raw_logprobs", "processed_logprobs"]
    mask_reflected_in_logprobs: bool
    """Measured by the capability probe, never assumed."""
    echo_prompt_logprobs: bool
    explicit_token_logprobs: bool
    assistant_prefill: bool
    prefix_caching: bool
    local_tokenizer: bool = True
    """``tokenize``/``detokenize`` work, so label surfaces can be checked for being one token.
    Without it every surface of a label is read as-is (hosted APIs with no public tokenizer)."""
    top_logprobs_exact: bool = True
    """The top-k holds exactly the requested count (vLLM). False when the API may return fewer,
    dropping alternatives below a probability threshold (gpt-6-luna); the missing-label bound
    then also uses the leftover mass."""


class CapabilityError(NotImplementedError):
    """The backend does not support the requested operation (per its ``Capabilities``)."""


class Backend(Protocol):
    capabilities: Capabilities

    async def next_token_logprobs(
        self, prompt: RenderedPrompt, *, allowed: Sequence[str] | None, top_k: int
    ) -> NextTokenDist:
        """First-token logprobs after the prefill; ``allowed`` constrains decoding if given."""
        ...

    async def sequence_logprobs(
        self, prefix: RenderedPrompt, continuations: Sequence[str]
    ) -> list[SeqScore]:
        """Score each continuation appended directly after the prefill, in input order."""
        ...

    async def tokenize(self, text: str) -> list[int]: ...

    async def detokenize(self, ids: Sequence[int]) -> str: ...

    async def health(self) -> BackendInfo: ...
