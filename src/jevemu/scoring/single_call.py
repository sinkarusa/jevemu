"""``auto_single``: every question in one model call, every answer label one token.

``auto_noecho`` scores a Choice question with more than :data:`LETTER_LIMIT` options (banking77:
77, CLINC150: 150) with the S3 trie over the option names, several calls per question.
:class:`SingleCallStrategy` labels such options with two-capital codes instead and reads them all
from one S2 constrained first-token call:

- **Labels.** The first ``n`` codes of :data:`~jevemu.render.labels.CODE_LABELS` (``AA``,
  ``AB``, ... in lexicographic order) whose bare and space-prefixed surfaces (``"AB"``,
  ``" AB"``) are each one token for the served model, checked through the backend's tokenizer and
  cached per backend (:func:`single_token_codes`). Qwen3.6 and Gemma 4 have 522 and 588 such
  codes. The prompt lists them like letters (``AB) balance``).
- **Read.** ``structured_outputs.choice`` allows both surfaces of every code. With the mask
  reflected in the logprobs (vLLM 0.30.0) every token the grammar allows ranks above the masked
  ones, so a top-k of :func:`~jevemu.scoring.first_token.grammar_first_tokens` entries (both
  surfaces of every code plus the prefixes the grammar also allows: the space and each first
  letter, bare and spaced) holds every code's exact logprob. Surfaces merge per code and
  renormalize over the codes, as letters do.
- **Server.** That top-k must fit ``--max-logprobs``: 163 for banking77 and 315 for CLINC150 on
  Qwen3.6, at most 2 x 255 + 26 + 26 + 1 = 563 for Jev's 255 options, hence the launch default
  576 (``scripts/serve_vllm.sh``). vLLM's explicit-token logprobs (``logprob_token_ids``) avoid
  the top-k but accept at most 128 ids per request in 0.30.0, fewer than banking77's 154 code
  surfaces.

Every other question (Score, Noul, Choice with up to :data:`LETTER_LIMIT` options) is scored as
``auto_noecho`` scores it: S2 constrained letters, digits or ``Yes``/``No``. Echo is off on both
paths, so a label missing from the top-k gets ``auto_noecho``'s upper bound, never extra calls.
A question one call cannot score raises ``CapabilityError`` instead of falling back.

    Emulator(backend, strategy=SingleCallStrategy())
"""

from __future__ import annotations

from jevemu.backends.base import Backend, Capabilities, CapabilityError
from jevemu.render.labels import (
    CODE_LABELS,
    SURFACE_PREFIXES,
    LabelCapacityError,
    code_scheme,
)
from jevemu.render.renderer import PromptRenderer
from jevemu.scoring.base import ScoreResult
from jevemu.scoring.constrained import ConstrainedStrategy
from jevemu.scoring.first_token import (
    fits_top_k,
    grammar_first_tokens,
    score_first_token,
    single_token_surfaces,
)
from jevemu.scoring.no_echo import NoEchoAutoStrategy, echo_off, without_echo
from jevemu.types import ChoiceQuestion, JSONish, Question

__all__ = ["LETTER_LIMIT", "SingleCallStrategy", "single_token_codes"]

LETTER_LIMIT = 32
"""Most Choice options labelled by letters: what ``auto_noecho`` read in one top-64 call (two
surfaces per letter at ``--max-logprobs 64``). Larger Choice questions get codes."""

_CODE_BATCH = 64
"""Codes checked per round of tokenizer calls while looking for enough single-token codes."""


async def single_token_codes(backend: Backend, n: int) -> tuple[str, ...]:
    """The first ``n`` codes of :data:`~jevemu.render.labels.CODE_LABELS` whose surfaces are
    each one token for ``backend`` (cached per backend).

    Raises ``LabelCapacityError`` when fewer than ``n`` of all the codes qualify.
    """
    found: list[str] = []
    for start in range(0, len(CODE_LABELS), _CODE_BATCH):
        batch = CODE_LABELS[start : start + _CODE_BATCH]
        single = await single_token_surfaces(backend, batch)
        found += [code for code in batch if all(single[p + code] for p in SURFACE_PREFIXES)]
        if len(found) >= n:
            return tuple(found[:n])
    raise LabelCapacityError("codes", n, len(found))


def _code_options(question: Question) -> tuple[str, ...] | None:
    """The option keys of a Choice question labelled by codes; ``None`` for letter questions."""
    if isinstance(question, ChoiceQuestion) and len(question.criteria) > LETTER_LIMIT:
        return tuple(question.criteria)
    return None


def _codes_readable(capabilities: Capabilities) -> bool:
    return (
        capabilities.assistant_prefill
        and capabilities.structured_choice
        and capabilities.mask_reflected_in_logprobs
        and capabilities.local_tokenizer
    )


class SingleCallStrategy:
    """One S2 constrained first-token call per question: letters (as ``auto_noecho``) up to
    :data:`LETTER_LIMIT` Choice options, single-token two-capital codes above.

    Results name the strategy that scored them (``constrained``, or ``first_token`` where
    ``auto_noecho`` would use it) and the label scheme (``codes`` for the code path);
    ``n_backend_calls`` is 1.
    """

    name = "auto_single"

    def __init__(self) -> None:
        self._letters = NoEchoAutoStrategy()

    def supports(self, capabilities: Capabilities, question: Question) -> bool:
        if _code_options(question) is not None:
            return _codes_readable(capabilities)
        return capabilities.assistant_prefill and fits_top_k(capabilities, question)

    async def score(
        self, backend: Backend, renderer: PromptRenderer, state: JSONish, question: Question
    ) -> ScoreResult:
        caps = backend.capabilities
        options = _code_options(question)
        if options is None:
            if not (caps.assistant_prefill and fits_top_k(caps, question)):
                raise CapabilityError(
                    "the question's labels do not fit one first-token read on this backend "
                    f"(top_logprobs_max {caps.top_logprobs_max}, assistant_prefill "
                    f"{caps.assistant_prefill}); auto_single scores in one call or not at all"
                )
            return await self._letters.score(backend, renderer, state, question)
        if not _codes_readable(caps):
            raise CapabilityError(
                "code labels need an assistant prefill, a structured choice whose mask is "
                "reflected in the logprobs, and a local tokenizer"
            )
        view = without_echo(backend)
        scheme = code_scheme(options, await single_token_codes(view, len(options)))
        needed = grammar_first_tokens(scheme.surfaces)
        if needed > caps.top_logprobs_max:
            raise CapabilityError(
                f"{len(options)} code labels need {needed} top logprobs to be read in one call; "
                f"the server allows {caps.top_logprobs_max} (launch vLLM with "
                f"--max-logprobs >= {needed})"
            )
        result = await score_first_token(
            view,
            renderer,
            state,
            question,
            constrained=True,
            name=ConstrainedStrategy.name,
            scheme=scheme,
        )
        return echo_off(result)
