"""S2 constrained first-token strategy: S1 plus a constraint over the labels.

On a vLLM backend the constraint is ``structured_outputs.choice`` over the label surfaces, read
after the ``Answer:`` prefill. On a hosted chat API (no ``assistant_prefill``) it is a Structured
Outputs JSON schema whose value is an ``enum`` of the bare labels, read where that value starts
(:func:`~jevemu.backends.chat_api.parse_structured_response`).

When the backend reflects the grammar mask in its logprobs (vLLM 0.30.0, OpenAI and Wafer do),
the allowed tokens' logprobs are already renormalized over the allowed set, so the distribution
over the labels is exact. Otherwise the reported logprobs are unconstrained and S2 reads like S1,
but the generated token is guaranteed to be a label.

**Which surfaces are allowed, and where they are read ("Mind the Gap", EMNLP 2025).** After the
``Answer:`` prefill the model writes the label with its leading space. The constraint allows
every single-token surface of each label, at the reading position of
:func:`~jevemu.scoring.first_token.reading_plan`:

- Letters and ``Yes``/``No``: both ``"A"`` and ``" A"``, read right after ``Answer:``.
- Digits: ``" 7"`` is two tokens on Qwen3 (``" "`` + ``"7"``), so the prefill becomes
  ``Answer: `` and the bare digits are allowed there.

Live check (2026-09-24; vLLM 0.30.0, Qwen/Qwen3-0.6B bf16, ``question_first`` layout unless
noted; probabilities renormalized over the labels):

- Choice "I bought a banana" (fruit/tool/vehicle): unconstrained ``" A"`` 0.980, ``" B"``
  0.020; constrained to both variants the same (bare variants ~0, the grammar-prefix token
  ``" "`` ~0); constrained to bare letters only ``"A"`` 0.932, ``"B"`` 0.068. Forcing the bare
  form moves mass, so the spaced variants must be allowed.
- Noul "talk to a human": unconstrained ``" Yes"`` 0.967; both variants 0.967; bare only 0.992.
  Extending the prefill with a space breaks single-token labels: after ``Answer: `` only 0.22
  of the mass is on ``Yes``/``No`` (the model writes digits).
- Score "package destroyed" (5 levels): after ``Answer:`` the unconstrained model puts 0.9998 on
  the space token, and allowing ``" 3"`` hands 0.9997 to ``" "``. Bare digits forced right
  after ``Answer:`` read ``3`` 0.976 / ``4`` 0.023, but the model's own path through the space
  gives ``3`` 0.755 / ``4`` 0.245 after ``Answer: `` and 0.798 / 0.202 by echo of ``" 3"``.
  Over 10 score prompts (5 questions x 2 layouts) reading after the gap matched echo within
  0.06 per level, while the bare read after ``Answer:`` was off by up to 0.58 (review 0-9,
  ``state_first``: ``1`` 0.233 vs 0.813 by echo).
"""

from __future__ import annotations

from jevemu.backends.base import Backend, Capabilities
from jevemu.render.renderer import PromptRenderer
from jevemu.scoring.base import ScoreResult
from jevemu.scoring.first_token import fits_top_k, score_first_token
from jevemu.types import JSONish, Question

__all__ = ["ConstrainedStrategy"]


class ConstrainedStrategy:
    """S2: first-token logprobs under a constraint over the label surfaces."""

    name = "constrained"

    def supports(self, capabilities: Capabilities, question: Question) -> bool:
        return capabilities.structured_choice and fits_top_k(capabilities, question)

    async def score(
        self, backend: Backend, renderer: PromptRenderer, state: JSONish, question: Question
    ) -> ScoreResult:
        return await score_first_token(
            backend, renderer, state, question, constrained=True, name=self.name
        )
