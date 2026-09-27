"""End-to-end checks against a live vLLM server (``scripts/serve_vllm.sh``).

Skipped unless ``JEVEMU_VLLM_URL`` is set; ``JEVEMU_VLLM_MODEL`` (default Qwen/Qwen3-0.6B) and
``JEVEMU_VLLM_MAX_LOGPROBS`` (default 576) must match the server launch.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from jevemu.backends import probe
from jevemu.backends.base import ChatMessage, RenderedPrompt
from jevemu.backends.probe import ServerFlags, probe_capabilities
from jevemu.backends.vllm_http import DEFAULT_CAPABILITIES, VLLMHTTPBackend

URL = os.environ.get("JEVEMU_VLLM_URL", "")
MODEL = os.environ.get("JEVEMU_VLLM_MODEL", "Qwen/Qwen3-0.6B")
MAX_LOGPROBS = int(os.environ.get("JEVEMU_VLLM_MAX_LOGPROBS", "576"))

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not URL, reason="set JEVEMU_VLLM_URL to run live vLLM tests"),
]

MCQ = RenderedPrompt(
    messages=(
        ChatMessage("system", "Answer with the letter of the correct option."),
        ChatMessage("user", "Which of these is a fruit?\nA) Hammer\nB) Apple\nC) Brick"),
        ChatMessage("assistant", "Answer:"),
    ),
    template_id="live/mcq",
)


def capital_prompt() -> RenderedPrompt:
    """A prefix never seen before, so the first request cannot hit the prefix cache."""
    return RenderedPrompt(
        messages=(
            ChatMessage("system", f"Session {uuid.uuid4().hex}."),
            ChatMessage("user", "What is the capital of France?"),
            ChatMessage("assistant", "The capital of France is"),
        ),
        template_id="live/capital",
    )


def make_backend() -> VLLMHTTPBackend:
    caps = dataclasses.replace(DEFAULT_CAPABILITIES, top_logprobs_max=MAX_LOGPROBS)
    return VLLMHTTPBackend(URL, MODEL, capabilities=caps)


@pytest.fixture
async def backend() -> AsyncIterator[VLLMHTTPBackend]:
    async with make_backend() as live:
        yield live


async def test_constraint_forces_an_answer_the_model_would_never_give(
    backend: VLLMHTTPBackend,
) -> None:
    free = await backend.next_token_logprobs(MCQ, allowed=None, top_k=5)
    forced = await backend.next_token_logprobs(MCQ, allowed=["Q", "Z"], top_k=5)
    assert free.sampled.token.strip() not in {"Q", "Z"}
    assert forced.sampled.token in {"Q", "Z"}


async def test_echo_matches_first_token_logprob_and_sums_tokens(
    backend: VLLMHTTPBackend,
) -> None:
    prompt = capital_prompt()
    dist = await backend.next_token_logprobs(prompt, allowed=None, top_k=20)
    paris, london, longer = await backend.sequence_logprobs(
        prompt, [" Paris", " London", " Paris, France."]
    )
    first_token = {t.token: t.logprob for t in dist.top}
    # The echo logprob of the first continuation token is the first-token logprob of that
    # token. Batch composition moves logprobs by ~0.1-0.2 nats on ordinary tokens (probe
    # report), far less than the gap to a neighboring position, which a slicing error would
    # pick up.
    noise = 0.2
    assert paris.tokens == (" Paris",)
    assert paris.total == pytest.approx(first_token[" Paris"], abs=noise)
    assert paris.total > london.total
    assert longer.tokens[0] == " Paris"
    assert longer.token_logprobs[0] == pytest.approx(paris.total, abs=noise)
    assert len(longer.tokens) > 1
    assert longer.total < paris.total + noise


async def test_prefill_is_continued_not_closed(backend: VLLMHTTPBackend) -> None:
    prompt = capital_prompt()
    ids = await backend.tokenize_chat(prompt)
    rendered = await backend.detokenize(ids)
    dist = await backend.next_token_logprobs(prompt, allowed=None, top_k=20)
    assert rendered.endswith(prompt.prefill)
    assert dist.prompt_tokens == len(ids)
    assert " Paris" in {t.token for t in dist.top}


async def test_prefill_trailing_space_reaches_the_model(backend: VLLMHTTPBackend) -> None:
    # Score digits are read after "Answer: "; Qwen3.5/3.8 chat templates trim that space from
    # the final assistant message, so the model would read the digit right after "Answer:".
    spaced = dataclasses.replace(
        MCQ, messages=(*MCQ.messages[:-1], ChatMessage("assistant", "Answer: "))
    )
    ids = await backend.tokenize_chat(spaced)
    dist = await backend.next_token_logprobs(spaced, allowed=None, top_k=5)
    assert (await backend.detokenize(ids)).endswith("\nAnswer: ")
    assert len(ids) == len(await backend.tokenize_chat(MCQ)) + len(await backend.tokenize(" "))
    assert dist.prompt_tokens == len(ids)


async def test_probe_measures_the_capabilities_jevemu_relies_on(
    backend: VLLMHTTPBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flags = ServerFlags(max_logprobs=MAX_LOGPROBS)
    caps = await probe_capabilities(backend, flags, cache_dir=tmp_path)
    assert caps.structured_choice
    assert caps.echo_prompt_logprobs
    assert caps.assistant_prefill
    assert caps.top_logprobs_max == MAX_LOGPROBS

    async def no_probe(*_: object) -> None:
        raise AssertionError("expected a probe cache hit")

    monkeypatch.setattr(probe, "run_probe", no_probe)
    assert await probe_capabilities(backend, flags, cache_dir=tmp_path) == caps


def test_backend_survives_separate_event_loops() -> None:
    # Emulator.system_one_sync-style usage: one backend, one asyncio.run per call.
    backend = make_backend()
    first = asyncio.run(backend.tokenize("Answer: A"))
    second = asyncio.run(backend.tokenize("Answer: A"))
    assert first == second
