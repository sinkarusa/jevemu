from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx

from jevemu.backends import vllm_http
from jevemu.backends.base import ChatMessage, RenderedPrompt
from jevemu.backends.vllm_http import (
    DEFAULT_CAPABILITIES,
    VLLMError,
    VLLMHTTPBackend,
    VLLMHTTPError,
    build_chat_tokenize_request,
    build_detokenize_request,
    build_echo_request,
    build_first_token_request,
    build_tokenize_request,
    parse_first_token_response,
)

BASE = "http://vllm.test"
PROMPT = RenderedPrompt(
    messages=(
        ChatMessage("system", "Pick one."),
        ChatMessage("user", "Which is a fruit?\nA) Hammer\nB) Apple"),
        ChatMessage("assistant", "Answer:"),
    ),
    template_id="t@1",
)


def _keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, inner in value.items():
            yield key
            yield from _keys(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from _keys(inner)


ALL_BODIES = {
    "first_token": build_first_token_request("m", [1, 2], allowed=None, top_logprobs=5),
    "first_token_constrained": build_first_token_request(
        "m", [1, 2], allowed=["A", "B"], top_logprobs=5
    ),
    "chat_tokenize": build_chat_tokenize_request(
        "m", PROMPT, chat_template_kwargs={"enable_thinking": False}
    ),
    "tokenize": build_tokenize_request("m", " A"),
    "detokenize": build_detokenize_request("m", [1, 2]),
    "echo": build_echo_request("m", [1, 2, 3]),
}


@pytest.mark.parametrize("name", sorted(ALL_BODIES))
def test_no_request_emits_guided_fields(name: str) -> None:
    # vLLM >= 0.12 silently ignores guided_* (unconstrained output with HTTP 200).
    assert not [k for k in _keys(ALL_BODIES[name]) if k.startswith("guided_")]


def test_structured_outputs_sent_top_level_only_when_allowed() -> None:
    free = ALL_BODIES["first_token"]
    forced = ALL_BODIES["first_token_constrained"]
    assert "structured_outputs" not in free
    assert forced["structured_outputs"] == {"choice": ["A", "B"]}
    assert "extra_body" not in forced


@pytest.mark.parametrize("allowed", [[], ["A", ""]])
def test_empty_allowed_labels_are_rejected(allowed: list[str]) -> None:
    with pytest.raises(ValueError, match="allowed"):
        build_first_token_request("m", [1, 2], allowed=allowed, top_logprobs=5)


def _completion_payload(top: dict[int, float], *, tokens: bool = True) -> dict[str, Any]:
    """Completions response keyed by token id (``return_tokens_as_token_ids``)."""
    best = max(top, key=lambda t: top[t])
    logprobs = {
        "tokens": [f"token_id:{best}"],
        "token_logprobs": [top[best]],
        "top_logprobs": [{f"token_id:{i}": v for i, v in top.items()}],
    }
    return {
        "choices": [
            {
                "text": chr(best) if tokens else "",
                "finish_reason": "length" if tokens else "stop",
                "logprobs": logprobs if tokens else None,
            }
        ],
        "usage": {"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": 0}},
    }


def test_response_without_token_logprobs_is_an_error() -> None:
    with pytest.raises(VLLMError, match="no token logprobs"):
        parse_first_token_response(
            _completion_payload({ord("A"): -0.1}, tokens=False),
            constrained=False,
            top_logprobs=5,
            texts={},
        )


BYTE_B = 1066
"""A second token decoding to ``B``, as Gemma's byte token ``<0x42>`` does."""


def _char_detokenize(request: httpx.Request) -> httpx.Response:
    ids = json.loads(request.content)["tokens"]
    return httpx.Response(
        200, json={"prompt": "".join("B" if i == BYTE_B else chr(i) for i in ids)}
    )


def _char_tokenize(template_suffix: str) -> Any:
    """``/tokenize`` with one id per character; chat messages render as ``<role>content``, and
    the generation prompt adds ``template_suffix`` after ``<model>``, as Gemma 4 12B/26B-A4B
    add an empty thought channel only there (a continued assistant message lacks it)."""

    def tokenize(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "messages" in body:
            text = "".join(f"<{m['role']}>{m['content']}" for m in body["messages"])
            if body.get("add_generation_prompt") and not body.get("continue_final_message"):
                text += "<model>" + template_suffix
        else:
            text = body["prompt"]
        return httpx.Response(200, json={"tokens": [ord(c) for c in text]})

    return tokenize


async def test_top_k_is_clamped_to_the_probed_cap() -> None:
    backend = VLLMHTTPBackend(
        BASE, "m", capabilities=dataclasses.replace(DEFAULT_CAPABILITIES, top_logprobs_max=64)
    )
    with respx.mock(base_url=BASE) as router:
        router.post("/tokenize").mock(side_effect=_char_tokenize(""))
        router.post("/detokenize").mock(side_effect=_char_detokenize)
        route = router.post("/v1/completions").respond(
            json=_completion_payload({ord("B"): -0.1, ord("A"): -2.4})
        )
        await backend.next_token_logprobs(PROMPT, allowed=None, top_k=500)
    assert json.loads(route.calls.last.request.content)["logprobs"] == 64


@pytest.fixture
def no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vllm_http, "_RETRY_DELAYS", (0.0, 0.0))


@pytest.mark.usefixtures("no_retry_sleep")
@pytest.mark.parametrize(
    "first", [httpx.Response(503), httpx.ConnectError("refused")], ids=["5xx", "connect"]
)
async def test_transient_failures_are_retried(first: httpx.Response | Exception) -> None:
    backend = VLLMHTTPBackend(BASE, "m")
    with respx.mock(base_url=BASE) as router:
        route = router.post("/tokenize").mock(
            side_effect=[first, httpx.Response(200, json={"tokens": [7, 8]})]
        )
        assert await backend.tokenize("hi") == [7, 8]
    assert route.call_count == 2


@pytest.mark.usefixtures("no_retry_sleep")
async def test_client_errors_are_not_retried() -> None:
    backend = VLLMHTTPBackend(BASE, "m")
    with respx.mock(base_url=BASE) as router:
        route = router.post("/tokenize").respond(
            400, json={"error": {"message": "bad input", "code": 400}}
        )
        with pytest.raises(VLLMHTTPError) as info:
            await backend.tokenize("hi")
    assert (info.value.status_code, info.value.detail) == (400, "bad input")
    assert route.call_count == 1


@pytest.mark.usefixtures("no_retry_sleep")
async def test_persistent_server_errors_surface_after_retries() -> None:
    backend = VLLMHTTPBackend(BASE, "m")
    with respx.mock(base_url=BASE) as router:
        route = router.post("/tokenize").respond(503)
        with pytest.raises(VLLMHTTPError) as info:
            await backend.tokenize("hi")
    assert info.value.status_code == 503
    assert route.call_count == len(vllm_http._RETRY_DELAYS) + 1


async def test_continuation_that_merges_with_the_prefill_keeps_the_prefix_exact() -> None:
    # Joint tokenization "Answer: A" re-tokenizes the prefill's last token; the adapter must
    # keep the prefix ids and append the continuation's own tokens, scoring only those.
    head, prefill, merged_joint, standalone = [1], [2, 3], [2, 9], [7]
    backend = VLLMHTTPBackend(BASE, "m")

    def tokenize(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "messages" in body:
            return httpx.Response(200, json={"tokens": head})
        tokens = {"Answer:": prefill, "Answer: A": merged_joint, " A": standalone}
        return httpx.Response(200, json={"tokens": tokens[body["prompt"]]})

    logprobs = {"tokens": ["<s>", "Answer", ":", " A"], "token_logprobs": [None, -1.0, -0.5, -0.25]}
    echo = {"choices": [{"logprobs": logprobs}]}
    with respx.mock(base_url=BASE) as router:
        router.post("/tokenize").mock(side_effect=tokenize)
        completions = router.post("/v1/completions").respond(json=echo)
        (score,) = await backend.sequence_logprobs(PROMPT, [" A"])
    assert json.loads(completions.calls.last.request.content)["prompt"] == [1, 2, 3, 7]
    assert (score.tokens, score.token_logprobs) == ((" A",), (-0.25,))


@pytest.mark.parametrize("prefill", ["Answer:", "Answer: "])
async def test_the_model_reads_the_prefill_after_the_generation_prompt(prefill: str) -> None:
    # The prefill must follow what the template emits only on the generation prompt (Gemma 4
    # 12B/26B-A4B: an empty thought channel, missing from a continued assistant message), and a
    # trailing space (the "Answer: " digit prefill) must reach the model.
    backend = VLLMHTTPBackend(
        BASE, "m", capabilities=dataclasses.replace(DEFAULT_CAPABILITIES, top_logprobs_max=64)
    )
    prompt = dataclasses.replace(
        PROMPT, messages=(*PROMPT.messages[:-1], ChatMessage("assistant", prefill))
    )
    top = {ord("3"): -0.01, ord("2"): -4.7, ord("4"): -9999.0}
    with respx.mock(base_url=BASE) as router:  # no chat route: a chat call fails the test
        router.post("/tokenize").mock(side_effect=_char_tokenize("<thought/>"))
        router.post("/detokenize").mock(side_effect=_char_detokenize)
        completions = router.post("/v1/completions").respond(json=_completion_payload(top))
        dist = await backend.next_token_logprobs(prompt, allowed=["2", "3", "4"], top_k=2)
    body = json.loads(completions.calls.last.request.content)
    assert "".join(map(chr, body["prompt"])) == (
        "<system>Pick one.<user>Which is a fruit?\nA) Hammer\nB) Apple<model><thought/>" + prefill
    )
    assert body["structured_outputs"] == {"choice": ["2", "3", "4"]}
    assert [(t.token, t.logprob) for t in dist.top] == [("3", -0.01), ("2", -4.7)]
    assert (dist.sampled.token, dist.constrained, dist.prompt_tokens) == ("3", True, 12)


async def test_tokens_that_decode_alike_are_all_read() -> None:
    # Gemma's vocabulary holds "B" twice (the byte token <0x42> too); both may be allowed by a
    # choice constraint. Keyed by text, the completions top-k map keeps only the last (least
    # likely) of them, which read the label as nearly impossible.
    backend = VLLMHTTPBackend(BASE, "m")
    top = {ord("B"): -0.02, ord("A"): -4.0, BYTE_B: -19.75}
    with respx.mock(base_url=BASE) as router:
        router.post("/tokenize").mock(side_effect=_char_tokenize(""))
        router.post("/detokenize").mock(side_effect=_char_detokenize)
        router.post("/v1/completions").respond(json=_completion_payload(top))
        dist = await backend.next_token_logprobs(PROMPT, allowed=["A", "B"], top_k=5)
    assert [(t.token, t.logprob) for t in dist.top] == [("B", -0.02), ("A", -4.0), ("B", -19.75)]
