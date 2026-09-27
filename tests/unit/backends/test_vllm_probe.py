"""Probe decisions against a scripted vLLM server (the live server shows only one behavior)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from jevemu.backends.probe import (
    ConstraintNotEnforced,
    ServerFlags,
    format_report,
    probe_capabilities,
    run_probe,
)
from jevemu.backends.vllm_http import VLLMHTTPBackend

BASE = "http://vllm.test"

# Unconstrained first-token table: the model prefers markdown over a bare label.
RAW = {" **": -0.3, " B": -1.4, " A": -10.3, " Paris": -2.9, "A": -12.0, "B": -11.0}


class FakeVLLM:
    """Just enough of vLLM's OpenAI server for the probe; ids are code points. Chat messages
    render as ``<role>content`` (``<model>`` for the assistant); the generation prompt adds
    ``<model>`` + ``generation_suffix``."""

    def __init__(
        self,
        *,
        cap: int = 64,
        mask_reflected: bool = True,
        enforce: bool = True,
        truncate_at: int | None = None,
        generation_suffix: str = "",
    ) -> None:
        self.cap = cap
        self.mask_reflected = mask_reflected
        self.enforce = enforce
        self.truncate_at = truncate_at
        self.generation_suffix = generation_suffix
        self.first_token_calls = 0
        self._words: list[str] = []
        """Multi-character tokens, with ids from 0x110000 up (single characters: code points)."""

    def _id(self, token: str) -> int:
        if len(token) == 1:
            return ord(token)
        if token not in self._words:
            self._words.append(token)
        return 0x110000 + self._words.index(token)

    def _text(self, token_id: int) -> str:
        return chr(token_id) if token_id < 0x110000 else self._words[token_id - 0x110000]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        if path == "/version":
            return httpx.Response(200, json={"version": "0.30.0"})
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m", "max_model_len": 8192}]})
        if path == "/metrics":
            return httpx.Response(
                200, text='vllm:cache_config_info{enable_prefix_caching="True"} 1\n'
            )
        if path == "/tokenize":
            return httpx.Response(200, json={"tokens": [ord(c) for c in self._render(body)]})
        if path == "/detokenize":
            return httpx.Response(200, json={"prompt": "".join(map(self._text, body["tokens"]))})
        assert path == "/v1/completions"
        if body.get("echo"):
            tokens = [chr(i) for i in body["prompt"]]
            values = [None] + [-1.0] * (len(tokens) - 1)
            logprobs = {"tokens": tokens, "token_logprobs": values}
            return httpx.Response(200, json={"choices": [{"logprobs": logprobs}]})
        return self._first_token(body)

    def _render(self, body: dict[str, Any]) -> str:
        if "messages" not in body:
            return str(body["prompt"])
        text = "".join(
            f"<{'model' if m['role'] == 'assistant' else m['role']}>{m['content']}"
            for m in body["messages"]
        )
        if body.get("add_generation_prompt") and not body.get("continue_final_message"):
            text += "<model>" + self.generation_suffix
        return text

    def _first_token(self, body: dict[str, Any]) -> httpx.Response:
        self.first_token_calls += 1
        n = body["logprobs"]
        if n > self.cap:
            message = f"Requested sample logprobs of {n}, which is greater than max allowed"
            return httpx.Response(400, json={"error": {"message": message}})
        table = dict(RAW)
        allowed = (body.get("structured_outputs") or {}).get("choice")
        sampled = " **"
        if allowed and self.enforce:
            sampled = allowed[-1]
            if self.mask_reflected:
                logz = math.log(sum(math.exp(RAW.get(a, -20.0)) for a in allowed))
                table = {t: (RAW.get(t, -20.0) - logz if t in allowed else -9999.0) for t in RAW}
                table.update({a: RAW.get(a, -20.0) - logz for a in allowed})
        ranked = sorted(table.items(), key=lambda kv: -kv[1])
        ranked += [(f"<pad{i}>", -9999.0 if allowed else -30.0 - i) for i in range(n)]
        if "logprob_token_ids" in body:
            wanted = {self._text(i) for i in body["logprob_token_ids"]}
            top = [(t, lp) for t, lp in ranked if t in wanted or t == sampled]
        else:
            top = ranked[: min(n, self.truncate_at or n)]
        key = (
            (lambda t: f"token_id:{self._id(t)}")
            if body.get("return_tokens_as_token_ids")
            else (lambda t: t)
        )
        logprobs = {
            "tokens": [key(sampled)],
            "token_logprobs": [table.get(sampled, -1.0)],
            "top_logprobs": [{key(t): lp for t, lp in top}],
        }
        usage = {
            "prompt_tokens": len(body["prompt"]),
            "prompt_tokens_details": {"cached_tokens": 32},
        }
        return httpx.Response(
            200, json={"choices": [{"text": sampled, "logprobs": logprobs}], "usage": usage}
        )


async def probe(server: FakeVLLM, flags: ServerFlags | None = None) -> Any:
    with respx.mock(base_url=BASE) as router:
        router.route().mock(side_effect=server)
        return await run_probe(VLLMHTTPBackend(BASE, "m"), flags or ServerFlags(max_logprobs=64))


@pytest.mark.parametrize("reflected", [True, False])
async def test_mask_reflection_is_measured_not_assumed(reflected: bool) -> None:
    report = await probe(FakeVLLM(mask_reflected=reflected))
    assert report.mask.reflected is reflected
    assert report.capabilities.mask_reflected_in_logprobs is reflected
    assert report.capabilities.structured_choice
    if not reflected:
        assert " **" in report.mask.invalid_finite


async def test_cap_above_declared_limit_is_an_error_and_cap_is_kept() -> None:
    report = await probe(FakeVLLM(cap=64))
    assert report.max_logprobs.above_cap_behavior == "error"
    assert report.capabilities.top_logprobs_max == 64


async def test_silent_truncation_lowers_the_effective_cap() -> None:
    report = await probe(FakeVLLM(cap=1000, truncate_at=50))
    assert report.max_logprobs.above_cap_behavior == "truncated"
    assert report.capabilities.top_logprobs_max == 50


async def test_overstated_cap_is_searched_down_to_the_server_limit() -> None:
    report = await probe(FakeVLLM(cap=40), ServerFlags(max_logprobs=64))
    assert report.max_logprobs.at_cap.status == 400
    assert report.capabilities.top_logprobs_max == 40


@pytest.mark.parametrize("suffix", ["", "<thought/>"], ids=["same", "generation-only-channel"])
async def test_prefill_render_divergence_is_recorded_not_fatal(suffix: str) -> None:
    # Gemma 4 12B/26B-A4B emit an empty thought channel only on the generation prompt, so the
    # template's continued-assistant rendering of the prefill differs from the adapter's prompt.
    report = await probe(FakeVLLM(generation_suffix=suffix))
    assert report.prefill.rendered_tail.endswith("<model>" + suffix + "The capital of France is")
    assert report.prefill.continued_matches is (suffix == "")
    assert report.capabilities.assistant_prefill
    assert ("renders differently" in format_report(report)) is bool(suffix)


async def test_ignored_constraint_raises(tmp_path: Path) -> None:
    with respx.mock(base_url=BASE) as router:
        router.route().mock(side_effect=FakeVLLM(enforce=False))
        with pytest.raises(ConstraintNotEnforced, match="ignored structured_outputs"):
            await probe_capabilities(
                VLLMHTTPBackend(BASE, "m"), ServerFlags(max_logprobs=64), cache_dir=tmp_path
            )


@pytest.mark.parametrize(
    "changed",
    [
        {"image": "img@sha256:2"},
        {"server_chat_template_kwargs": {"enable_thinking": False}},
        {"chat_template_kwargs": {"enable_thinking": False}},
    ],
    ids=["image", "server-template-kwargs", "request-template-kwargs"],
)
async def test_cached_report_is_reused_until_server_identity_changes(
    tmp_path: Path, changed: dict[str, Any]
) -> None:
    server = FakeVLLM()
    same: dict[str, Any] = {"image": "img@sha256:1", "server_chat_template_kwargs": {}}
    with respx.mock(base_url=BASE) as router:
        router.route().mock(side_effect=server)
        backend = VLLMHTTPBackend(BASE, "m", **same)
        first = await probe_capabilities(backend, ServerFlags(max_logprobs=64), cache_dir=tmp_path)
        calls = server.first_token_calls
        again = await probe_capabilities(backend, ServerFlags(max_logprobs=64), cache_dir=tmp_path)
        assert (again, server.first_token_calls) == (first, calls)
        other = VLLMHTTPBackend(BASE, "m", **{**same, **changed})
        await probe_capabilities(other, ServerFlags(max_logprobs=64), cache_dir=tmp_path)
        assert server.first_token_calls > calls
