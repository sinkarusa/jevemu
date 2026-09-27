"""Record golden HTTP fixtures for ``VLLMHTTPBackend`` from a live vLLM server.

    JEVEMU_VLLM_MAX_LOGPROBS=64 scripts/serve_vllm.sh --preset qwen3-0.6b
    uv run python scripts/record_vllm_fixtures.py --url http://localhost:8000

The server must cap logprobs at 64: ``top_logprobs_above_cap_error`` requests 65.

Each fixture in ``tests/golden/vllm/`` holds one backend call (``call``), every HTTP exchange it
made (``exchanges``), and the server identity (``meta``: vLLM version, model, HF revision,
image digest). ``tests/golden/test_vllm_http_replay.py`` replays them with respx.

After re-recording, the drift test in that module fails if any response shape changed; review
the change, then refresh its snapshot with ``JEVEMU_UPDATE_VLLM_SCHEMA=1``.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from jevemu.backends.base import RenderedPrompt
from jevemu.backends.probe import MCQ_PROMPT, PREFILL_PROMPT
from jevemu.backends.vllm_http import (
    DEFAULT_CAPABILITIES,
    IMAGE_ENV,
    METRICS_PATH,
    MODEL_REVISION_ENV,
    VLLMHTTPBackend,
    VLLMHTTPError,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "golden" / "vllm"
FIXTURE_SCHEMA_VERSION = 1

MCQ = dataclasses.replace(MCQ_PROMPT, template_id="golden/mcq")
PREFILL = dataclasses.replace(PREFILL_PROMPT, template_id="golden/prefill")


class RecordingTransport(httpx.AsyncBaseTransport):
    """Passes requests through and keeps (request JSON, response JSON/text) pairs."""

    def __init__(self) -> None:
        self._inner = httpx.AsyncHTTPTransport()
        self.exchanges: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        await response.aread()
        recorded: dict[str, Any] = {"status": response.status_code}
        if request.url.path == METRICS_PATH:  # 50 kB of counters; keep the info metric only
            lines = [ln for ln in response.text.splitlines() if "vllm:cache_config_info" in ln]
            recorded["text"] = "\n".join(lines) + "\n"
        else:
            recorded["json"] = json.loads(response.content)
        self.exchanges.append(
            {
                "request": {
                    "method": request.method,
                    "path": request.url.path,
                    "json": json.loads(request.content) if request.content else None,
                },
                "response": recorded,
            }
        )
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


def _prompt_spec(prompt: RenderedPrompt) -> dict[str, Any]:
    return {
        "messages": [{"role": m.role, "content": m.content} for m in prompt.messages],
        "template_id": prompt.template_id,
    }


async def _record(url: str, model: str) -> dict[str, dict[str, Any]]:
    image = os.environ.get(IMAGE_ENV)
    revision = os.environ.get(MODEL_REVISION_ENV)
    fixtures: dict[str, dict[str, Any]] = {}

    async def capture(
        name: str,
        call: dict[str, Any],
        run: Callable[[VLLMHTTPBackend], Awaitable[object]],
        capabilities: dict[str, Any] | None = None,
    ) -> None:
        caps = dataclasses.replace(DEFAULT_CAPABILITIES, **(capabilities or {}))
        transport = RecordingTransport()  # closed with the backend's client
        async with VLLMHTTPBackend(
            url,
            model,
            capabilities=caps,
            image=image,
            model_revision=revision,
            transport=transport,
            max_concurrency=4,
        ) as backend:
            try:
                await run(backend)
            except VLLMHTTPError:
                if not name.endswith("_error"):
                    raise
        call = {**call, "capabilities": capabilities or {}}
        fixtures[name] = {"call": call, "exchanges": transport.exchanges}

    def first_token(
        allowed: list[str] | None, top_k: int
    ) -> Callable[[VLLMHTTPBackend], Awaitable[object]]:
        return lambda b: b.next_token_logprobs(MCQ, allowed=allowed, top_k=top_k)

    await capture("health", {"method": "health"}, lambda b: b.health(), {"top_logprobs_max": 64})
    for name, allowed in (("first_token", None), ("first_token_constrained", ["A", "B"])):
        await capture(
            name,
            {
                "method": "next_token_logprobs",
                "prompt": _prompt_spec(MCQ),
                "allowed": allowed,
                "top_k": 20,
            },
            first_token(allowed, 20),
            {"top_logprobs_max": 64},
        )
    await capture(
        "top_logprobs_above_cap_error",
        {
            "method": "next_token_logprobs",
            "prompt": _prompt_spec(MCQ),
            "allowed": None,
            "top_k": 65,
        },
        first_token(None, 65),
        {"top_logprobs_max": 65},
    )
    continuations = [" Paris", " London"]
    await capture(
        "echo_sequence_logprobs",
        {
            "method": "sequence_logprobs",
            "prompt": _prompt_spec(PREFILL),
            "continuations": continuations,
        },
        lambda b: b.sequence_logprobs(PREFILL, continuations),
    )
    await capture(
        "tokenize", {"method": "tokenize", "text": "Answer: A"}, lambda b: b.tokenize("Answer: A")
    )
    ids = fixtures["tokenize"]["exchanges"][0]["response"]["json"]["tokens"]
    await capture("detokenize", {"method": "detokenize", "ids": ids}, lambda b: b.detokenize(ids))

    version = fixtures["health"]["exchanges"]
    vllm_version = next(e for e in version if e["request"]["path"] == "/version")["response"]
    meta = {
        "fixture_schema_version": FIXTURE_SCHEMA_VERSION,
        "vllm_version": vllm_version["json"]["version"],
        "model": model,
        "model_revision": revision,
        "image": image,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return {name: {"meta": meta, **fixture} for name, fixture in fixtures.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=os.environ.get("JEVEMU_VLLM_URL", "http://localhost:8000"))
    parser.add_argument("--model", default=os.environ.get("JEVEMU_VLLM_MODEL", "Qwen/Qwen3-0.6B"))
    parser.add_argument("--out", type=Path, default=FIXTURE_DIR)
    args = parser.parse_args(argv)
    fixtures = asyncio.run(_record(args.url, args.model))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, fixture in fixtures.items():
        path = args.out / f"{name}.json"
        path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(fixture['exchanges'])} exchanges)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
