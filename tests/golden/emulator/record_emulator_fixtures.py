"""Record the emulator golden set from a live vLLM server (``scripts/serve_vllm.sh``).

    eval "$(scripts/serve_vllm.sh | grep '^export ')"
    uv run python tests/golden/emulator/record_emulator_fixtures.py

Writes ``emulator_vllm.jsonl`` (every backend call, via ``RecordingBackend``; capabilities from
the V2 probe) and ``expected.json`` (each response without latencies), replacing both. The
expected responses come from replaying the fixture: identical requests share one recorded
answer (the latest), while live answers to repeated requests differ by bf16 prefix-cache and
batch noise.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from emulator_golden_cases import EXPECTED, FIXTURE, REQUESTS, load_request, make_emulator

from jevemu.backends.base import Backend
from jevemu.backends.probe import ServerFlags, probe_capabilities
from jevemu.backends.recorded import RecordedBackend, RecordingBackend
from jevemu.backends.vllm_http import VLLMHTTPBackend


def without_latency(response: dict[str, Any]) -> dict[str, Any]:
    """``response`` (JSON) minus the per-question ``latency_ms`` diagnostics (wall clock)."""
    diagnostics = {
        question_id: {k: v for k, v in diag.items() if k != "latency_ms"}
        for question_id, diag in (response.get("x_jevemu") or {}).items()
    }
    return {**response, "x_jevemu": diagnostics}


async def run_all(backend: Backend) -> dict[str, Any]:
    emulator = make_emulator(backend)
    results: dict[str, Any] = {}
    for name in REQUESTS:
        response = await emulator.system_one(load_request(name))
        results[name] = without_latency(response.model_dump(mode="json"))
    return results


async def record(url: str, model: str, max_logprobs: int) -> dict[str, Any]:
    async with VLLMHTTPBackend(url, model) as live:
        live.capabilities = await probe_capabilities(live, ServerFlags(max_logprobs=max_logprobs))
        FIXTURE.unlink(missing_ok=True)
        await run_all(RecordingBackend(live, FIXTURE))
    return await run_all(RecordedBackend(FIXTURE))


def main() -> int:
    url = os.environ.get("JEVEMU_VLLM_URL")
    if not url:
        print("set JEVEMU_VLLM_URL (see scripts/serve_vllm.sh)", file=sys.stderr)
        return 2
    model = os.environ.get("JEVEMU_VLLM_MODEL", "Qwen/Qwen3-0.6B")
    max_logprobs = int(os.environ.get("JEVEMU_VLLM_MAX_LOGPROBS", "576"))
    expected = asyncio.run(record(url, model, max_logprobs))
    EXPECTED.write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(f"recorded {len(expected)} responses into {FIXTURE.name} and {EXPECTED.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
