"""Record the scoring golden set from a live vLLM server (``scripts/serve_vllm.sh``).

    eval "$(scripts/serve_vllm.sh | grep '^export ')"
    uv run python tests/golden/scoring/record_scoring_fixtures.py

Writes ``scoring_vllm.jsonl`` (every backend call, via ``RecordingBackend``; capabilities from
the V2 probe) and ``expected.json`` (each strategy's result), replacing both. The expected
results come from replaying the fixture: strategies that send an identical request share one
recorded answer (the latest), while the live answers to repeated requests differ by bf16
prefix-cache and batch noise.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from scoring_golden_cases import CASES, EXPECTED, FIXTURE, pairs, strategies

from jevemu.backends.base import Backend
from jevemu.backends.probe import ServerFlags, probe_capabilities
from jevemu.backends.recorded import RecordedBackend, RecordingBackend
from jevemu.backends.vllm_http import VLLMHTTPBackend
from jevemu.render import PromptRenderer


async def run_all(backend: Backend) -> dict[str, Any]:
    by_name = strategies()
    results: dict[str, Any] = {}
    for case, name in pairs():
        layout, state, question, _ = CASES[case]
        result = await by_name[name].score(backend, PromptRenderer(layout), state, question)
        results[f"{case}/{name}"] = {
            "strategy": result.strategy,
            "keys": list(result.keys),
            "probabilities": [result.probabilities[k] for k in result.keys],
            "observed_mass": result.observed_mass,
            "missing": list(result.missing),
            "truncated": result.truncated,
            "n_backend_calls": result.n_backend_calls,
            "warnings": list(result.warnings),
        }
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
    print(f"recorded {len(expected)} results into {FIXTURE.name} and {EXPECTED.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
