"""Probe a live vLLM server's scoring-relevant behavior and print the report.

    uv run python scripts/probe_vllm.py --url http://localhost:8000 --json-out report.json

Exit status 1 when the server ignores ``structured_outputs`` (constraint not enforced).
Defaults come from the same environment variables as docker/vllm/compose.yaml.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from jevemu.backends.probe import (
    ServerFlags,
    default_cache_dir,
    format_report,
    load_or_run_probe,
    run_probe,
)
from jevemu.backends.vllm_http import (
    IMAGE_ENV,
    MODEL_REVISION_ENV,
    MODELS_PATH,
    VLLMHTTPBackend,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    env = os.environ.get
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=env("JEVEMU_VLLM_URL", "http://localhost:8000"))
    parser.add_argument(
        "--model", default=env("JEVEMU_VLLM_MODEL"), help="default: first served model"
    )
    parser.add_argument(
        "--max-logprobs",
        type=int,
        default=int(env("JEVEMU_VLLM_MAX_LOGPROBS", "576")),
        help="declared --max-logprobs of the server (verified by the probe)",
    )
    parser.add_argument(
        "--logprobs-mode",
        choices=["raw_logprobs", "processed_logprobs"],
        default=env("JEVEMU_VLLM_LOGPROBS_MODE", "raw_logprobs"),
        help="declared --logprobs-mode of the server (not observable over HTTP)",
    )
    parser.add_argument("--image", default=env(IMAGE_ENV), help="server image, pinned by digest")
    parser.add_argument(
        "--model-revision", default=env(MODEL_REVISION_ENV), help="served model's HF commit SHA"
    )
    parser.add_argument("--chat-template-kwargs", type=json.loads, default=None, help="JSON object")
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--refresh", action="store_true", help="re-probe and overwrite the cache")
    cache.add_argument("--no-cache", action="store_true", help="neither read nor write the cache")
    parser.add_argument("--json-out", type=Path, help="also write the full report as JSON")
    return parser.parse_args(argv)


async def _first_model(url: str) -> str:
    async with httpx.AsyncClient(base_url=url.rstrip("/").removesuffix("/v1")) as client:
        response = await client.get(MODELS_PATH)
        response.raise_for_status()
        return str(response.json()["data"][0]["id"])


async def _main(args: argparse.Namespace) -> int:
    model = args.model or await _first_model(args.url)
    flags = ServerFlags(max_logprobs=args.max_logprobs, logprobs_mode=args.logprobs_mode)
    async with VLLMHTTPBackend(
        args.url,
        model,
        image=args.image,
        model_revision=args.model_revision,
        chat_template_kwargs=args.chat_template_kwargs,
    ) as backend:
        if args.no_cache:
            report = await run_probe(backend, flags)
        else:
            report = await load_or_run_probe(
                backend, flags, cache_dir=args.cache_dir, refresh=args.refresh
            )
    print(format_report(report))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0 if report.constraint.enforced else 1


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(_parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
