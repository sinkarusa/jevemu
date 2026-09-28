"""Client for a local CLM-8B server (``clm-serve`` from https://github.com/Contrastive-LM/CLM).

CLM-8B is a dual encoder: Qwen3-8B, served by vLLM as a pooling (embedding) model, and a 20M
parameter projection head that scores each option against the state. ``clm-serve`` answers the
TypeSafe ``POST /v1/systemone`` wire format as Jev does, so :class:`ClmClient` is a
:class:`~jevemu.jev_client.JevClient` with other defaults: the pinned served model id
:data:`MODEL`, ``http://localhost:8700``, no API key unless the server sets ``CLM_API_KEY``, no
price per token (its cost is the GPU time of the run, as for the vLLM emulator), no rate limit
and its own response cache.

``docker/clm/compose.yaml`` serves the pins below and must be kept in step with them. The served
id ends in the first 12 hex digits of the head's sha256, which the container checks before
serving the file under that id. A different head is therefore served under a different id:
requests pinned to this one fail (422, unknown model), and since the response cache key covers
the model id, no cached answer of another head (or of Jev) can be returned for it. The encoder
pins are not part of the id; changing them means a new :data:`MODEL` here and in the compose file.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any

import httpx

from jevemu.cache import ResponseCache
from jevemu.errors import JevVersionDrift
from jevemu.jev_client import Clock, JevClient, JevModelInfo, RetryPolicy, Sleep

__all__ = [
    "API_KEY_ENV",
    "DEFAULT_URL",
    "ENCODER_IMAGE",
    "ENCODER_MAX_TOKENS",
    "ENCODER_MODEL",
    "ENCODER_REVISION",
    "HEAD_FILE",
    "HEAD_REPO",
    "HEAD_REVISION",
    "HEAD_SHA256",
    "MODEL",
    "PACKAGE",
    "PACKAGE_WHEEL_SHA256",
    "URL_ENV",
    "ClmClient",
    "default_cache_path",
    "provenance",
]

HEAD_REPO = "Contrastive-LM/CLM-v0.1-8B"
HEAD_REVISION = "e939398d4556fcd9400c76fa8c5a513202f42b0a"
HEAD_FILE = "CLM_v0.1-8B.pt"
HEAD_SHA256 = "b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5"
ENCODER_MODEL = "Qwen/Qwen3-8B"
ENCODER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
ENCODER_IMAGE = (
    "vllm/vllm-openai:v0.30.0"
    "@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90"
)
ENCODER_MAX_TOKENS = 2048
"""``vllm serve --max-model-len`` and ``clm-serve --max-tokens``: longer texts are truncated."""
PACKAGE = "contrastive-lm==0.1.0"
PACKAGE_WHEEL_SHA256 = "7f3eed12d3aa10173f71bac7e50fc387565b485a2a2fc4c4323af32b1950055f"

MODEL = f"clm-v0.1-8b-{HEAD_SHA256[:12]}"
"""The id ``docker/clm/compose.yaml`` serves the pinned head under."""
DEFAULT_URL = "http://localhost:8700"
URL_ENV = "JEVEMU_CLM_URL"
"""Environment variable ``scripts/run_split.py`` reads the server URL from."""
API_KEY_ENV = "CLM_API_KEY"

_SERVED_MODEL = re.compile(r"clm-.+-[0-9a-f]{12}")
"""A head-pinned id; ``clm-latest`` and ``clm-raw`` do not match."""


def default_cache_path() -> Path:
    """``~/.cache/jevemu/clm_cache.sqlite``."""
    return Path.home() / ".cache" / "jevemu" / "clm_cache.sqlite"


def provenance() -> dict[str, Any]:
    """The pins behind :data:`MODEL`, as recorded in run manifests."""
    return {
        "head": {
            "repo": HEAD_REPO,
            "revision": HEAD_REVISION,
            "file": HEAD_FILE,
            "sha256": HEAD_SHA256,
        },
        "encoder": {
            "model": ENCODER_MODEL,
            "revision": ENCODER_REVISION,
            "image": ENCODER_IMAGE,
            "max_tokens": ENCODER_MAX_TOKENS,
        },
        "package": {"requirement": PACKAGE, "wheel_sha256": PACKAGE_WHEEL_SHA256},
    }


class ClmClient(JevClient):
    """:class:`~jevemu.jev_client.JevClient` for ``clm-serve`` (module docstring).

    ``cache=None`` uses :func:`default_cache_path`. The key is the argument, else
    ``CLM_API_KEY``, else none (the server's default: auth off). ``max_rpm=None``: no rate limit.
    """

    service = "CLM"
    api_key_env = API_KEY_ENV
    api_key_required = False
    model_pattern = _SERVED_MODEL
    usd_per_input_token = 0.0

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = MODEL,
        base_url: str = DEFAULT_URL,
        cache: ResponseCache | None = None,
        max_rpm: int | None = None,
        retry: RetryPolicy | None = None,
        timeout: float | httpx.Timeout = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        super().__init__(
            api_key,
            model=model,
            base_url=base_url,
            cache=cache,
            max_rpm=max_rpm,
            retry=retry,
            timeout=timeout,
            transport=transport,
            clock=clock,
            sleep=sleep,
        )

    def _default_cache(self) -> ResponseCache:
        return ResponseCache(default_cache_path())

    async def served_model(self) -> JevModelInfo:
        """The pinned model's ``GET /v1/models`` entry; :class:`~jevemu.errors.JevVersionDrift`
        if the server does not serve it (another head, or the server's default ``clm-latest``)."""
        models = await self.list_models()
        for info in models:
            if info.name == self.model:
                return info
        served = ", ".join(info.name for info in models)
        raise JevVersionDrift(self.model, served, service=self.service)
