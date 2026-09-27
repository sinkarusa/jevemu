"""Command-line entry point (``jevemu``)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from jevemu import __version__
from jevemu.backends.vllm_http import VLLMHTTPBackend
from jevemu.emulator import Emulator
from jevemu.types import SystemOneRequest, SystemOneResponse

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """jevemu: Jev System One emulator on vLLM."""


@app.command("version")
def show_version() -> None:
    """Print the installed jevemu version."""
    typer.echo(__version__)


@app.command("ask")
def ask(
    request: Annotated[Path, typer.Argument(help="System One request JSON ('-' reads stdin).")],
    url: Annotated[
        str, typer.Option(help="vLLM server URL.", envvar="JEVEMU_VLLM_URL")
    ] = "http://localhost:8000",
    model: Annotated[
        str, typer.Option(help="Served model id.", envvar="JEVEMU_VLLM_MODEL")
    ] = "Qwen/Qwen3-0.6B",
    diagnostics: Annotated[
        bool, typer.Option("--diagnostics", help="Include per-question x_jevemu diagnostics.")
    ] = False,
    round_to: Annotated[
        float | None, typer.Option(help="Round reported numbers like Jev (e.g. 0.01).")
    ] = None,
) -> None:
    """Answer a System One request with the emulator and print the response JSON."""
    text = typer.get_text_stream("stdin").read() if str(request) == "-" else request.read_text()
    parsed = SystemOneRequest.model_validate_json(text)
    response = asyncio.run(_ask(parsed, url, model, diagnostics=diagnostics, round_to=round_to))
    body = response.model_dump(mode="json")
    if body.get("x_jevemu") is None:
        body.pop("x_jevemu", None)
    typer.echo(json.dumps(body, indent=2, ensure_ascii=False))


async def _ask(
    request: SystemOneRequest,
    url: str,
    model: str,
    *,
    diagnostics: bool,
    round_to: float | None,
) -> SystemOneResponse:
    async with VLLMHTTPBackend(url, model) as backend:
        emulator = Emulator(backend, include_diagnostics=diagnostics, round_to=round_to)
        return await emulator.system_one(request)
