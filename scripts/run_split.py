"""Run one system on benchmark splits, summarize a run directory, compare two, or cost several.

    # Jev reference (TYPESAFE_API_KEY from the environment or .env; default response cache, so
    # a rerun costs nothing)
    uv run python scripts/run_split.py run --system jev --benchmarks all --split select \\
        --out runs/select --concurrency 16 --max-usd 1.0

    # CLM-8B on the server docker/clm/compose.yaml runs (scripts/serve_clm.sh); --url defaults
    # to JEVEMU_CLM_URL, else http://localhost:8700
    uv run python scripts/run_split.py run --system clm --benchmarks all --split select \\
        --out runs/select --concurrency 16

    # An emulator configuration on the vLLM server scripts/serve_vllm.sh started
    eval "$(scripts/serve_vllm.sh --preset NAME | grep '^export ')"
    uv run python scripts/run_split.py run --system emulator --system-id NAME \\
        --strategy auto_single --benchmarks all --split select --out runs/select

    # GPT-6 Luna over the OpenAI Chat Completions API (OPENAI_API_KEY from the environment or
    # .env; response cache ~/.cache/jevemu/openai_cache.sqlite, so a rerun costs nothing), read
    # by Structured Outputs. Benchmarks with more options than the 5 top logprobs (mmlu_pro,
    # banking77, clinc150) are skipped with a note.
    uv run python scripts/run_split.py run --system openai --strategy constrained \\
        --prompt-cache-mode explicit --system-id gpt-6-luna.constrained.state_first \\
        --benchmarks all --split screen --out runs/select --concurrency 16 --max-usd 1

    # Any OpenRouter model whose providers return logprobs (OPENROUTER_API_KEY; response cache
    # ~/.cache/jevemu/openrouter_cache.sqlite). --provider pins providers, no fallbacks; the
    # default system id gets "@<provider>" (deepseek__deepseek-v4.1-flash@makora-constrained-...).
    uv run python scripts/run_split.py run --system openrouter --strategy constrained \\
        --model deepseek/deepseek-v4.1-flash --provider makora \\
        --benchmarks all --split screen --out runs/select --concurrency 16 --max-usd 1

    uv run python scripts/run_split.py summarize runs/select/jev
    uv run python scripts/run_split.py compare runs/select/NAME runs/select/jev
    uv run python scripts/run_split.py stats runs/select/jev runs/select/NAME \\
        --split select --gpu-watts 280 --usd-per-wh 0.00027

``run`` answers each benchmark's split with ``jevemu.bench.runner.run_split`` (resumable: rerun
the same command to finish an interrupted or budget-stopped run; failed items are retried) into
``<out>/<system_id>/``, then prints the summary table. The emulator reads ``--url``/``--model``
from ``JEVEMU_VLLM_URL``/``JEVEMU_VLLM_MODEL`` by default and records diagnostics per item.
``--system clm`` sends the items to ``clm-serve`` (``jevemu.clm_client.ClmClient``: the Jev
client with CLM's pinned served model, no key unless ``CLM_API_KEY`` is set, no per-token price,
response cache ``~/.cache/jevemu/clm_cache.sqlite``); its runs are priced by GPU time like the
emulator's.
``--system openai`` wraps ``jevemu.backends.openai_chat.OpenAIChatBackend`` (``--model`` default
``gpt-6-luna``) in the same ``Emulator``: prompts without the ``Answer:`` prefill, spend capped by
``--max-usd``, ``--service-tier flex`` for Flex prices and ``--prompt-cache-mode`` for
``prompt_cache_options.mode``. ``--strategy constrained`` reads the answer by Structured Outputs
(a JSON ``enum`` of the labels, the distribution read where the value starts:
``jevemu.backends.chat_api.parse_structured_response``); ``first_token`` reads the first token of
a free reply. Either way every label must fit the top logprobs (5 for OpenAI), so a benchmark
with a question of more options is skipped.
``--system openrouter`` does the same over ``jevemu.backends.openrouter_chat.OpenRouterChatBackend``
(``--model`` default ``deepseek/deepseek-v4.1-flash``): reasoning disabled, ``--top-logprobs``
(default 20) and ``--max-tokens`` (default 16), priced from OpenRouter's public endpoints listing
(fetched at start, recorded in the manifest; spend is OpenRouter's billed ``usage.cost``). A
structured read needs a provider whose top logprobs are each position's own (Makora, not Wafer:
``docs/research/openrouter_probe_report.md``).
``--strategy auto_single`` (``jevemu.scoring.SingleCallStrategy``) is the selection runs' policy:
every question in one model call, every answer label one token. Choice questions above 32
options get single-token two-capital codes read from one constrained top-k; the server needs
``--max-logprobs`` 576 (``scripts/serve_vllm.sh``'s default), and a question one call cannot
score fails. ``--strategy auto_noecho`` (``jevemu.scoring.NoEchoAutoStrategy``: ``auto`` with
echo off, trie at tau 1e-3) is the older multi-call policy.
``--debiaser SPEC`` debiases the strategy's log-probabilities with
``jevemu.debias.debiaser_from_spec(SPEC)`` (``pride``, ``pride:alpha=0.05``, ``permutation:n=4``,
``contextual``, ``batch:priors=PATH``; default none); the manifest records its id.
``stats`` prints token, latency and cost statistics per system and per benchmark
(``jevemu.bench.compare.stats`` with ``jevemu.eval.costs.PriceBook``): API systems at their
built-in per-token price, the emulator's GPU time at an explicit ``--gpu-usd-per-hour`` (or
``JEVEMU_GPU_USD_PER_HOUR``), else at an electricity estimate: ``--gpu-watts`` x ``--usd-per-wh``
(or ``JEVEMU_GPU_WATTS``/``JEVEMU_USD_PER_WH``; defaults 410 W (observed) and $0.00027/Wh).
Exit status: 0 when every split is complete, 1 when items failed, 2 when the budget stopped
the run.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import get_args

from jevemu.backends.chat_api import ChatCompletionsBackend
from jevemu.backends.openai_chat import DEFAULT_MODEL as OPENAI_DEFAULT_MODEL
from jevemu.backends.openai_chat import OpenAIChatBackend, PromptCacheMode, ServiceTier
from jevemu.backends.openrouter_chat import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_LOGPROBS,
    OpenRouterChatBackend,
    fetch_endpoints,
)
from jevemu.backends.openrouter_chat import DEFAULT_MODEL as OPENROUTER_DEFAULT_MODEL
from jevemu.backends.vllm_http import VLLMHTTPBackend
from jevemu.bench.compare import compare, stats, stats_table, summarize
from jevemu.bench.runner import RunResult, frozen_split, run_split
from jevemu.clm_client import DEFAULT_URL as CLM_DEFAULT_URL
from jevemu.clm_client import MODEL as CLM_MODEL
from jevemu.clm_client import URL_ENV as CLM_URL_ENV
from jevemu.clm_client import ClmClient
from jevemu.compat.typesafe import DEFAULT_BACKEND_MODEL, DEFAULT_URL, MODEL_ENV, URL_ENV
from jevemu.debias import debiaser_from_spec
from jevemu.emulator import Emulator
from jevemu.eval.costs import (
    DEFAULT_GPU_WATTS,
    DEFAULT_USD_PER_WH,
    GPU_USD_PER_HOUR_ENV,
    GPU_WATTS_ENV,
    USD_PER_WH_ENV,
    PriceBook,
)
from jevemu.eval.metrics import DEFAULT_BOOTSTRAP_RESAMPLES
from jevemu.eval.splits import DATASETS, DEFAULT_SCREEN_SIZE, DROPPED_DATASETS, SPLITS
from jevemu.jev_client import JevClient, load_env_file
from jevemu.render import DEFAULT_LAYOUT, LAYOUTS
from jevemu.scoring import (
    AutoStrategy,
    ConstrainedStrategy,
    EchoStrategy,
    FirstTokenStrategy,
    NoEchoAutoStrategy,
    ScoringStrategy,
    SingleCallStrategy,
    TrieStrategy,
)
from jevemu.scoring.first_token import fits_top_k
from jevemu.types import SystemOneClient

STRATEGIES: dict[str, Callable[[], ScoringStrategy]] = {
    "auto": AutoStrategy,
    "auto_noecho": NoEchoAutoStrategy,
    "auto_single": SingleCallStrategy,
    "first_token": FirstTokenStrategy,
    "constrained": ConstrainedStrategy,
    "trie": TrieStrategy,
    "echo_sum": lambda: EchoStrategy("sum"),
    "echo_mean": lambda: EchoStrategy("mean"),
    "echo_pmi": lambda: EchoStrategy("pmi"),
}
RENDERERS = ("default", *LAYOUTS)
HTTP_TIMEOUT_S = 600.0


def benchmark_names(value: str) -> list[str]:
    if value == "all":
        return list(DATASETS)
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(DATASETS) - set(DROPPED_DATASETS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown benchmarks {unknown}; expected {DATASETS}")
    return names


def default_system_id(args: argparse.Namespace) -> str:
    if args.system in ("jev", "clm"):
        return str(args.system)
    parts = [str(args.model).replace("/", "__")]
    if args.system == "openrouter" and args.provider:
        parts[0] += "@" + "+".join(args.provider).replace("/", "__")
    if args.strategy != "auto":
        parts.append(args.strategy)
    layout = DEFAULT_LAYOUT if args.renderer == "default" else args.renderer
    if layout != "question_first":  # no suffix: question_first, as the Stage 1 runs are named
        parts.append(layout)
    if args.debiaser is not None:
        parts.append(args.debiaser.split(":", 1)[0])
    if args.system == "openai":
        if args.service_tier != "default":
            parts.append(args.service_tier)
        if args.prompt_cache_mode is not None:
            parts.append(f"cache_{args.prompt_cache_mode}")
    return "-".join(parts)


async def run_all(args: argparse.Namespace, system: SystemOneClient, system_id: str) -> int:
    results: list[RunResult] = []
    for benchmark in args.benchmarks:
        result = await run_split(
            system,
            system_id,
            benchmark,
            args.split,
            args.out,
            concurrency=args.concurrency,
            seed=args.seed,
            screen_size=args.screen_size,
            limit=args.limit,
        )
        results.append(result)
        print(
            f"{benchmark}.{args.split}: {result.status}, {result.n_ok}/{result.n_items} ok, "
            f"{result.n_errors} errors, {result.n_called} called, {result.n_skipped} skipped, "
            f"${result.spent_usd:.6f}, {result.elapsed_s:.1f} s",
            flush=True,
        )
        if result.status == "stopped_budget":
            print(f"stopped: {result.stop_reason}; rerun to resume", file=sys.stderr)
            return 2
    return 0 if all(r.status == "complete" for r in results) else 1


async def run_typesafe(args: argparse.Namespace, system_id: str, client: JevClient) -> int:
    """Jev, or another server of its wire format, through a ``JevClient``."""
    async with client:
        code = await run_all(args, client, system_id)
        print(
            f"{client.service}: ${client.spent_usd:.6f} spent, {client.network_calls} network "
            f"calls, {client.cache_hits} cache hits"
        )
    return code


async def run_jev(args: argparse.Namespace, system_id: str) -> int:
    load_env_file()
    return await run_typesafe(args, system_id, JevClient(max_usd=args.max_usd))


async def run_clm(args: argparse.Namespace, system_id: str) -> int:
    load_env_file()
    client = ClmClient(model=args.model, base_url=args.url, timeout=HTTP_TIMEOUT_S)
    return await run_typesafe(args, system_id, client)


async def run_emulator(args: argparse.Namespace, system_id: str) -> int:
    async with VLLMHTTPBackend(
        args.url,
        args.model,
        max_concurrency=max(32, args.concurrency),
        timeout=HTTP_TIMEOUT_S,
    ) as backend:
        emulator = Emulator(
            backend,
            renderer=args.renderer,
            strategy=STRATEGIES[args.strategy](),
            debiaser=None if args.debiaser is None else debiaser_from_spec(args.debiaser),
            include_diagnostics=True,
            max_concurrency=args.concurrency,
        )
        return await run_all(args, emulator, system_id)


async def run_chat_api(
    args: argparse.Namespace, system_id: str, backend: ChatCompletionsBackend
) -> int:
    """An ``Emulator`` over a hosted chat API, on the benchmarks whose questions fit the
    backend's top logprobs."""
    async with backend:
        fitting = []
        for benchmark in args.benchmarks:
            split = frozen_split(
                args.out, benchmark, args.split, seed=args.seed, screen_size=args.screen_size
            )
            if all(fits_top_k(backend.capabilities, item.question) for item in split.items):
                fitting.append(benchmark)
            else:
                print(
                    f"{benchmark}: skipped, its questions have more options than the "
                    f"{backend.capabilities.top_logprobs_max} top logprobs "
                    f"{backend.service} returns",
                    flush=True,
                )
        args.benchmarks = fitting
        emulator = Emulator(
            backend,
            renderer=args.renderer,
            strategy=STRATEGIES[args.strategy](),
            debiaser=None if args.debiaser is None else debiaser_from_spec(args.debiaser),
            include_diagnostics=True,
            max_concurrency=args.concurrency,
        )
        try:
            return await run_all(args, emulator, system_id)
        finally:
            print(
                f"{backend.service}: ${backend.spent_usd:.6f} spent, "
                f"{backend.network_calls} network calls, {backend.cache_hits} cache hits, "
                f"{backend.empty_replies} empty replies resent"
            )
            for identity, count in backend.identities.most_common():
                print(
                    f"  model {identity.model}, system_fingerprint "
                    f"{identity.system_fingerprint}, provider {identity.provider}: "
                    f"{count} responses"
                )


async def run_openai(args: argparse.Namespace, system_id: str) -> int:
    load_env_file()
    backend = OpenAIChatBackend(
        args.model,
        max_usd=args.max_usd,
        max_rpm=args.max_rpm,
        max_concurrency=args.concurrency,
        service_tier=args.service_tier,
        prompt_cache_mode=args.prompt_cache_mode,
        timeout=HTTP_TIMEOUT_S,
    )
    return await run_chat_api(args, system_id, backend)


async def run_openrouter(args: argparse.Namespace, system_id: str) -> int:
    load_env_file()
    backend = OpenRouterChatBackend(
        args.model,
        endpoints=await fetch_endpoints(args.model),
        providers=args.provider,
        top_logprobs=args.top_logprobs,
        max_tokens=args.max_tokens,
        max_usd=args.max_usd,
        max_rpm=args.max_rpm,
        max_concurrency=args.concurrency,
        timeout=HTTP_TIMEOUT_S,
    )
    print(f"OpenRouter price snapshot: {backend.price.describe()}", flush=True)
    return await run_chat_api(args, system_id, backend)


RUNNERS = {
    "jev": run_jev,
    "clm": run_clm,
    "emulator": run_emulator,
    "openai": run_openai,
    "openrouter": run_openrouter,
}
DEFAULT_MODELS = {
    "clm": CLM_MODEL,
    "openai": OPENAI_DEFAULT_MODEL,
    "openrouter": OPENROUTER_DEFAULT_MODEL,
}


def provider_slugs(value: str) -> list[str]:
    return [slug.strip() for slug in value.split(",") if slug.strip()]


def cmd_run(args: argparse.Namespace) -> int:
    if args.model is None:
        args.model = DEFAULT_MODELS.get(args.system) or os.environ.get(
            MODEL_ENV, DEFAULT_BACKEND_MODEL
        )
    if args.url is None:
        args.url = (
            os.environ.get(CLM_URL_ENV, CLM_DEFAULT_URL)
            if args.system == "clm"
            else os.environ.get(URL_ENV, DEFAULT_URL)
        )
    system_id = args.system_id or default_system_id(args)
    code = asyncio.run(RUNNERS[args.system](args, system_id))
    run_dir = Path(args.out) / system_id
    summary = summarize(run_dir, split=args.split, n_resamples=args.resamples)
    print()
    print(summary.to_markdown())
    return code


def cmd_summarize(args: argparse.Namespace) -> int:
    summary = summarize(args.run_dir, split=args.split, n_resamples=args.resamples)
    print(f"{summary.system_id} ({summary.split}): {summary.system}")
    print()
    print(summary.to_markdown())
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    comparison = compare(
        args.run_a,
        args.run_b,
        split=args.split,
        n_resamples=args.resamples,
        benchmarks=args.benchmarks,
    )
    print(f"a = {comparison.run_a}, b = {comparison.run_b} ({comparison.split}); Δ = a - b")
    print()
    print(comparison.to_markdown())
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    prices = PriceBook.default(
        args.gpu_usd_per_hour, gpu_watts=args.gpu_watts, usd_per_wh=args.usd_per_wh
    )
    runs = [stats(run_dir, prices, split=args.split) for run_dir in args.run_dirs]
    print(f"Token and cost statistics on `{args.split}` (all benchmarks of each run)")
    print()
    print(stats_table(runs))
    for run in runs:
        print(f"### `{run.system_id}`")
        print()
        print("\n".join(f"- {note}" for note in run.notes()))
        print()
        print(run.to_markdown())
        if run.total.n_price_mismatch:
            print(
                f"warning: {run.system_id}: {run.total.n_price_mismatch} items' recorded cost "
                "differs from the price table by more than 1%",
                file=sys.stderr,
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="answer benchmark splits with one system")
    run.add_argument("--system", choices=tuple(RUNNERS), required=True)
    run.add_argument("--system-id", help="run directory name (default: jev / clm / the model name)")
    run.add_argument("--benchmarks", type=benchmark_names, default=list(DATASETS))
    run.add_argument("--split", choices=SPLITS, default="select")
    run.add_argument("--out", type=Path, default=Path("runs/select"))
    run.add_argument("--concurrency", type=int, default=16)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--screen-size", type=int, default=DEFAULT_SCREEN_SIZE)
    run.add_argument("--limit", type=int, help="first N items of each split")
    run.add_argument(
        "--max-usd", type=float, default=1.0, help="API spend cap (Jev, OpenAI, OpenRouter)"
    )
    run.add_argument(
        "--url",
        help=f"server URL (emulator: ${URL_ENV}, else {DEFAULT_URL}; "
        f"clm: ${CLM_URL_ENV}, else {CLM_DEFAULT_URL})",
    )
    run.add_argument(
        "--model",
        help=f"backend model (emulator: ${MODEL_ENV}, else {DEFAULT_BACKEND_MODEL}; "
        f"clm: {CLM_MODEL}; openai: {OPENAI_DEFAULT_MODEL}; "
        f"openrouter: {OPENROUTER_DEFAULT_MODEL})",
    )
    run.add_argument(
        "--max-rpm", type=int, default=450, help="openai/openrouter: request starts per minute"
    )
    run.add_argument(
        "--provider",
        type=provider_slugs,
        default=[],
        metavar="SLUG[,SLUG]",
        help="openrouter: pin provider slugs in this order, no fallbacks (default: any provider "
        "listing logprobs)",
    )
    run.add_argument(
        "--top-logprobs",
        type=int,
        default=DEFAULT_TOP_LOGPROBS,
        help="openrouter: top logprobs requested",
    )
    run.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="openrouter: max_tokens"
    )
    run.add_argument("--service-tier", choices=get_args(ServiceTier), default="default")
    run.add_argument(
        "--prompt-cache-mode",
        choices=get_args(PromptCacheMode),
        help="openai: prompt_cache_options.mode (default: the API default)",
    )
    run.add_argument("--renderer", choices=RENDERERS, default="default")
    run.add_argument("--strategy", choices=sorted(STRATEGIES), default="auto")
    run.add_argument("--debiaser", metavar="SPEC", help="jevemu.debias spec (default: none)")
    run.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    run.set_defaults(handler=cmd_run)

    summary = commands.add_parser("summarize", help="score a run directory")
    summary.add_argument("run_dir", type=Path)
    summary.add_argument("--split", choices=SPLITS, default="select")
    summary.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    summary.set_defaults(handler=cmd_summarize)

    paired = commands.add_parser("compare", help="pair two run directories (a - b)")
    paired.add_argument("run_a", type=Path)
    paired.add_argument("run_b", type=Path)
    paired.add_argument("--split", choices=SPLITS, default="select")
    paired.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    paired.add_argument(
        "--benchmarks",
        type=benchmark_names,
        default=None,
        help="only these (comma list), e.g. a sensitivity analysis leaving one out",
    )
    paired.set_defaults(handler=cmd_compare)

    costs = commands.add_parser("stats", help="token and cost statistics of run directories")
    costs.add_argument("run_dirs", type=Path, nargs="+")
    costs.add_argument("--split", choices=SPLITS, default="select")
    costs.add_argument(
        "--gpu-usd-per-hour",
        type=float,
        help=f"explicit local GPU price; overrides the electricity estimate "
        f"(default: ${GPU_USD_PER_HOUR_ENV})",
    )
    costs.add_argument(
        "--gpu-watts",
        type=float,
        help=f"average GPU power draw for the electricity estimate "
        f"(default: ${GPU_WATTS_ENV}, else {DEFAULT_GPU_WATTS:g} W, observed)",
    )
    costs.add_argument(
        "--usd-per-wh",
        type=float,
        help=f"electricity price per watt-hour (default: ${USD_PER_WH_ENV}, "
        f"else ${DEFAULT_USD_PER_WH:g}/Wh, NJ average)",
    )
    costs.set_defaults(handler=cmd_stats)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    code: int = args.handler(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
