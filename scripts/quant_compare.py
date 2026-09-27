"""Quantization ladder: do 8-bit and 4-bit Qwen3.5-9B checkpoints change the emulator's accuracy
or label probabilities beyond bf16 noise? Report: docs/research/quantization_report.md.

    uv run python scripts/quant_compare.py run                         # new runs/quant/<UTC ts>/
    uv run python scripts/quant_compare.py run --out runs/quant/<ts>   # resume: finished runs skip
    uv run python scripts/quant_compare.py analyze runs/quant/<ts>     # rewrite the report tables

``run`` measures every entry of ``RUNS`` in order (``--only NAME`` to pick some). Each run gets a
fresh container (``scripts/serve_vllm.sh --preset``; any running jevemu server is stopped first),
so every run starts with an empty prefix cache, and the server is stopped afterwards. Per run the
emulator (unrounded probabilities, diagnostics on) answers, with a fixed request concurrency:

- GPQA-Diamond (198) and LEXam-en (619) banks, seed 0, "I don't know" option, evaluate-idk
  order, constrained letters (S2, what ``auto`` picks for them);
- the same GPQA-Diamond items with echo scoring (S4, ``mode="sum"``);
- every Choice, Score and Noul question of the Jev doc request fixtures
  (``tests/golden/fixtures/jev_docs``), constrained.

Concurrency 1 means one HTTP request in flight (the backend's semaphore), so batch composition
cannot move logprobs; ``bf16-c16`` keeps 16 questions and 16 HTTP requests in flight instead.
``bf16-restart`` repeats the reference on a new container: the floor any quantization delta must
clear. Outputs per run: ``<name>.jsonl`` (one record per scored question: keys, renormalized
probabilities, raw label logprobs before renormalization, observed mass, answer, gold, prompt and
cached tokens, latency), ``<name>.log`` (the server log without access lines) and
``<name>.json`` (log extracts, template checks, backend info, capabilities, bank hashes, GPU). The
metadata file is written last and marks the run finished.

``analyze`` pairs every run with ``bf16`` item by item and rewrites the block between
``<!-- quant_compare:begin -->`` and ``<!-- quant_compare:end -->`` in the report (aggregates only:
GPQA questions and per-item outputs stay in the gitignored ``runs/``).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import numpy as np

from jevemu.backends.base import Backend, Capabilities, ChatMessage, RenderedPrompt
from jevemu.backends.vllm_http import VLLMHTTPBackend
from jevemu.emulator import Emulator
from jevemu.eval.bank import QUESTION_KEY, QuestionBank, build_question_bank
from jevemu.eval.metrics import (
    DEFAULT_ECE_BINS,
    BootstrapCI,
    accuracy,
    brier,
    ece,
    kl_divergence,
    mcnemar_exact,
    nll,
    paired_bootstrap_ci,
    total_variation,
)
from jevemu.render.renderer import PromptRenderer
from jevemu.scoring import ConstrainedStrategy, EchoStrategy, ScoreResult, ScoringStrategy
from jevemu.types import (
    ChoiceAnswer,
    JSONish,
    NoulAnswer,
    Question,
    ScoreAnswer,
    SystemOneRequest,
)

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ("docker", "compose", "-f", str(ROOT / "docker" / "vllm" / "compose.yaml"))
SERVE = ROOT / "scripts" / "serve_vllm.sh"
PRESETS = ROOT / "docker" / "vllm" / "presets"
DOC_FIXTURES = ROOT / "tests" / "golden" / "fixtures" / "jev_docs"
DEFAULT_REPORT = ROOT / "docs" / "research" / "quantization_report.md"
BEGIN, END = "<!-- quant_compare:begin -->", "<!-- quant_compare:end -->"

DATASETS = ("gpqa_diamond", "lexam_en")
SEED = 0
REFERENCE = "bf16"
HTTP_TIMEOUT_S = 600.0

Role = Literal["reference", "floor", "quant", "context"]


@dataclass(frozen=True)
class Run:
    name: str
    preset: str
    concurrency: int
    role: Role
    label: str


RUNS = (
    Run("bf16", "qwen3.5-9b-bf16", 1, "reference", "bf16 (reference)"),
    Run("bf16-restart", "qwen3.5-9b-bf16", 1, "floor", "bf16, new container"),
    Run("bf16-c16", "qwen3.5-9b-bf16", 16, "floor", "bf16, concurrency 16"),
    Run("fp8", "qwen3.5-9b-fp8", 1, "quant", "FP8 (RedHatAI)"),
    Run("int8", "qwen3.5-9b-int8", 1, "quant", "INT8 g32 (cyankiwi)"),
    Run("int4-cyankiwi", "qwen3.5-9b-int4-cyankiwi", 1, "quant", "INT4 g32 (cyankiwi)"),
    Run(
        "int4-quanttrio",
        "qwen3.5-9b-int4-quanttrio",
        1,
        "quant",
        "INT4 AWQ g128 (QuantTrio), fp16 activations (checkpoint dtype)",
    ),
    Run(
        "int4-quanttrio-bf16",
        "qwen3.5-9b-int4-quanttrio-bf16",
        1,
        "quant",
        "INT4 AWQ g128 (QuantTrio), bf16 activations",
    ),
    Run("27b-awq", "qwen3.8-27b-awq", 1, "context", "Qwen3.8-27B INT4 (cyankiwi)"),
)
RUNS_BY_NAME = {run.name: run for run in RUNS}

Part = tuple[str, str, str]
"""(part, strategy, dataset): ("bank", "constrained", "gpqa_diamond"), ("doc", ...), ..."""

# --- server lifecycle --------------------------------------------------------------------------


def _clean_env() -> dict[str, str]:
    """This process's environment without JEVEMU_VLLM_* (a preset must not inherit a previous
    run's exports)."""
    return {k: v for k, v in os.environ.items() if not k.startswith("JEVEMU_VLLM_")}


def stop_server() -> None:
    subprocess.run([*COMPOSE, "down"], cwd=ROOT, check=True, capture_output=True, text=True)


def start_server(preset: str) -> tuple[dict[str, str], float]:
    """Start ``preset`` and return the exports ``serve_vllm.sh`` prints and the seconds to
    healthy."""
    started = time.monotonic()
    proc = subprocess.run(
        [str(SERVE), "--preset", preset], cwd=ROOT, env=_clean_env(), capture_output=True, text=True
    )
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        raise RuntimeError(f"serve_vllm.sh --preset {preset} failed:\n{proc.stderr[-6000:]}")
    line = next(line for line in proc.stdout.splitlines() if line.startswith("export "))
    exports = dict(pair.split("=", 1) for pair in shlex.split(line)[1:])
    return exports, elapsed


LOG_VALUES = {
    "weights_gib": re.compile(r"Model loading took ([\d.]+) GiB memory"),
    "load_seconds": re.compile(r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds"),
    "kv_cache_gib": re.compile(r"Available KV cache memory: ([\d.]+) GiB"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
    "attention_block_size": re.compile(r"Setting attention block size to (\d+) tokens"),
    "serve_command": re.compile(r"^(vllm serve .*)$"),
    "quantization": re.compile(r"^Initializing a V1 LLM engine .*\bquantization=([\w-]+)"),
}
TEXT_VALUES = frozenset({"serve_command", "quantization"})
KERNEL_LINE = re.compile(
    r"marlin|machete|cutlass|exllama|fp8|awq|wna16|w8a8|w4a16|quantiz|LinearKernel|kernel for",
    re.IGNORECASE,
)
NOT_KERNEL_LINE = ("vllm serve", "Initializing a V1 LLM engine", "non-default args")
_LOG_LINE = re.compile(
    r"^\S+\s+\|\s(?:\(\w+ pid=\d+\)\s)?"  # container and process prefix
    r"(?:(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL) \d\d-\d\d [\d:]+ \[[^\]]+\] )?"
    r"(?P<message>.*)$"
)


def server_log() -> str:
    """The running container's log without HTTP access lines."""
    proc = subprocess.run(
        [*COMPOSE, "logs", "--no-color", "vllm"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return "".join(
        line + "\n"
        for line in proc.stdout.splitlines()
        if '"GET /' not in line and '"POST /' not in line
    )


def server_log_facts(log: str) -> dict[str, Any]:
    """Memory, cache and kernel facts from a server log."""
    parsed = [m for line in log.splitlines() if (m := _LOG_LINE.match(line))]
    lines = [m["message"].strip() for m in parsed]
    facts: dict[str, Any] = {}
    for key, pattern in LOG_VALUES.items():
        for line in lines:
            if match := pattern.search(line):
                value = match.group(1)
                facts[key] = value if key in TEXT_VALUES else float(value.replace(",", ""))
                break
    kernel: list[str] = []
    for line in lines:
        if KERNEL_LINE.search(line) and line not in kernel and not line.startswith(NOT_KERNEL_LINE):
            kernel.append(line[:400])
    facts["kernel_lines"] = kernel[:25]
    warnings = {m["message"].strip()[:300] for m in parsed if m["level"] in ("WARNING", "ERROR")}
    facts["warnings"] = sorted(warnings)[:40]
    return facts


def gpu_facts() -> dict[str, str]:
    fields = "name,driver_version,memory.total,memory.used"
    proc = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(zip(fields.split(","), (v.strip() for v in proc.stdout.split(",")), strict=True))


# --- measurement -------------------------------------------------------------------------------


class RecordingStrategy:
    """Delegates to ``inner`` and keeps each question's ``ScoreResult`` (the emulator reports
    renormalized probabilities; the comparison also needs the raw label logprobs)."""

    def __init__(self, inner: ScoringStrategy) -> None:
        self.inner = inner
        self.name = inner.name
        self.results: dict[int, ScoreResult] = {}

    def supports(self, capabilities: Capabilities, question: Question) -> bool:
        return self.inner.supports(capabilities, question)

    async def score(
        self, backend: Backend, renderer: PromptRenderer, state: JSONish, question: Question
    ) -> ScoreResult:
        result = await self.inner.score(backend, renderer, state, question)
        self.results[id(question)] = result
        return result


def _finite(values: Iterable[float]) -> list[float | None]:
    """JSON-safe logprobs: ``-inf`` (and any non-finite value) becomes ``null``."""
    return [v if math.isfinite(v) else None for v in values]


def _unfinite(values: Iterable[float | None]) -> list[float]:
    return [-math.inf if v is None else float(v) for v in values]


@dataclass(frozen=True)
class Job:
    """One request to answer and how to label its records."""

    part: Part
    request: SystemOneRequest
    ids: Mapping[str, str]
    """Question id in the request -> record id."""
    gold: str | None = None


def bank_jobs(bank: QuestionBank, strategy: str) -> list[Job]:
    return [
        Job(
            ("bank", strategy, item.dataset),
            item.request,
            {QUESTION_KEY: item.item_id},
            item.gold,
        )
        for item in bank.items
    ]


def doc_jobs() -> list[Job]:
    jobs = []
    for path in sorted(DOC_FIXTURES.glob("*__request.json")):
        request = SystemOneRequest.model_validate_json(path.read_text())
        stem = path.name.removesuffix("__request.json")
        jobs.append(
            Job(
                ("doc", "constrained", "jev_docs"),
                request,
                {q: f"{stem}:{q}" for q in request.questions},
            )
        )
    return jobs


async def answer_jobs(
    emulator: Emulator, recorder: RecordingStrategy, jobs: Sequence[Job], concurrency: int
) -> list[dict[str, Any]]:
    """Answer every job; with ``concurrency > 1`` the emulator's and backend's semaphores keep
    that many questions and HTTP requests in flight."""

    async def one(job: Job) -> list[dict[str, Any]]:
        started = time.perf_counter()
        response = await emulator.system_one(job.request)
        latency_ms = (time.perf_counter() - started) * 1000.0
        records = []
        for question_id, question in job.request.questions.items():
            result = recorder.results.pop(id(question))
            if result.strategy != recorder.name:
                raise RuntimeError(f"{job.ids[question_id]}: scored by {result.strategy}")
            answer = response.answers[question_id]
            probabilities = [math.exp(lp) for lp in result.logprobs]
            if abs(math.fsum(probabilities) - 1.0) > 1e-6:
                raise RuntimeError(
                    f"{job.ids[question_id]}: probabilities sum to {sum(probabilities)}"
                )
            value: dict[str, Any]
            if isinstance(answer, ChoiceAnswer):
                value = {"choice": answer.choice}
            elif isinstance(answer, ScoreAnswer):
                value = {"score": answer.score}
            elif isinstance(answer, NoulAnswer):
                value = {"noul": answer.noul}
            records.append(
                {
                    "part": job.part[0],
                    "strategy": job.part[1],
                    "dataset": job.part[2],
                    "id": job.ids[question_id],
                    "type": question.type,
                    "gold": job.gold,
                    "keys": list(result.keys),
                    "answer": value,
                    "probabilities": probabilities,
                    "raw_logprobs": _finite(result.raw_logprobs),
                    "observed_mass": result.observed_mass,
                    "missing": list(result.missing),
                    "truncated": result.truncated,
                    "warnings": list(result.warnings),
                    "prompt_tokens": result.prompt_tokens,
                    "cached_tokens": result.cached_tokens,
                    "n_backend_calls": result.n_backend_calls,
                    "latency_ms": latency_ms,
                }
            )
        return records

    if concurrency == 1:
        nested = [await one(job) for job in jobs]
    else:
        nested = list(await asyncio.gather(*(one(job) for job in jobs)))
    return [record for records in nested for record in records]


async def template_check(backend: VLLMHTTPBackend) -> dict[str, Any]:
    """Thinking is off by server default, and a trailing prefill space survives: the template
    may trim it, the adapter then appends the space's own token."""
    body = {
        "model": backend.model,
        "messages": [{"role": "user", "content": "Hi"}],
        "add_generation_prompt": True,
        "add_special_tokens": False,
    }
    ids = (await backend.request_json("POST", "/tokenize", body))["tokens"]
    render = await backend.detokenize(ids)
    user = ChatMessage("user", "Rate it.")
    spaced = RenderedPrompt((user, ChatMessage("assistant", "Answer: ")), "quant_compare/check")
    bare = RenderedPrompt((user, ChatMessage("assistant", "Answer:")), "quant_compare/check")
    raw = {
        "model": backend.model,
        "messages": [
            {"role": "user", "content": "Rate it."},
            {"role": "assistant", "content": "Answer: "},
        ],
        "continue_final_message": True,
        "add_generation_prompt": False,
        "add_special_tokens": False,
    }
    template_spaced = (await backend.request_json("POST", "/tokenize", raw))["tokens"]
    adapter_spaced, adapter_bare = (
        await backend.tokenize_chat(spaced),
        await backend.tokenize_chat(bare),
    )
    space = await backend.tokenize(" ")
    return {
        "generation_prompt_tail": render[-40:],
        "thinking_off": render.endswith("<think>\n\n</think>\n\n"),
        "template_trims_trailing_space": template_spaced == adapter_bare,
        "adapter_keeps_trailing_space": adapter_spaced == [*adapter_bare, *space],
    }


async def measure(
    run: Run, exports: Mapping[str, str], banks: Mapping[str, QuestionBank]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    constrained = RecordingStrategy(ConstrainedStrategy())
    echo = RecordingStrategy(EchoStrategy(mode="sum"))
    records: list[dict[str, Any]] = []
    parts: dict[str, Any] = {}
    async with VLLMHTTPBackend(
        exports["JEVEMU_VLLM_URL"],
        exports["JEVEMU_VLLM_MODEL"],
        max_concurrency=run.concurrency,
        timeout=HTTP_TIMEOUT_S,
    ) as backend:
        checks = await template_check(backend)
        plans = [
            (constrained, [job for ds in DATASETS for job in bank_jobs(banks[ds], "constrained")]),
            (constrained, doc_jobs()),
            (echo, bank_jobs(banks["gpqa_diamond"], "echo_sum")),
        ]
        infos = []
        for recorder, jobs in plans:
            emulator = Emulator(
                backend,
                strategy=recorder,
                max_concurrency=run.concurrency,
                include_diagnostics=True,
            )
            infos.append(await emulator.warm_up())
            started = time.monotonic()
            got = await answer_jobs(emulator, recorder, jobs, run.concurrency)
            parts[f"{jobs[0].part[0]}/{recorder.name}"] = {
                "records": len(got),
                "seconds": round(time.monotonic() - started, 1),
                "median_latency_ms": statistics.median(r["latency_ms"] for r in got),
            }
            records += got
        info = infos[0]
        caps = backend.capabilities
    meta = {
        "backend_info": dataclasses.asdict(info),
        "capabilities": dataclasses.asdict(caps),
        "template_check": checks,
        "parts": parts,
    }
    return records, meta


def run_one(run: Run, out: Path, banks: Mapping[str, QuestionBank], limit: int | None) -> None:
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stop_server()
    exports, startup_s = start_server(run.preset)
    try:
        os.environ.update(exports)  # the emulator's capability probe keys on these
        records, meta = asyncio.run(measure(run, exports, banks))
        gpu = gpu_facts()
        log = server_log()
    finally:
        stop_server()
    (out / f"{run.name}.log").write_text(log)
    part = out / f"{run.name}.jsonl.partial"
    part.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    part.replace(out / f"{run.name}.jsonl")
    meta |= {
        "run": dataclasses.asdict(run),
        "limit": limit,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exports": exports,
        "preset_file": (PRESETS / f"{run.preset}.env").read_text(),
        "startup_seconds": round(startup_s, 1),
        "server_log": server_log_facts(log),
        "gpu_after_run": gpu,
        "banks": {ds: bank.metadata.model_dump(mode="json") for ds, bank in banks.items()},
        "versions": {
            "python": platform.python_version(),
            "jevemu": metadata.version("jevemu"),
            "numpy": np.__version__,
        },
    }
    (out / f"{run.name}.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def cmd_run(args: argparse.Namespace) -> int:
    out: Path = args.out or ROOT / "runs" / "quant" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    out.mkdir(parents=True, exist_ok=True)
    selected = [RUNS_BY_NAME[name] for name in args.only] if args.only else list(RUNS)
    banks = {
        ds: build_question_bank(ds, seed=SEED, idk=True, order="evaluate_idk", limit=args.limit)
        for ds in DATASETS
    }
    for run in selected:
        if (out / f"{run.name}.json").exists():
            print(f"{run.name}: finished earlier, skipped", flush=True)
            continue
        print(f"{run.name}: serving {run.preset} (concurrency {run.concurrency})", flush=True)
        started = time.monotonic()
        run_one(run, out, banks, args.limit)
        print(f"{run.name}: done in {time.monotonic() - started:.0f}s", flush=True)
    print(out)
    return 0


# --- analysis ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Loaded:
    run: Run
    meta: dict[str, Any]
    records: dict[tuple[str, str, str, str], dict[str, Any]]
    """(part, strategy, dataset, id) -> record."""


def load(out: Path, run: Run) -> Loaded | None:
    meta_path = out / f"{run.name}.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    records = {}
    for line in (out / f"{run.name}.jsonl").read_text().splitlines():
        record = json.loads(line)
        record["raw_logprobs"] = _unfinite(record["raw_logprobs"])
        records[(record["part"], record["strategy"], record["dataset"], record["id"])] = record
    return Loaded(run, meta, records)


@dataclass(frozen=True)
class Slice:
    title: str
    strategy: str
    datasets: tuple[str, ...]


SLICES = (
    Slice("GPQA-Diamond, constrained (S2)", "constrained", ("gpqa_diamond",)),
    Slice("LEXam-en, constrained (S2)", "constrained", ("lexam_en",)),
    Slice("Both banks, constrained (S2)", "constrained", ("gpqa_diamond", "lexam_en")),
    Slice("GPQA-Diamond, echo sum (S4)", "echo_sum", ("gpqa_diamond",)),
)


def slice_records(loaded: Loaded, sl: Slice) -> list[dict[str, Any]]:
    rows = [
        r
        for (part, strategy, dataset, _), r in loaded.records.items()
        if part == "bank" and strategy == sl.strategy and dataset in sl.datasets
    ]
    return sorted(rows, key=lambda r: (r["dataset"], r["id"]))


@dataclass(frozen=True)
class Arrays:
    ids: list[str]
    probs: np.ndarray[Any, np.dtype[np.float64]]
    raw: np.ndarray[Any, np.dtype[np.float64]]
    mass: np.ndarray[Any, np.dtype[np.float64]]
    gold: np.ndarray[Any, np.dtype[np.int64]]
    choice: list[str]
    gold_key: list[str]
    idk_key: list[str]
    prompt_tokens: list[int]


def arrays(rows: Sequence[dict[str, Any]]) -> Arrays:
    return Arrays(
        ids=[r["id"] for r in rows],
        probs=np.array([r["probabilities"] for r in rows], dtype=np.float64),
        raw=np.array([r["raw_logprobs"] for r in rows], dtype=np.float64),
        mass=np.array([r["observed_mass"] for r in rows], dtype=np.float64),
        gold=np.array([r["keys"].index(r["gold"]) for r in rows], dtype=np.int64),
        choice=[r["answer"]["choice"] for r in rows],
        gold_key=[r["gold"] for r in rows],
        idk_key=[r["keys"][-1] for r in rows],
        prompt_tokens=[r["prompt_tokens"] for r in rows],
    )


def absolute(a: Arrays) -> dict[str, float]:
    correct = np.array([c == g for c, g in zip(a.choice, a.gold_key, strict=True)])
    idk = np.array([c == k for c, k in zip(a.choice, a.idk_key, strict=True)])
    top = a.probs.max(axis=1)
    return {
        "n": float(len(a.ids)),
        "accuracy": accuracy(a.choice, a.gold_key),
        "answered_accuracy": float(correct[~idk].mean()) if (~idk).any() else math.nan,
        "idk_rate": float(idk.mean()),
        "p_gold": float(a.probs[np.arange(len(a.ids)), a.gold].mean()),
        "nll": float(nll(a.probs[np.arange(len(a.ids)), a.gold]).mean()),
        "brier": float(brier(a.probs, a.gold).mean()),
        "ece_mass": ece(top, correct),
        "ece_width": ece(top, correct, binning="equal_width"),
        "confidence": float(top.mean()),
    }


def paired(ref: Arrays, other: Arrays) -> dict[str, Any]:
    if ref.ids != other.ids:
        raise ValueError("paired runs cover different items")
    n = len(ref.ids)
    idx = np.arange(n)
    ok_ref = np.array([c == g for c, g in zip(ref.choice, ref.gold_key, strict=True)], dtype=float)
    ok = np.array([c == g for c, g in zip(other.choice, other.gold_key, strict=True)], dtype=float)
    nll_ref, nll_q = nll(ref.probs[idx, ref.gold]), nll(other.probs[idx, other.gold])
    brier_ref, brier_q = brier(ref.probs, ref.gold), brier(other.probs, other.gold)
    kl = kl_divergence(ref.probs, other.probs)
    finite = np.isfinite(ref.raw) & np.isfinite(other.raw)
    return {
        "d_accuracy": paired_bootstrap_ci(ok, ok_ref),
        "mcnemar": mcnemar_exact(ok, ok_ref),
        "d_nll": paired_bootstrap_ci(nll_q, nll_ref),
        "d_brier": paired_bootstrap_ci(brier_q, brier_ref),
        "agreement": float(
            np.mean([a == b for a, b in zip(ref.choice, other.choice, strict=True)])
        ),
        "kl_mean": float(kl.mean()),
        "kl_median": float(np.median(kl)),
        "kl_p95": float(np.quantile(kl, 0.95)),
        "kl_max": float(kl.max()),
        "tv_mean": float(total_variation(ref.probs, other.probs).mean()),
        "d_p_gold": float(np.abs(other.probs[idx, other.gold] - ref.probs[idx, ref.gold]).mean()),
        "raw_mae": float(np.abs(other.raw - ref.raw)[finite].mean()),
        "raw_nonfinite": int((~finite).sum()),
        "d_mass": float((other.mass - ref.mass).mean()),
        "abs_d_mass": float(np.abs(other.mass - ref.mass).mean()),
        "prompt_tokens_equal": ref.prompt_tokens == other.prompt_tokens,
    }


def doc_deltas(ref: Loaded, other: Loaded) -> dict[str, Any]:
    pairs = [(r, other.records[key]) for key, r in ref.records.items() if key[0] == "doc"]
    choice = [(a, b) for a, b in pairs if a["type"] == "choice"]
    score = [
        abs(b["answer"]["score"] - a["answer"]["score"]) for a, b in pairs if a["type"] == "score"
    ]
    noul = [abs(b["answer"]["noul"] - a["answer"]["noul"]) for a, b in pairs if a["type"] == "noul"]
    tv = [float(total_variation([a["probabilities"]], [b["probabilities"]])[0]) for a, b in choice]
    return {
        "choice_n": len(choice),
        "choice_agreement": float(np.mean([a["answer"] == b["answer"] for a, b in choice])),
        "choice_tv_mean": float(np.mean(tv)),
        "choice_tv_max": float(np.max(tv)),
        "score_n": len(score),
        "score_mean": float(np.mean(score)),
        "score_max": float(np.max(score)),
        "noul_n": len(noul),
        "noul_mean": float(np.mean(noul)),
        "noul_max": float(np.max(noul)),
    }


def fmt(x: float, digits: int = 3) -> str:
    """Fixed decimals, or 2 significant digits in scientific notation for tiny values."""
    if math.isnan(x):
        return "n/a"
    if x == 0.0:
        return "0"
    if abs(x) < 10**-digits:
        return f"{x:.1e}"
    return f"{x:.{digits}f}"


def signed(x: float, digits: int = 3) -> str:
    return ("+" if x > 0 else "") + fmt(x, digits)


def ci(value: BootstrapCI) -> str:
    return f"{signed(value.estimate)} [{signed(value.low)}, {signed(value.high)}]"


def pvalue(p: float) -> str:
    return "1" if p >= 0.9995 else f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def kernel_summary(meta: Mapping[str, Any]) -> str:
    lines: list[str] = meta["server_log"]["kernel_lines"]
    found = []
    for line in lines:
        if match := re.search(r"Using (\w+) for (\w+)", line):
            found.append(f"{match.group(1)} ({match.group(2)})")
        elif "Weight-only FP8" in line or ("FP8" in line and "Marlin" in line):
            found.append("FP8 weight-only Marlin (no native FP8 on this GPU)")
    method = meta["server_log"].get("quantization", "n/a")
    if method == "None":
        return "none (bf16)"
    unique = list(dict.fromkeys(found))
    return f"`{method}`: " + ("; ".join(unique) if unique else "no kernel line logged")


def table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def ms_per_question(item: Loaded, part: str) -> str:
    """Median latency at concurrency 1; wall time per question under concurrency (latency
    then includes queueing)."""
    stats = item.meta["parts"][part]
    if item.run.concurrency == 1:
        return f"{stats['median_latency_ms']:.0f}"
    return f"{stats['seconds'] * 1000 / stats['records']:.0f} (wall / n)"


def setup_section(out: Path, ref: Loaded, shown: Sequence[Loaded]) -> list[str]:
    meta = ref.meta
    gpu, log = meta["gpu_after_run"], meta["server_log"]
    fixtures = {key[3].split(":")[0] for key in ref.records if key[0] == "doc"}
    banks = ", ".join(
        f"`{ds}` {bank['n_items']} items (sha256 `{bank['items_sha256'][:12]}`)"
        for ds, bank in meta["banks"].items()
    )
    rows = []
    for item in shown:
        m, facts = item.meta, item.meta["server_log"]
        exports = m["exports"]
        rows.append(
            (
                f"`{item.run.name}`",
                item.run.label,
                f"`{exports['JEVEMU_VLLM_MODEL']}` @ "
                f"`{exports['JEVEMU_VLLM_MODEL_REVISION'][:10]}`",
                kernel_summary(m),
                f"{facts.get('weights_gib', math.nan):.2f}",
                f"{facts.get('kv_cache_gib', math.nan):.2f} GiB, "
                f"{int(facts.get('kv_cache_tokens', 0)):,} tokens",
                f"{m['startup_seconds']:.0f}",
                str(item.run.concurrency),
                f"{ms_per_question(item, 'bank/constrained')} / "
                f"{ms_per_question(item, 'bank/echo_sum')}",
            )
        )
    return [
        "### Setup",
        "",
        f"- Data: `{out.relative_to(ROOT)}` (gitignored; per-item outputs stay there).",
        f"- vLLM `{meta['backend_info']['engine_version']}`, image "
        f"`{meta['exports']['JEVEMU_VLLM_IMAGE']}`.",
        f"- GPU: {gpu['name']}, {gpu['memory.total']}, driver {gpu['driver_version']}.",
        "- `bf16` launch (the other 9B runs change only the model, revision or dtype): "
        f"`{log.get('serve_command', 'n/a')}`.",
        f"- Banks: seed {SEED}, IDK option, evaluate-idk order; {banks}.",
        f"- Doc examples: every question of the {len(fixtures)} Jev doc request fixtures.",
        "",
        *table(
            (
                "Run",
                "What",
                "Checkpoint @ revision",
                "Quantization: linear kernel (vLLM log)",
                "Weights GiB",
                "KV cache",
                "Startup s",
                "Concurrency",
                "ms/question S2 / S4",
            ),
            rows,
        ),
        "",
    ]


def absolute_section(shown: Sequence[Loaded]) -> list[str]:
    lines = [
        "### Accuracy and calibration",
        "",
        "Accuracy counts IDK (E) as wrong; A-D accuracy excludes items answered E. NLL clips "
        "p(gold) at 1e-6. Brier is the multiclass sum over A-E. ECE is top-label with "
        f"{DEFAULT_ECE_BINS} equal-mass bins (design default) and {DEFAULT_ECE_BINS} equal-width "
        "bins; confidence is the top-label probability.",
        "",
    ]
    header = (
        "Run",
        "n",
        "Accuracy",
        "A-D accuracy",
        "IDK rate",
        "Mean p(gold)",
        "NLL",
        "Brier",
        "ECE (mass)",
        "ECE (width)",
        "Mean confidence",
    )
    keys = ["accuracy", "answered_accuracy", "idk_rate", "p_gold", "nll", "brier"]
    keys += ["ece_mass", "ece_width", "confidence"]
    for sl in SLICES:
        rows = []
        for item in shown:
            m = absolute(arrays(slice_records(item, sl)))
            rows.append((f"`{item.run.name}`", f"{m['n']:.0f}", *(f"{m[k]:.3f}" for k in keys)))
        lines += [f"**{sl.title}**", "", *table(header, rows), ""]
    return lines


def paired_sections(ref: Loaded, compared: Sequence[Loaded]) -> list[str]:
    deltas = {
        (sl.title, item.run.name): paired(
            arrays(slice_records(ref, sl)), arrays(slice_records(item, sl))
        )
        for sl in SLICES
        for item in compared
    }
    scores = [
        "### Paired with bf16: accuracy and proper scores",
        "",
        "Differences are run minus `bf16` over the same items, with 95% paired-bootstrap "
        "intervals (10,000 resamples, seed 0; percentile below 500 items, BCa from 500). McNemar "
        "is exact; b/c = items only this run / only `bf16` got right.",
        "",
    ]
    probs = [
        "### Paired with bf16: probabilities",
        "",
        "Per item over the A-E distribution: KL(bf16 || run) in nats, total variation, "
        "|Δ p(gold)|. Raw label logprobs are before renormalization (constrained: the merged `A` "
        "and ` A` surfaces; echo: the ` A`..` E` continuation totals); observed mass is the label "
        "mass before renormalization.",
        "",
    ]
    for sl in SLICES:
        score_rows, prob_rows = [], []
        for item in compared:
            d = deltas[(sl.title, item.run.name)]
            mc = d["mcnemar"]
            score_rows.append(
                (
                    f"`{item.run.name}`",
                    ci(d["d_accuracy"]),
                    f"{pvalue(mc.p_value)} ({mc.only_a}/{mc.only_b})",
                    ci(d["d_nll"]),
                    ci(d["d_brier"]),
                    f"{d['agreement']:.3f}",
                )
            )
            nonfinite = f" ({d['raw_nonfinite']} non-finite)" if d["raw_nonfinite"] else ""
            prob_rows.append(
                (
                    f"`{item.run.name}`",
                    *(fmt(d[k], 4) for k in ("kl_mean", "kl_median", "kl_p95", "kl_max")),
                    fmt(d["tv_mean"], 4),
                    fmt(d["d_p_gold"], 4),
                    fmt(d["raw_mae"], 4) + nonfinite,
                    f"{signed(d['d_mass'], 4)} / {fmt(d['abs_d_mass'], 4)}",
                )
            )
        score_header = ["Run", "Δ accuracy [95% CI]", "McNemar p (b/c)", "Δ NLL [95% CI]"]
        score_header += ["Δ Brier [95% CI]", "Top-1 agreement"]
        prob_header = ["Run", "KL mean", "KL median", "KL p95", "KL max", "TV mean"]
        prob_header += ["Mean \\|Δ p(gold)\\|", "Raw logprob MAE", "Δ observed mass (mean / abs)"]
        scores += [f"**{sl.title}**", "", *table(score_header, score_rows), ""]
        probs += [f"**{sl.title}**", "", *table(prob_header, prob_rows), ""]
    same = [
        f"`{item.run.name}`"
        for item in compared
        if all(deltas[(sl.title, item.run.name)]["prompt_tokens_equal"] for sl in SLICES)
    ]
    checks = [
        "- Prompt token counts equal to `bf16` on every bank item (same template and tokenizer): "
        + (", ".join(same) or "none")
    ]
    return scores + probs + ["### Checks", "", *checks]


def docs_section(ref: Loaded, compared: Sequence[Loaded]) -> list[str]:
    docs = {item.run.name: doc_deltas(ref, item) for item in compared}
    n = next(iter(docs.values()))
    header = (
        "Run",
        f"Choice (n={n['choice_n']}): agreement, TV mean / max",
        f"Score (n={n['score_n']}): \\|Δ expected score\\| mean / max",
        f"Noul (n={n['noul_n']}): \\|Δ P(yes)\\| mean / max",
    )
    rows = [
        (
            f"`{name}`",
            f"{d['choice_agreement']:.3f}, {fmt(d['choice_tv_mean'], 4)} / "
            f"{fmt(d['choice_tv_max'], 4)}",
            f"{fmt(d['score_mean'], 4)} / {fmt(d['score_max'], 4)}",
            f"{fmt(d['noul_mean'], 4)} / {fmt(d['noul_max'], 4)}",
        )
        for name, d in docs.items()
    ]
    return ["### Jev doc examples (constrained), paired with bf16", "", *table(header, rows), ""]


LOGIT_STEP = 0.125
"""bf16 spacing of logits in [16, 32): label logprob differences land on multiples of it."""


def grid_share(loaded: Loaded, strategy: str, *, tol: float = 2e-3) -> float:
    """Share of bank items whose finite raw label logprobs differ by multiples of
    :data:`LOGIT_STEP` (within ``tol``): the resolution the reported probabilities move in."""
    hits = total = 0
    for (part, strat, _, _), record in loaded.records.items():
        if part != "bank" or strat != strategy:
            continue
        raw = np.array([x for x in record["raw_logprobs"] if math.isfinite(x)])
        steps = (raw - raw[0]) / LOGIT_STEP
        hits += bool(np.all(np.abs(steps - np.round(steps)) * LOGIT_STEP < tol))
        total += 1
    return hits / total if total else math.nan


def run_checks(shown: Sequence[Loaded]) -> list[str]:
    lines = []
    for item in shown:
        tc, caps = item.meta["template_check"], item.meta["capabilities"]
        cached = sum(r["cached_tokens"] or 0 for r in item.records.values())
        lines.append(
            f"- `{item.run.name}`: thinking off {tc['thinking_off']}; template trims `Answer: ` "
            f"{tc['template_trims_trailing_space']}, adapter keeps the space "
            f"{tc['adapter_keeps_trailing_space']}; probed top-logprobs cap "
            f"{caps['top_logprobs_max']}, mask reflected {caps['mask_reflected_in_logprobs']}; "
            f"prefix-cache hits {cached:,} tokens; label logprob differences on the "
            f"{LOGIT_STEP}-nat grid: S2 {grid_share(item, 'constrained'):.3f}, "
            f"S4 {grid_share(item, 'echo_sum'):.3f} of items"
        )
    return lines


def render(out: Path, runs: Mapping[str, Loaded]) -> str:
    ref = runs[REFERENCE]
    compared = [runs[r.name] for r in RUNS if r.name in runs and r.role in ("floor", "quant")]
    shown = [runs[r.name] for r in RUNS if r.name in runs]
    lines = [
        BEGIN,
        "",
        "<!-- Generated by `uv run python scripts/quant_compare.py analyze "
        f"{out.relative_to(ROOT)}`; edit the prose outside this block. -->",
        "",
        *setup_section(out, ref, shown),
        *absolute_section(shown),
        *docs_section(ref, compared),
        *paired_sections(ref, compared),
        *run_checks(shown),
        "",
        END,
    ]
    return "\n".join(lines) + "\n"


def cmd_analyze(args: argparse.Namespace) -> int:
    out: Path = args.out.resolve()
    runs = {run.name: loaded for run in RUNS if (loaded := load(out, run)) is not None}
    if REFERENCE not in runs:
        raise SystemExit(f"{out}: no finished {REFERENCE} run")
    limits = {loaded.meta["limit"] for loaded in runs.values()}
    if len(limits) != 1:
        raise SystemExit(f"runs used different --limit values: {sorted(map(str, limits))}")
    block = render(out, runs)
    report: Path = args.report
    text = report.read_text() if report.exists() else f"# Quantization report\n\n{BEGIN}\n{END}\n"
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{report}: missing {BEGIN} / {END} markers")
    head, rest = text.split(BEGIN, 1)
    tail = rest.split(END, 1)[1]
    report.write_text(head + block.rstrip("\n") + tail)
    print(f"wrote {report} from {len(runs)} runs in {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="serve and measure each run (resumable)")
    run.add_argument("--out", type=Path, help="run directory (default: new runs/quant/<UTC ts>)")
    run.add_argument("--only", nargs="+", choices=sorted(RUNS_BY_NAME), help="runs to measure")
    run.add_argument("--limit", type=int, help="first N questions per bank (smoke tests only)")
    analyze = sub.add_parser("analyze", help="write the report tables from a run directory")
    analyze.add_argument("out", type=Path)
    analyze.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    return cmd_run(args) if args.command == "run" else cmd_analyze(args)


if __name__ == "__main__":
    sys.exit(main())
