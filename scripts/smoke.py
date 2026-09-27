"""Smoke check: Jev and the emulator answer the same 20 LEXam-en bank questions end to end.

    eval "$(scripts/serve_vllm.sh | grep '^export ')"
    uv run python scripts/smoke.py               # --max-usd 0.5 --limit 20 --out runs/smoke

Builds the LEXam-en bank (seed 0, with the "I don't know" option, evaluate-idk option order,
first ``--limit`` questions) and sends every item to

- the real Jev API (``JevClient``: pinned ``jev-1.13.0``, default on-disk response cache so a
  rerun costs nothing, spend capped by ``--max-usd``; ``TYPESAFE_API_KEY`` from the environment
  or ``.env``), four requests in flight, and
- the emulator on the vLLM server at ``JEVEMU_VLLM_URL`` serving ``JEVEMU_VLLM_MODEL``, one
  request at a time so each latency is a single request's.

Every response must be a ``SystemOneResponse`` whose ``answer`` is a Choice answer with
probabilities over exactly the item's options (A..E) summing to 1 (within Jev's rounding).
Writes ``<out>/<UTC timestamp>/``: ``bank.jsonl``, ``results.jsonl`` (one line per item and
system: item_id, gold, system, model, choice, probabilities, confidence, latency_ms) and
``summary.md``. Jev latency is the HTTP round trip recorded when the response was first
fetched (also for cache hits); emulator latency is the client-measured ``system_one`` call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from jevemu.backends.base import BackendInfo
from jevemu.backends.vllm_http import VLLMHTTPBackend
from jevemu.emulator import Emulator
from jevemu.eval.bank import QUESTION_KEY, BankItem, QuestionBank, build_question_bank
from jevemu.jev_client import JevClient, load_env_file
from jevemu.types import ChoiceAnswer, SystemOneResponse

DATASET = "lexam_en"
SEED = 0
JEV_CONCURRENCY = 4


class SmokeFailure(RuntimeError):
    """A response failed validation."""


@dataclass(frozen=True)
class Result:
    item: BankItem
    system: str
    model: str
    answer: ChoiceAnswer
    latency_ms: float

    def record(self) -> dict[str, object]:
        options = list(self.item.question.criteria)
        return {
            "item_id": self.item.item_id,
            "gold": self.item.gold,
            "system": self.system,
            "model": self.model,
            "choice": self.answer.choice,
            "probabilities": {key: self.answer.probabilities[key] for key in options},
            "confidence": self.answer.confidence,
            "latency_ms": self.latency_ms,
        }


def validate(item: BankItem, response: SystemOneResponse, system: str) -> ChoiceAnswer:
    where = f"{system} on {item.item_id}"
    answer = response.answers.get(QUESTION_KEY)
    if not isinstance(answer, ChoiceAnswer):
        raise SmokeFailure(f"{where}: no Choice answer under {QUESTION_KEY!r}: {answer!r}")
    options = set(item.question.criteria)
    if set(answer.probabilities) != options:
        raise SmokeFailure(f"{where}: probabilities over {sorted(answer.probabilities)}")
    total = math.fsum(answer.probabilities.values())
    if abs(total - 1.0) > 0.03:  # Jev rounds each probability to 0.01
        raise SmokeFailure(f"{where}: probabilities sum to {total}")
    if answer.choice not in options:
        raise SmokeFailure(f"{where}: choice {answer.choice!r} is not an option")
    return answer


async def run_jev(bank: QuestionBank, max_usd: float) -> tuple[list[Result], dict[str, object]]:
    load_env_file()
    limit = asyncio.Semaphore(JEV_CONCURRENCY)
    async with JevClient(max_usd=max_usd) as client:

        async def ask(item: BankItem) -> Result:
            async with limit:
                response, record = await client.system_one_with_meta(item.request)
            answer = validate(item, response, "jev")
            return Result(item, "jev", response.model, answer, record.latency_ms)

        results = await asyncio.gather(*(ask(item) for item in bank.items))
        accounting: dict[str, object] = {
            "spent_usd": client.spent_usd,
            "network_calls": client.network_calls,
            "cache_hits": client.cache_hits,
        }
    return list(results), accounting


async def run_emulator(
    bank: QuestionBank, url: str, model: str
) -> tuple[list[Result], BackendInfo, dict[str, int]]:
    async with VLLMHTTPBackend(url, model) as backend:
        emulator = Emulator(backend, include_diagnostics=True)
        info = await emulator.warm_up()
        results: list[Result] = []
        strategies: dict[str, int] = {}
        for item in bank.items:
            started = time.perf_counter()
            response = await emulator.system_one(item.request)
            latency_ms = (time.perf_counter() - started) * 1000.0
            answer = validate(item, response, "emulator")
            results.append(Result(item, "emulator", response.model, answer, latency_ms))
            for diagnostics in (response.x_jevemu or {}).values():
                strategies[diagnostics.strategy] = strategies.get(diagnostics.strategy, 0) + 1
    return results, info, strategies


def idk_key(item: BankItem) -> str | None:
    return list(item.question.criteria)[-1] if item.idk else None


def metrics(results: list[Result]) -> dict[str, float]:
    n = len(results)
    correct = [r.answer.choice == r.item.gold for r in results]
    idk = [r.answer.choice == idk_key(r.item) for r in results]
    answered = [c for c, i in zip(correct, idk, strict=True) if not i]
    return {
        "n": n,
        "accuracy": sum(correct) / n,
        "accuracy_answered": sum(answered) / len(answered) if answered else math.nan,
        "idk_rate": sum(idk) / n,
        "mean_p_gold": statistics.fmean(r.answer.probabilities[r.item.gold] for r in results),
        "mean_latency_ms": statistics.fmean(r.latency_ms for r in results),
        "median_latency_ms": statistics.median(r.latency_ms for r in results),
    }


def summary(
    bank: QuestionBank,
    jev: list[Result],
    emulator: list[Result],
    accounting: dict[str, object],
    info: BackendInfo,
    strategies: dict[str, int],
) -> str:
    rows = [("jev", jev[0].model, metrics(jev)), ("emulator", emulator[0].model, metrics(emulator))]
    agree = sum(a.answer.choice == b.answer.choice for a, b in zip(jev, emulator, strict=True))
    lines = [
        "# Smoke: Jev vs emulator on LEXam-en",
        "",
        f"- Bank: `{bank.metadata.dataset}`, seed {SEED}, idk=True, order `evaluate_idk`, "
        f"{len(bank.items)} items, items_sha256 `{bank.metadata.items_sha256[:16]}`",
        f"- Emulator: `{info.model}` (revision `{info.model_revision}`) on {info.backend} "
        f"{info.engine_version}, image `{info.flags.get('image', 'unknown')}`; strategies "
        + ", ".join(f"{name} x{count}" for name, count in sorted(strategies.items())),
        f"- Jev: `{jev[0].model}`; this run spent ${accounting['spent_usd']:.6f} "
        f"({accounting['network_calls']} network calls, {accounting['cache_hits']} cache hits)",
        f"- Choice agreement Jev vs emulator: {agree}/{len(jev)}",
        "",
        "| System | Model | Accuracy (all) | Accuracy on A-D answers | IDK rate "
        "| Mean p(gold) | Mean latency (ms) | Median latency (ms) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for system, model, m in rows:
        lines.append(
            f"| {system} | `{model}` | {m['accuracy']:.2f} | {m['accuracy_answered']:.2f} "
            f"| {m['idk_rate']:.2f} | {m['mean_p_gold']:.3f} | {m['mean_latency_ms']:.1f} "
            f"| {m['median_latency_ms']:.1f} |"
        )
    lines += [
        "",
        "Accuracy on A-D answers excludes items answered with the IDK option (E). Jev latency is "
        "the recorded HTTP round trip; emulator latency is one sequential `system_one` call.",
    ]
    return "\n".join(lines) + "\n"


async def main_async(args: argparse.Namespace) -> Path:
    bank = build_question_bank(DATASET, seed=SEED, idk=True, order="evaluate_idk", limit=args.limit)
    jev, accounting = await run_jev(bank, args.max_usd)
    emulator, info, strategies = await run_emulator(bank, args.url, args.model)

    out = args.out / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    bank.to_jsonl(out / "bank.jsonl")
    with (out / "results.jsonl").open("w", encoding="utf-8") as fh:
        for result in (*jev, *emulator):
            fh.write(json.dumps(result.record(), ensure_ascii=False) + "\n")
    (out / "summary.md").write_text(
        summary(bank, jev, emulator, accounting, info, strategies), encoding="utf-8"
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=os.environ.get("JEVEMU_VLLM_URL", "http://localhost:8000"))
    parser.add_argument("--model", default=os.environ.get("JEVEMU_VLLM_MODEL", "Qwen/Qwen3-0.6B"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-usd", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=Path("runs/smoke"))
    out = asyncio.run(main_async(parser.parse_args()))
    print((out / "summary.md").read_text(encoding="utf-8"))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
