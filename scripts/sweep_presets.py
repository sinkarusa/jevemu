"""Answer benchmark splits with several vLLM presets in turn, then rank the run directories.

    # Serve each preset, run it on the screen split, stop the server; resumable
    uv run python scripts/sweep_presets.py run --presets qwen3.8-27b-int4-redhat,qwen3-14b-int4 \\
        --split screen --out runs/select --strategy auto_single --concurrency 16

    # Ranked markdown tables (macro accuracy, per benchmark, paired deltas vs the best and Jev)
    uv run python scripts/sweep_presets.py report --presets ... --split screen --out runs/select \\
        --strategy auto_single --reference runs/select/jev

``run`` handles one preset at a time on the single GPU: ``docker compose down``,
``scripts/serve_vllm.sh --preset NAME`` (its ``export`` line becomes the runner's environment),
``scripts/run_split.py run`` into ``<out>/<preset>.<strategy>/`` (``.<layout>`` appended unless
the layout is ``question_first``; ``--renderer default`` is ``state_first``), ``docker compose
down``. A preset whose run directory already holds a complete run of every requested benchmark
is skipped without starting a server, so rerunning the same command resumes the sweep; a
partially answered preset resumes inside ``run_split``. A preset that fails to start or whose
run fails is recorded and the sweep moves on; the server is stopped whatever happens. Server
and runner output go to
``<out>/sweep/<preset>.<split>.{serve,run}.log`` and one JSON line per preset attempt to
``<out>/sweep/sweep.<split>.jsonl`` (status, seconds to healthy, run seconds, runner exit code).

``report`` ranks the run directories by macro accuracy (ties broken by macro NLL) and prints
the ranking (macro accuracy and NLL with 95% stratified bootstrap intervals, items/s, paired
macro deltas against the best candidate and ``--reference`` on shared items), then per-benchmark
accuracy, NLL, ECE and IDK rate tables. Items/s is answered items over the summed wall time of
the runner invocations that answered them, so server startup is excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jevemu.bench.compare import Comparison, RunSummary, compare, summarize
from jevemu.bench.manifest import RunManifest, run_key
from jevemu.eval.item_scores import Interval, ScoreSummary
from jevemu.eval.metrics import DEFAULT_BOOTSTRAP_RESAMPLES
from jevemu.eval.splits import DATASETS, SPLITS, SplitName
from jevemu.render import DEFAULT_LAYOUT

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ("docker", "compose", "-f", str(ROOT / "docker" / "vllm" / "compose.yaml"))
SERVE_TIMEOUT_S = 1200
SWEEP_DIR = "sweep"


def system_id(preset: str, strategy: str, renderer: str) -> str:
    layout = DEFAULT_LAYOUT if renderer == "default" else renderer
    # No suffix means question_first, as the Stage 1 run directories are named.
    parts = [preset, strategy] + ([] if layout == "question_first" else [layout])
    return ".".join(parts)


def csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def is_complete(run_dir: Path, benchmarks: Sequence[str], split: SplitName) -> bool:
    manifest = RunManifest.load(run_dir)
    if manifest is None:
        return False
    for benchmark in benchmarks:
        entry = manifest.runs.get(run_key(benchmark, split))
        if entry is None or entry.status != "complete" or entry.n_ok != entry.n_items:
            return False
    return True


def stop_server(log: Path | None = None) -> None:
    if log is not None:
        with log.open("a", encoding="utf-8") as out:
            subprocess.run([*COMPOSE, "logs", "--no-color", "vllm"], stdout=out, stderr=out)
    subprocess.run([*COMPOSE, "down"], capture_output=True, check=False)


def serve(preset: str, log: Path) -> dict[str, str]:
    """Start ``preset`` and return the variables its ``export`` line sets; raises on failure."""
    result = subprocess.run(
        [str(ROOT / "scripts" / "serve_vllm.sh"), "--preset", preset],
        capture_output=True,
        text=True,
        timeout=SERVE_TIMEOUT_S,
        check=False,
    )
    with log.open("a", encoding="utf-8") as out:
        out.write(f"# {_now()}: serve_vllm.sh --preset {preset}\n{result.stdout}{result.stderr}")
    if result.returncode != 0:
        raise RuntimeError(f"serve_vllm.sh exited {result.returncode}")
    exports = [line for line in result.stdout.splitlines() if line.startswith("export ")]
    if len(exports) != 1:
        raise RuntimeError("serve_vllm.sh printed no export line")
    env: dict[str, str] = {}
    for word in shlex.split(exports[0])[1:]:
        name, _, value = word.partition("=")
        env[name] = value
    return env


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_preset(args: argparse.Namespace, preset: str, sweep_dir: Path) -> dict[str, Any]:
    sid = system_id(preset, args.strategy, args.renderer)
    entry: dict[str, Any] = {"preset": preset, "system_id": sid, "started_at": _now()}
    if is_complete(args.out / sid, args.benchmarks, args.split):
        entry["status"] = "skipped_complete"
        return entry
    serve_log = sweep_dir / f"{preset}.{args.split}.serve.log"
    run_log = sweep_dir / f"{preset}.{args.split}.run.log"
    stop_server()
    try:
        started = time.monotonic()
        try:
            env = serve(preset, serve_log)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            entry.update(status="serve_failed", error=str(exc))
            return entry
        entry["serve_s"] = round(time.monotonic() - started, 1)
        # Recorded in the manifest's launch_env (not part of the system identity).
        env["JEVEMU_VLLM_PRESET"] = preset
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_split.py"),
            "run",
            "--system=emulator",
            f"--system-id={sid}",
            f"--benchmarks={','.join(args.benchmarks)}",
            f"--split={args.split}",
            f"--out={args.out}",
            f"--strategy={args.strategy}",
            f"--renderer={args.renderer}",
            f"--concurrency={args.concurrency}",
        ]
        code = -1
        run_s = 0.0
        for attempt in range(1, args.attempts + 1):
            started = time.monotonic()
            with run_log.open("a", encoding="utf-8") as out:
                out.write(f"# attempt {attempt} at {_now()}: {shlex.join(command)}\n")
                out.flush()
                code = subprocess.run(
                    command,
                    env={**os.environ, **env},
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    check=False,
                ).returncode
            run_s += time.monotonic() - started
            if code == 0:
                break
        entry.update(
            status="complete" if code == 0 else "run_failed",
            exit_code=code,
            attempts=attempt,
            run_s=round(run_s, 1),
        )
        return entry
    finally:
        stop_server(serve_log)
        entry["finished_at"] = _now()


def cmd_run(args: argparse.Namespace) -> int:
    sweep_dir = args.out / SWEEP_DIR
    sweep_dir.mkdir(parents=True, exist_ok=True)
    status_path = sweep_dir / f"sweep.{args.split}.jsonl"
    failed = []
    for preset in args.presets:
        print(f"{_now()} {preset}: starting", flush=True)
        entry = run_preset(args, preset, sweep_dir)
        with status_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(entry) + "\n")
        print(f"{_now()} {preset}: {json.dumps(entry)}", flush=True)
        if entry["status"] not in ("complete", "skipped_complete"):
            failed.append(preset)
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


# --- report ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Ranked:
    preset: str
    run_dir: Path
    summary: RunSummary
    items_per_s: float


def items_per_second(run_dir: Path, split: str) -> float:
    """Answered items over the summed wall time of the invocations that answered them."""
    manifest = RunManifest.load(run_dir)
    if manifest is None:
        return float("nan")
    called = 0
    seconds = 0.0
    for entry in manifest.runs.values():
        if entry.split != split:
            continue
        for invocation in entry.invocations:
            if invocation.finished_at is None or invocation.n_called == 0:
                continue
            called += invocation.n_called
            seconds += (invocation.finished_at - invocation.started_at).total_seconds()
    return called / seconds if seconds > 0 else float("nan")


def _interval(value: Interval, digits: int = 4) -> str:
    return f"{value.estimate:.{digits}f} [{value.low:.{digits}f}, {value.high:.{digits}f}]"


def _delta(value: Interval, digits: int = 4) -> str:
    return f"{value.estimate:+.{digits}f} [{value.low:+.{digits}f}, {value.high:+.{digits}f}]"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    ranked: list[Ranked] = []
    missing = []
    for preset in args.presets:
        run_dir = args.out / system_id(preset, args.strategy, args.renderer)
        if not is_complete(run_dir, args.benchmarks, args.split):
            missing.append(preset)
            continue
        summary = summarize(run_dir, split=args.split, n_resamples=args.resamples)
        ranked.append(Ranked(preset, run_dir, summary, items_per_second(run_dir, args.split)))
    ranked.sort(key=lambda r: (-r.summary.macro_accuracy.estimate, r.summary.macro_nll.estimate))
    if not ranked:
        print("no complete runs", file=sys.stderr)
        return 1
    best = ranked[0]

    def paired(a: Path, b: Path) -> Comparison:
        return compare(a, b, split=args.split, n_resamples=args.resamples)

    rows = []
    for rank, r in enumerate(ranked, 1):
        vs_best = "" if r is best else _delta(paired(r.run_dir, best.run_dir).macro_delta_accuracy)
        vs_ref = paired(r.run_dir, args.reference) if args.reference else None
        rows.append(
            [
                str(rank),
                f"`{r.preset}`",
                _interval(r.summary.macro_accuracy),
                _interval(r.summary.macro_nll, 3),
                vs_best,
                _delta(vs_ref.macro_delta_accuracy) if vs_ref else "",
                _delta(vs_ref.macro_delta_nll, 3) if vs_ref else "",
                f"{r.items_per_s:.2f}",
                str(r.summary.n_errors),
            ]
        )
    header = [
        "Rank",
        "Preset",
        "Macro acc. [95% CI]",
        "Macro NLL [95% CI]",
        "Δ acc. vs #1 [95% CI]",
        "Δ acc. vs ref [95% CI]",
        "Δ NLL vs ref [95% CI]",
        "Items/s",
        "Errors",
    ]
    print(f"## Ranking ({args.split}, strategy {args.strategy}, renderer {args.renderer})\n")
    print(_table(header, rows))
    reference = summarize(args.reference, split=args.split) if args.reference else None
    columns = [(r.preset, r.summary) for r in ranked]
    if reference is not None:
        columns.append((f"{reference.system_id} (reference)", reference))
    metrics: tuple[tuple[str, Callable[[ScoreSummary], float]], ...] = (
        ("Accuracy", lambda s: s.accuracy.estimate),
        ("NLL", lambda s: s.nll),
        ("ECE", lambda s: s.ece),
    )
    for title, metric in metrics:
        body = [
            [name, *(f"{metric(s.benchmarks[b].scores):.3f}" for b in args.benchmarks)]
            for name, s in columns
        ]
        print(f"## {title} per benchmark\n")
        print(_table(["Preset", *args.benchmarks], body))
    idk = [b for b in args.benchmarks if b.endswith("_idk")]
    if idk:
        body = [
            [name, *(f"{s.benchmarks[b].scores.idk_rate or 0.0:.3f}" for b in idk)]
            for name, s in columns
        ]
        print("## IDK rate\n")
        print(_table(["Preset", *idk], body))
    if missing:
        print(f"not complete (left out): {', '.join(missing)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("run", cmd_run), ("report", cmd_report)):
        sub = commands.add_parser(name)
        sub.add_argument("--presets", type=csv_list, required=True)
        sub.add_argument("--benchmarks", type=csv_list, default=list(DATASETS))
        sub.add_argument("--split", choices=SPLITS, default="screen")
        sub.add_argument("--out", type=Path, default=Path("runs/select"))
        sub.add_argument("--strategy", default="auto_single")
        sub.add_argument("--renderer", default="default")
        sub.set_defaults(handler=handler)
        if name == "run":
            sub.add_argument("--concurrency", type=int, default=16)
            sub.add_argument("--attempts", type=int, default=2, help="runner tries per preset")
        else:
            sub.add_argument("--reference", type=Path, help="run directory to pair with")
            sub.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    args = parser.parse_args()
    code: int = args.handler(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
