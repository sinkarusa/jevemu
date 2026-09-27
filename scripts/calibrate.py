"""Calibration of recorded runs: cross-fitted study, Jev confidence fit, priors and registries.

    # Cross-fitted calibration study of any runs of one split (Jev and emulator alike);
    # every system gets the same calibrators on the same folds. Writes report.md,
    # metrics.json and predictions.<system>.jsonl under --out.
    uv run python scripts/calibrate.py crossfit jev=runs/select/jev \\
        emu=runs/select/<preset>.auto_single.state_first --split holdout --reference jev \\
        --debias none --out runs/calibration/holdout_single

    # Fit Jev's confidence on one split, check it on another
    uv run python scripts/calibrate.py confidence runs/select/jev --fit-split select \\
        --eval-split holdout

    # Label priors for Emulator(debiaser=batch/pride) from an emulator run's raw probabilities
    uv run python scripts/calibrate.py priors runs/select/<emu> --method batch \\
        --out runs/calibration/batch_priors.json

    # A calibrator registry for Emulator(calibrators=...) fitted on a whole emulator run, on the
    # probabilities the deployed debiaser produces: --priors none (no debiaser: the strategy's
    # raw probabilities), --priors PRIORS.json (raw divided by the priors the deployed
    # batch/pride debiaser loads), or by default the run's recorded probabilities
    uv run python scripts/calibrate.py registry runs/select/<emu> --calibrator temperature \\
        --priors runs/calibration/batch_priors.json --out runs/calibration/registry.json

Runs are ``<out>/<system_id>/`` directories of ``scripts/run_split.py``; the frozen split files
are read from the sibling ``splits/`` directory unless ``--splits-dir`` is given. ``crossfit``
arguments are ``NAME=RUN_DIR`` (or a bare directory, named after it).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from jevemu.bench.manifest import RunManifest
from jevemu.bench.runner import load_records
from jevemu.confidence import CONFIDENCE_FUNCTIONS, ConfidenceFn
from jevemu.confidence.fit import (
    ConfidenceSample,
    FitRow,
    evaluate,
    fit_power,
    sample_from_answer,
)
from jevemu.debias import LabelPriors, fit_batch_priors, fit_pride_priors, full_cycle
from jevemu.eval.crossfit import (
    DEBIAS_METHODS,
    DEFAULT_CALIBRATORS,
    DEFAULT_FOLDS,
    DEFAULT_MIN_GROUP,
    CalItem,
    DebiasMethod,
    apply_label_priors,
    fit_registry,
    load_run_items,
)
from jevemu.eval.metrics import DEFAULT_BOOTSTRAP_RESAMPLES
from jevemu.eval.report import DEFAULT_PRIMARY, run_calibration
from jevemu.eval.splits import DATASETS, DROPPED_DATASETS, SPLITS, SplitName

logger = logging.getLogger("calibrate")


def _run_arg(text: str) -> tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep:
        path, name = text, Path(text).name
    if not Path(path).is_dir():
        raise argparse.ArgumentTypeError(f"no run directory {path}")
    return name, Path(path)


def _list(text: str) -> list[str]:
    return [part for part in text.split(",") if part]


def _benchmarks(text: str) -> list[str] | None:
    if text == "all":
        return None
    names = _list(text)
    unknown = sorted(set(names) - set(DATASETS) - set(DROPPED_DATASETS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown benchmarks {unknown}; expected {DATASETS}")
    return names


def _debias(text: str) -> list[DebiasMethod]:
    methods: list[DebiasMethod] = []
    for part in _list(text):
        matched = [m for m in DEBIAS_METHODS if m == part]
        if not matched:
            raise argparse.ArgumentTypeError(f"unknown debias {part!r}; known: {DEBIAS_METHODS}")
        methods.append(matched[0])
    return methods


def _items(args: argparse.Namespace, run_dir: Path, split: SplitName) -> list[CalItem]:
    loaded = load_run_items(
        run_dir, split=split, splits_dir=args.splits_dir, benchmarks=args.benchmarks
    )
    return [item for items in loaded.values() for item in items]


# --- crossfit ----------------------------------------------------------------------------------


def cmd_crossfit(args: argparse.Namespace) -> int:
    runs = dict(args.runs)
    if len(runs) != len(args.runs):
        raise SystemExit("run names must be unique")
    reference = args.reference if args.reference in runs else None
    if args.reference and reference is None:
        raise SystemExit(f"--reference {args.reference!r} is not one of {list(runs)}")
    started = time.perf_counter()
    report = run_calibration(
        runs,
        split=args.split,
        splits_dir=args.splits_dir,
        benchmarks=args.benchmarks,
        debias=args.debias,
        calibrators=args.calibrators,
        primary=args.primary,
        reference=reference,
        n_folds=args.folds,
        seed=args.seed,
        min_group=args.min_group,
        n_resamples=args.resamples,
    )
    report.write(args.out)
    print(report.markdown())
    logger.info("wrote %s in %.1f s", args.out, time.perf_counter() - started)
    return 0


# --- confidence --------------------------------------------------------------------------------


def _samples(run_dir: Path, split: str) -> list[ConfidenceSample]:
    samples = []
    for path in sorted(run_dir.glob(f"*.{split}.jsonl")):
        benchmark = path.name[: -len(f".{split}.jsonl")]
        for record in load_records(path).values():
            if record.answer is not None:
                sample = sample_from_answer(benchmark, record.answer)
                if sample is not None:
                    samples.append(sample)
    return samples


def _fit_table(rows: Sequence[FitRow], candidates: Sequence[str]) -> str:
    groups: list[tuple[str, int | None, int]] = []
    for row in rows:
        if row.candidate == candidates[0]:
            groups.append((row.kind, row.k, row.n))
    by_key: dict[tuple[str, str, int | None], FitRow] = {
        (r.candidate, r.kind, r.k): r for r in rows
    }
    header = ["Kind", "K", "n", *(f"`{c}` MAE" for c in candidates)]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for kind, k, n in groups:
        maes = [by_key[(c, kind, k)].mae for c in candidates]
        best = min(maes)
        cells = [f"**{m:.4f}**" if m == best else f"{m:.4f}" for m in maes]
        label = "all" if k is None else str(k)
        lines.append("| " + " | ".join([kind, label, str(n), *cells]) + " |")
    return "\n".join(lines) + "\n"


def cmd_confidence(args: argparse.Namespace) -> int:
    fit_samples = _samples(args.run, args.fit_split)
    eval_samples = _samples(args.run, args.eval_split) if args.eval_split else fit_samples
    if not fit_samples or not eval_samples:
        raise SystemExit(f"no Choice/Score answers in {args.run}")
    fitted = fit_power(fit_samples, args.base)
    candidates: dict[str, ConfidenceFn] = {**CONFIDENCE_FUNCTIONS, fitted.__name__: fitted}
    print(
        f"fitted {fitted.__name__} on {len(fit_samples)} {args.fit_split} answers: "
        f"gamma = {', '.join(f'{k} {g:.4f}' for k, g in fitted.gammas.items())}\n"
    )
    names = list(candidates)
    for label, samples in (("fit", fit_samples), ("eval", eval_samples)):
        split = args.fit_split if label == "fit" else args.eval_split or args.fit_split
        rows = evaluate(candidates, samples)
        totals = {r.candidate: r for r in rows if r.kind == "all"}
        best = min(totals.values(), key=lambda r: r.mae)
        print(
            f"## {split} ({len(samples)} answers): best `{best.candidate}` "
            f"MAE {best.mae:.4f}, exact {best.exact:.3f}, within 0.01 {best.within_001:.3f}, "
            f"bias {best.bias:+.4f}\n"
        )
        print(_fit_table(rows, names))
        if label == "eval" and not args.eval_split:
            break
    return 0


# --- priors and registry -----------------------------------------------------------------------


def cmd_priors(args: argparse.Namespace) -> int:
    items = _items(args, args.run, args.split)
    if any(item.raw is None for item in items):
        raise SystemExit(f"{args.run}: items without raw probabilities (need emulator diagnostics)")
    if args.method == "batch":
        priors = fit_batch_priors(
            (
                (item.prior_key, item.raw.tolist(), item.free)
                for item in items
                if item.raw is not None
            ),
            min_count=args.min_items or DEFAULT_MIN_GROUP,
        )
    else:
        priors = fit_pride_priors(
            (
                (item.prior_key, item.presentations, item.free)
                for item in items
                if full_cycle(item.presentations, item.free)
            ),
            min_count=args.min_items or 1,
        )
    if not priors.priors:
        raise SystemExit(f"{args.run}: no questions to estimate {args.method} priors from")
    priors.save(args.out)
    for key in sorted(priors.priors):
        values = ", ".join(f"{v:.3f}" for v in priors.priors[key])
        print(f"{key}  n={priors.counts[key]}  [{values}]")
    print(f"wrote {args.out}")
    return 0


def cmd_registry(args: argparse.Namespace) -> int:
    manifest = RunManifest.load(args.run)
    system = manifest.system if manifest else None
    if not system or system.get("kind") != "emulator":
        raise SystemExit(f"{args.run}: not an emulator run (manifest system kind)")
    items = _items(args, args.run, args.split)
    if args.priors is None:
        probs, source = {item.item_id: item.probs for item in items}, "recorded probabilities"
    elif any(item.raw is None for item in items):
        raise SystemExit(f"{args.run}: items without raw probabilities (need emulator diagnostics)")
    elif str(args.priors) == "none":
        probs = {item.item_id: item.raw for item in items if item.raw is not None}
        source = "raw probabilities (no debiaser)"
    else:
        priors = LabelPriors.load(args.priors)
        probs = apply_label_priors(items, priors)
        source = f"raw probabilities / {priors.method} priors {args.priors}"
    registry = fit_registry(
        items,
        probs,
        backend=system["backend"],
        model=system["backend_model"],
        template_id=system["renderer"]["template_id"],
        name=args.calibrator,
        min_group=args.min_group,
    )
    registry.save(args.out)
    print(f"wrote {args.out} ({len(items)} items, {args.calibrator} per signature, on {source})")
    return 0


# --- main --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--benchmarks", type=_benchmarks, default=None, help="comma list or all")
    commands = parser.add_subparsers(dest="command", required=True)

    cross = commands.add_parser("crossfit", help="cross-fitted calibration study")
    cross.add_argument("runs", nargs="+", type=_run_arg, metavar="NAME=RUN_DIR")
    cross.add_argument("--split", choices=SPLITS, default="holdout")
    cross.add_argument("--out", type=Path, required=True)
    cross.add_argument("--reference", default="jev", help="system the others are paired with")
    cross.add_argument("--debias", type=_debias, default=[], help="none,batch,pride (emulator)")
    cross.add_argument("--calibrators", type=_list, default=list(DEFAULT_CALIBRATORS))
    cross.add_argument("--primary", default=DEFAULT_PRIMARY)
    cross.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    cross.add_argument("--seed", type=int, default=0)
    cross.add_argument("--min-group", type=int, default=DEFAULT_MIN_GROUP)
    cross.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    cross.set_defaults(func=cmd_crossfit)

    conf = commands.add_parser("confidence", help="fit Jev's confidence function")
    conf.add_argument("run", type=Path)
    conf.add_argument("--fit-split", choices=SPLITS, default="select")
    conf.add_argument("--eval-split", choices=SPLITS, default=None)
    conf.add_argument("--base", choices=sorted(CONFIDENCE_FUNCTIONS), default="mode_distance")
    conf.set_defaults(func=cmd_confidence)

    pri = commands.add_parser("priors", help="label priors for the batch/pride debiasers")
    pri.add_argument("run", type=Path)
    pri.add_argument("--split", choices=SPLITS, default="holdout")
    pri.add_argument("--method", choices=("batch", "pride"), required=True)
    pri.add_argument(
        "--min-items",
        type=int,
        default=None,
        help=f"questions a prior key needs (default: batch {DEFAULT_MIN_GROUP}, pride 1)",
    )
    pri.add_argument("--out", type=Path, required=True)
    pri.set_defaults(func=cmd_priors)

    reg = commands.add_parser("registry", help="calibrator registry fitted on a whole run")
    reg.add_argument("run", type=Path)
    reg.add_argument("--split", choices=SPLITS, default="holdout")
    reg.add_argument("--calibrator", default="temperature")
    reg.add_argument("--min-group", type=int, default=DEFAULT_MIN_GROUP)
    reg.add_argument("--out", type=Path, required=True)
    reg.add_argument(
        "--priors",
        type=Path,
        default=None,
        help="the deployed debiaser: 'none' (fit on the strategy's raw probabilities) or a "
        "LabelPriors JSON (raw divided by it); default: the run's recorded probabilities",
    )
    reg.set_defaults(func=cmd_registry)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
