"""Run manifests: what produced a run directory and how far each split got.

A run directory is ``<out>/<system_id>/``. Its ``manifest.json`` names the system (for Jev the
pinned model version; for the emulator the backend model and revision, engine version, server
flags including the image, the ``JEVEMU_VLLM_*`` launch settings, the renderer's
``template_id``, the scoring strategy and the calibrator) and, per ``<benchmark>.<split>``
records file, the frozen split's metadata (``items_sha256``), status, counts, spend and every
invocation (start/end, jevemu git commit, items called, spend) that wrote to it. ``totals`` sums
the splits. The manifest is rewritten atomically after every change.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jevemu.backends.base import PRICE_ID_FLAG
from jevemu.calibrate import CalibratorRegistry
from jevemu.emulator import Emulator
from jevemu.eval.splits import SplitMetadata, SplitName
from jevemu.jev_client import JevClient
from jevemu.types import SystemOneClient

__all__ = [
    "MANIFEST_NAME",
    "MANIFEST_VERSION",
    "Invocation",
    "RunManifest",
    "RunStatus",
    "SplitRun",
    "Totals",
    "describe_system",
    "git_state",
    "jevemu_version",
    "run_key",
    "system_differences",
]

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = "1"

RunStatus = Literal["running", "complete", "incomplete", "stopped_budget", "failed"]
"""``incomplete``: interrupted or finished with per-item errors left to retry;
``stopped_budget``: the Jev budget guard refused a request; ``failed``: a fatal error (bad API
key, version drift) stopped the run."""

_ROOT = Path(__file__).resolve().parents[3]
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def run_key(benchmark: str, split: SplitName) -> str:
    """``"<benchmark>.<split>"``: the records file stem and the manifest's ``runs`` key."""
    return f"{benchmark}.{split}"


class Invocation(BaseModel):
    """One :func:`~jevemu.bench.runner.run_split` call on a split."""

    model_config = ConfigDict(extra="forbid")

    started_at: datetime
    finished_at: datetime | None = None
    status: RunStatus = "running"
    git_commit: str | None
    git_dirty: bool | None
    concurrency: int = Field(ge=1)
    limit: int | None
    n_called: int = Field(default=0, ge=0)
    """Items sent to the system (successes and errors)."""
    n_errors: int = Field(default=0, ge=0)
    n_cache_hits: int = Field(default=0, ge=0)
    """Jev and paid-API emulators: answers served from the response cache (cost 0)."""
    spent_usd: float = Field(default=0.0, ge=0.0)
    stop_reason: str | None = None


class SplitRun(BaseModel):
    """One ``<benchmark>.<split>.jsonl`` records file."""

    model_config = ConfigDict(extra="forbid")

    benchmark: str
    split: SplitName
    split_metadata: SplitMetadata
    status: RunStatus
    n_items: int = Field(ge=0)
    """Items requested by the latest invocation (the split, or its first ``limit`` items)."""
    n_ok: int = Field(ge=0)
    """Items with an answer (any invocation)."""
    n_errors: int = Field(ge=0)
    """Items whose latest attempt failed."""
    spent_usd: float = Field(ge=0.0)
    """Sum of the recorded items' cost."""
    invocations: list[Invocation] = Field(default_factory=list)


class Totals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_items: int = 0
    n_ok: int = 0
    n_errors: int = 0
    spent_usd: float = 0.0


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_version: str = MANIFEST_VERSION
    system_id: str
    system: dict[str, Any]
    jevemu_version: str
    runs: dict[str, SplitRun] = Field(default_factory=dict)
    totals: Totals = Field(default_factory=Totals)

    @classmethod
    def load(cls, run_dir: Path) -> RunManifest | None:
        path = run_dir / MANIFEST_NAME
        if not path.exists():
            return None
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, run_dir: Path) -> None:
        """Recompute :attr:`totals` and write ``manifest.json`` atomically."""
        self.totals = Totals(
            n_items=sum(r.n_items for r in self.runs.values()),
            n_ok=sum(r.n_ok for r in self.runs.values()),
            n_errors=sum(r.n_errors for r in self.runs.values()),
            spent_usd=sum(r.spent_usd for r in self.runs.values()),
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / MANIFEST_NAME
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)


def jevemu_version() -> str:
    try:
        return metadata.version("jevemu")
    except metadata.PackageNotFoundError:
        return "unknown"


def git_state(root: Path = _ROOT) -> tuple[str | None, bool | None]:
    """(HEAD commit, working tree has changes) of the jevemu checkout; ``None`` outside git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit or None, bool(status.strip())


async def describe_system(system: SystemOneClient) -> dict[str, Any]:
    """The identity of ``system`` recorded in the manifest (resuming requires it unchanged).

    The emulator is warmed up first so its backend identity is known. An emulator over a paid
    API also records the backend's ``price_id`` flag (:data:`~jevemu.backends.base.PRICE_ID_FLAG`),
    which makes :class:`~jevemu.eval.costs.PriceBook` token-price it instead of by GPU time, and
    a price not built into ``TOKEN_PRICES`` as ``token_price`` (``BackendInfo.price``: OpenRouter's
    snapshot). ``token_price`` is not identity: a resumed run keeps the first invocation's.
    """
    if isinstance(system, JevClient):
        return {"kind": "jev", "model": system.model, "base_url": system.base_url}
    if isinstance(system, Emulator):
        info = await system.warm_up()
        description: dict[str, Any] = {
            "kind": "emulator",
            "backend": info.backend,
            "backend_model": info.model,
            "model_revision": info.model_revision,
            "engine_version": info.engine_version,
            "server_flags": dict(sorted(info.flags.items())),
            "launch_env": {
                name: value
                for name, value in sorted(os.environ.items())
                if name.startswith("JEVEMU_VLLM_")
                and not any(marker in name for marker in _SECRET_MARKERS)
                and name != "JEVEMU_VLLM_URL"
            },
            "renderer": {
                "layout": system.renderer.layout,
                "template_id": system.renderer.template_id,
            },
            "strategy": system.strategy.name,
            "debiaser": None if system.debiaser is None else system.debiaser.id,
            "calibrator": _calibrator_id(system.calibrators),
            "confidence_fn": getattr(system.confidence_fn, "__name__", repr(system.confidence_fn)),
            "round_to": system.round_to,
        }
        price_id = info.flags.get(PRICE_ID_FLAG)
        if price_id is not None:
            description["price_id"] = price_id
        if info.price is not None:
            description["token_price"] = dict(info.price)
        return description
    return {"kind": type(system).__name__}


def _calibrator_id(registry: CalibratorRegistry | None) -> str | None:
    """``"sha256:<hex>"`` of the registry's canonical JSON, ``None`` without calibration."""
    if registry is None:
        return None
    text = json.dumps(registry.to_json(), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


_NOT_IDENTITY = frozenset({"base_url", "launch_env", "token_price"})
"""Description fields that may change without changing the system's answers."""


def system_differences(recorded: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Identity fields of two :func:`describe_system` results that differ (sorted)."""
    keys = (set(recorded) | set(current)) - _NOT_IDENTITY
    return sorted(key for key in keys if recorded.get(key) != current.get(key))
