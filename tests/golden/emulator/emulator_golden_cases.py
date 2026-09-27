"""Requests and emulator settings of the emulator golden set (shared by recorder and replay).

Not a test module. The requests are Jev doc examples (``tests/golden/fixtures/jev_docs``):
bare and described Noul, Choice, Score, a mixed request, and structured instructions with
null option descriptions and five-level Scores. Re-record after changing this list, the
emulator, the strategies or the prompt templates (``record_emulator_fixtures.py``).
"""

from __future__ import annotations

from pathlib import Path

from jevemu.backends.base import Backend
from jevemu.emulator import Emulator
from jevemu.types import SystemOneRequest

HERE = Path(__file__).parent
FIXTURE = HERE / "emulator_vllm.jsonl"
EXPECTED = HERE / "expected.json"
DOC_FIXTURES = HERE.parent / "fixtures" / "jev_docs"

REQUESTS = (
    "api__01",
    "api__02",
    "api__03",
    "api__04",
    "introduction_quickstart__02",
    "primitives_advanced__01",
)


def load_request(name: str) -> SystemOneRequest:
    path = DOC_FIXTURES / f"{name}__request.json"
    return SystemOneRequest.model_validate_json(path.read_text(encoding="utf-8"))


def make_emulator(backend: Backend) -> Emulator:
    return Emulator(backend, include_diagnostics=True)
