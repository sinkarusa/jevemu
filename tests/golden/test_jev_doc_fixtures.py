"""Every request/response example captured from the Jev docs round-trips through our types."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from jevemu.types import SystemOneRequest, SystemOneResponse

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "jev_docs"
FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))


def _model_for(path: Path) -> type[BaseModel]:
    if path.name.endswith("__request.json"):
        return SystemOneRequest
    if path.name.endswith("__response.json"):
        return SystemOneResponse
    raise AssertionError(f"fixture {path.name} is neither a request nor a response")


def _dump(model: BaseModel) -> Any:
    return model.model_dump(mode="json", exclude_unset=True)


def test_fixture_directory_is_populated() -> None:
    assert FIXTURES, f"no JSON fixtures found in {FIXTURE_DIR}"


@pytest.mark.parametrize("path", FIXTURES, ids=[p.name for p in FIXTURES])
def test_doc_example_round_trips(path: Path) -> None:
    original = json.loads(path.read_text())
    model_cls = _model_for(path)

    dumped = _dump(model_cls.model_validate(original))
    assert dumped == original

    revalidated = model_cls.model_validate_json(json.dumps(dumped))
    assert _dump(revalidated) == dumped
