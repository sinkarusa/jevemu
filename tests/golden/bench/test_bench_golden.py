"""Built-in benchmark goldens: five items per dataset adapter, byte-stable.

``fixtures/<benchmark>.json`` holds the first five rows of each pinned file (written by
``record_bench_fixtures.py``; Yelp's are synthetic because its terms forbid redistribution, and
GPQA is never sampled), and ``snapshots/<benchmark>.jsonl`` the items the adapter makes from
them. After an intended item change, regenerate the snapshots with
``JEVEMU_UPDATE_SNAPSHOTS=1 uv run pytest tests/golden/bench``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub import try_to_load_from_cache

from jevemu.bench import BenchConfig, BenchItem, load_benchmark
from jevemu.bench.datasets import (
    ag_news,
    arc,
    banking77,
    boolq,
    clinc150,
    hub,
    mmlu_pro,
    sst5,
    yelp,
)
from jevemu.eval.bank import PinnedFile

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures"
SNAPSHOTS = HERE / "snapshots"
UPDATE = os.environ.get("JEVEMU_UPDATE_SNAPSHOTS") == "1"

PINS: dict[str, PinnedFile] = {
    module.NAME: module.PIN
    for module in (mmlu_pro, arc, ag_news, banking77, clinc150, boolq, sst5, yelp)
}
SYNTHETIC = {yelp.NAME}


def _fixture(name: str) -> dict[str, Any]:
    fixture: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return fixture


def _write_dataset_file(fixture: dict[str, Any], directory: Path) -> Path:
    rows = fixture["rows"]
    if fixture["format"] == "jsonl":
        path = directory / "rows.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path
    path = directory / "rows.parquet"
    table = pa.Table.from_pylist(rows).replace_schema_metadata(fixture["schema_metadata"] or None)
    pq.write_table(table, path)
    return path


def _check_snapshot(items: list[BenchItem], name: str) -> None:
    path = SNAPSHOTS / f"{name}.jsonl"
    text = "".join(item.model_dump_json() + "\n" for item in items)
    if UPDATE:
        SNAPSHOTS.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
    assert text == path.read_text(encoding="utf-8"), f"{name} items changed"


@pytest.mark.parametrize("name", sorted(PINS))
def test_adapter_turns_the_fixture_rows_into_the_snapshot_items(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(name)
    source = fixture["_source"]
    path = _write_dataset_file(fixture, tmp_path)

    def fetch(pin: PinnedFile) -> tuple[Path, str]:
        # A pin moved without re-recording the fixture fails here.
        assert (pin.repo_id, pin.filename, pin.revision) == (
            source["repo_id"],
            source["filename"],
            source["revision"],
        )
        return path, source["sha256"]

    monkeypatch.setattr(hub, "fetch_pinned", fetch)
    items = load_benchmark(name, BenchConfig())
    assert len(items) == 5
    _check_snapshot(items, name)


def _cached(pin: PinnedFile) -> bool:
    found = try_to_load_from_cache(
        pin.repo_id, pin.filename, revision=pin.revision, repo_type="dataset"
    )
    return isinstance(found, str)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=pytest.mark.skipif(
                not _cached(pin), reason="needs the pinned file in the HF cache"
            ),
        )
        for name, pin in sorted(PINS.items())
        if name not in SYNTHETIC
    ],
)
def test_pinned_file_starts_with_the_fixture_rows(name: str) -> None:
    items = load_benchmark(name, BenchConfig(limit=5))
    text = "".join(item.model_dump_json() + "\n" for item in items)
    assert text == (SNAPSHOTS / f"{name}.jsonl").read_text(encoding="utf-8")
