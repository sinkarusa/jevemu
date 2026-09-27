"""Record the built-in benchmark golden fixtures: the first five rows of each pinned file.

    uv run python tests/golden/bench/record_bench_fixtures.py
    JEVEMU_UPDATE_SNAPSHOTS=1 uv run pytest tests/golden/bench

Writes ``fixtures/<benchmark>.json`` with the rows, the file's parquet schema metadata (it
holds the class names) and its SHA-256. Yelp's terms forbid redistribution, so its fixture is
synthetic, in the pinned file's schema. GPQA is never sampled here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from jevemu.bench.datasets import ag_news, arc, banking77, boolq, clinc150, mmlu_pro, sst5, yelp
from jevemu.eval.bank import PinnedFile, fetch_pinned

FIXTURES = Path(__file__).parent / "fixtures"
N_ROWS = 5

REAL = {
    module.NAME: module.PIN for module in (mmlu_pro, arc, ag_news, banking77, clinc150, boolq, sst5)
}

YELP_SYNTHETIC = [
    {"label": 0, "text": "Waited forty minutes for cold soup. The manager shrugged. Never again."},
    {"label": 3, "text": "Good tacos and friendly staff. Parking is a pain, but worth the trip."},
    {"label": 1, "text": "The room was clean enough, but the AC rattled all night.\\nMeh."},
    {"label": 4, "text": "Best haircut I've had in years! Booked my next one before I left."},
    {"label": 2, "text": "Decent coffee, average pastries. Nothing wrong, nothing special."},
]


def _source(pin: PinnedFile, sha256: str, rows: str) -> dict[str, Any]:
    return {
        "repo_id": pin.repo_id,
        "filename": pin.filename,
        "revision": pin.revision,
        "sha256": sha256,
        "license": pin.license,
        "rows": rows,
    }


def record(name: str, pin: PinnedFile) -> dict[str, Any]:
    path, sha256 = fetch_pinned(pin)
    source = _source(pin, sha256, f"first {N_ROWS} rows in file order")
    if path.suffix == ".jsonl":
        lines = path.read_text(encoding="utf-8").splitlines()[:N_ROWS]
        return {"_source": source, "format": "jsonl", "rows": [json.loads(line) for line in lines]}
    parquet = pq.ParquetFile(path)
    metadata = parquet.schema_arrow.metadata or {}
    rows = parquet.read_row_group(0).slice(0, N_ROWS).to_pylist()
    return {
        "_source": source,
        "format": "parquet",
        "schema_metadata": {key.decode(): value.decode() for key, value in metadata.items()},
        "rows": rows,
    }


def main() -> None:
    FIXTURES.mkdir(exist_ok=True)
    fixtures = {name: record(name, pin) for name, pin in REAL.items()}
    assert yelp.PIN.sha256 is not None
    fixtures[yelp.NAME] = {
        "_source": _source(yelp.PIN, yelp.PIN.sha256, "synthetic rows in the pinned schema"),
        "format": "parquet",
        "schema_metadata": {},
        "rows": YELP_SYNTHETIC,
    }
    for name, fixture in fixtures.items():
        text = json.dumps(fixture, indent=1, ensure_ascii=False) + "\n"
        (FIXTURES / f"{name}.json").write_text(text, encoding="utf-8")
        digest = hashlib.sha256(text.encode()).hexdigest()[:12]
        print(f"{name}: {len(fixture['rows'])} rows ({digest})")


if __name__ == "__main__":
    main()
