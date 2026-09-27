from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest

from jevemu.bench import BenchItem, BenchmarkSpec
from jevemu.eval.splits import FrozenSplit, SplitFormatError, freeze_split, split_items, split_of
from jevemu.types import NoulQuestion

QUESTION = NoulQuestion(instructions="Is it spam?")
STRATUM_SIZES = {"a": 7, "b": 4, "c": 1, "d": 10, "e": 3, "f": 250}


def _items(sizes: dict[str, int] = STRATUM_SIZES) -> list[BenchItem]:
    return [
        BenchItem(
            item_id=f"toy:{stratum}{i}",
            state=f"text {stratum}{i}",
            question=QUESTION,
            gold=i % 2 == 0,
            metadata={
                "question_id": f"{stratum}{i}",
                "stratum": stratum,
                "revision": "0" * 40,
                "source_sha256": "1" * 64,
            },
        )
        for stratum, n in sizes.items()
        for i in range(n)
    ]


def _ids(items: list[BenchItem]) -> set[str]:
    return {item.item_id for item in items}


def _strata(items: list[BenchItem]) -> Counter[str]:
    return Counter(str(item.metadata["stratum"]) for item in items)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_select_and_holdout_halve_every_stratum(seed: int) -> None:
    items = _items()
    select = split_of("toy", items, "select", seed=seed)
    holdout = split_of("toy", items, "holdout", seed=seed)
    assert _ids(select).isdisjoint(_ids(holdout))
    assert _ids(select) | _ids(holdout) == _ids(items)
    assert abs(len(select) - len(holdout)) <= 1
    for stratum in STRATUM_SIZES:
        assert abs(_strata(select)[stratum] - _strata(holdout)[stratum]) <= 1


def test_split_depends_on_the_seed_but_not_on_the_load_order() -> None:
    items = _items()
    select = split_of("toy", items, "select", seed=0)
    shuffled = items[:]
    random.Random(7).shuffle(shuffled)
    assert _ids(split_of("toy", shuffled, "select", seed=0)) == _ids(select)
    assert [item.item_id for item in select] == [
        i.item_id for i in items if i.item_id in _ids(select)
    ]
    assert _ids(split_of("toy", items, "select", seed=1)) != _ids(select)


def test_screen_is_a_proportional_stratified_subset_of_select() -> None:
    items = _items()
    select = split_of("toy", items, "select")
    screen = split_of("toy", items, "screen", screen_size=40)
    assert _ids(screen) <= _ids(select)
    assert len(screen) == 40
    for stratum, n in _strata(select).items():
        share = 40 * n / len(select)
        assert int(share) <= _strata(screen)[stratum] <= int(share) + 1
    assert split_of("toy", items, "screen", screen_size=len(select)) == select


def _toy_spec() -> BenchmarkSpec:
    items = _items({"x": 5, "y": 6})
    return BenchmarkSpec(
        name="toy",
        question_type="noul",
        loader=lambda cfg: items,
        splits={"test": "test"},
        license="CC0",
    )


def test_frozen_split_round_trips_and_rejects_edited_items(tmp_path: Path) -> None:
    spec = _toy_spec()
    frozen = freeze_split(spec, "holdout", seed=3)
    assert list(frozen.items) == split_items(spec, "holdout", seed=3)
    assert (frozen.metadata.n_benchmark, frozen.metadata.n_items) == (11, 5)
    assert frozen.metadata.revision == "0" * 40
    path = tmp_path / "split.jsonl"
    frozen.to_jsonl(path)
    assert FrozenSplit.from_jsonl(path) == frozen

    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace('"state":"text', '"state":"TEXT')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(SplitFormatError, match="items_sha256"):
        FrozenSplit.from_jsonl(path)


def test_frozen_split_round_trips_text_with_unicode_line_breaks(tmp_path: Path) -> None:
    items = [
        item.model_copy(update={"state": f"line\x85break\u2028{item.state}"})
        for item in _items({"x": 5, "y": 6})
    ]
    spec = BenchmarkSpec(
        name="toy",
        question_type="noul",
        loader=lambda cfg: items,
        splits={"test": "test"},
        license="0",
    )
    frozen = freeze_split(spec, "select")
    path = tmp_path / "split.jsonl"
    frozen.to_jsonl(path)
    assert FrozenSplit.from_jsonl(path) == frozen
