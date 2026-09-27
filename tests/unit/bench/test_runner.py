from __future__ import annotations

import asyncio
import json
import math
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest

from jevemu.bench import BenchConfig, BenchItem, BenchmarkSpec
from jevemu.bench.compare import compare, summarize
from jevemu.bench.manifest import RunManifest
from jevemu.bench.runner import load_records, records_path, run_split
from jevemu.errors import JevBudgetExceeded, JevValidationError
from jevemu.eval.bank import IDK_TEXT
from jevemu.eval.item_scores import score_answer
from jevemu.eval.splits import split_items
from jevemu.types import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)

KEYS = ("A", "B", "C", "D")
SOURCE = {"revision": "0" * 40, "source_sha256": "1" * 64}


def _choice_spec(name: str = "toychoice", n: int = 12) -> BenchmarkSpec:
    question = ChoiceQuestion(
        instructions="Pick one.", criteria={**{k: f"option {k}" for k in KEYS}, "E": IDK_TEXT}
    )
    items = [
        BenchItem(
            item_id=f"{name}:{i}",
            state=f"{name} {i}",
            question=question,
            gold=KEYS[i % 4],
            metadata={"question_id": str(i), "stratum": "s", "idk": True, **SOURCE},
        )
        for i in range(n)
    ]
    return BenchmarkSpec(name, "choice", lambda cfg: items, {"test": "test"}, "CC0")


def _noul_spec(name: str = "toynoul", n: int = 20) -> BenchmarkSpec:
    question = NoulQuestion(instructions="Yes?")
    items = [
        BenchItem(
            item_id=f"{name}:{i}",
            state=f"{name} {i}",
            question=question,
            gold=i % 2 == 0,
            metadata={"question_id": str(i), "stratum": "s", **SOURCE},
        )
        for i in range(n)
    ]
    return BenchmarkSpec(name, "noul", lambda cfg: items, {"test": "test"}, "CC0")


def _gold(spec: BenchmarkSpec) -> dict[str, object]:
    return {str(item.state): item.gold for item in spec.loader(BenchConfig())}


Policy = Callable[[str], Answer]


class FakeClient:
    """Answers by ``policy(state)``; can fail chosen states or run out of budget."""

    def __init__(
        self,
        policy: Policy,
        *,
        fail: frozenset[str] = frozenset(),
        budget_calls: int | None = None,
    ) -> None:
        self.policy = policy
        self.fail = fail
        self.budget_calls = budget_calls
        self.calls = 0
        self.answered: Counter[str] = Counter()

    async def system_one(self, request: SystemOneRequest) -> SystemOneResponse:
        state = str(request.state)
        self.calls += 1
        if self.budget_calls is not None and self.calls > self.budget_calls:
            raise JevBudgetExceeded(max_usd=0.01, committed_usd=0.01, estimated_usd=0.001)
        await asyncio.sleep(0)
        if state in self.fail:
            raise JevValidationError(422, {"detail": "rejected"})
        self.answered[state] += 1
        return SystemOneResponse(
            model="fake-1",
            answers={"answer": self.policy(state)},
            usage=Usage(input_tokens=10, output_tokens=0),
        )


def _choice_answer(choice: str) -> ChoiceAnswer:
    probabilities = {k: (0.6 if k == choice else 0.1) for k in (*KEYS, "E")}
    return ChoiceAnswer(choice=choice, probabilities=probabilities, confidence=0.5)


def _always(choice: str) -> Policy:
    return lambda state: _choice_answer(choice)


def _record_ids(path: Path) -> list[str]:
    return [json.loads(line)["item_id"] for line in path.read_text().splitlines()]


async def test_budget_stop_then_resume_answers_every_item_exactly_once(tmp_path: Path) -> None:
    spec = _choice_spec()
    first = FakeClient(_always("A"), budget_calls=5)
    stopped = await run_split(first, "sys", spec, "select", tmp_path, concurrency=3)
    assert stopped.status == "stopped_budget"
    assert (stopped.n_items, stopped.n_ok, stopped.n_errors) == (6, 5, 0)
    assert "JevBudgetExceeded" in (stopped.stop_reason or "")
    manifest = RunManifest.load(tmp_path / "sys")
    assert manifest is not None
    assert manifest.runs["toychoice.select"].status == "stopped_budget"
    assert manifest.runs["toychoice.select"].n_ok == 5

    path = records_path(tmp_path, "sys", "toychoice", "select")
    with path.open("a") as sink:  # a write torn by a crash
        sink.write('{"item_id": "toychoice:')

    second = FakeClient(_always("A"))
    done = await run_split(second, "sys", spec, "select", tmp_path, concurrency=3)
    assert done.status == "complete"
    assert (done.n_ok, done.n_called, done.n_skipped) == (6, 1, 5)
    answered = first.answered + second.answered
    assert sorted(answered.values()) == [1] * 6
    assert _record_ids(path) == [item.item_id for item in split_items(spec, "select")]

    third = FakeClient(_always("A"))
    again = await run_split(third, "sys", spec, "select", tmp_path, concurrency=3)
    assert (again.status, again.n_called, third.calls) == ("complete", 0, 0)
    manifest = RunManifest.load(tmp_path / "sys")
    assert manifest is not None
    assert [i.status for i in manifest.runs["toychoice.select"].invocations] == [
        "stopped_budget",
        "complete",
        "complete",
    ]


async def test_item_errors_are_recorded_and_only_failed_items_are_retried(tmp_path: Path) -> None:
    spec = _choice_spec()
    select = split_items(spec, "select")
    failing = {str(select[0].state), str(select[3].state)}
    first = FakeClient(_always("B"), fail=frozenset(failing))
    result = await run_split(first, "sys", spec, "select", tmp_path, concurrency=4)
    assert (result.status, result.n_ok, result.n_errors) == ("incomplete", 4, 2)
    records = load_records(result.path)
    failed = {item.item_id for item in select if str(item.state) in failing}
    assert {i for i, r in records.items() if not r.ok} == failed
    for item_id in failed:
        error = records[item_id].error
        assert error is not None
        assert (error.type, error.status) == ("JevValidationError", 422)

    second = FakeClient(_always("B"))
    retried = await run_split(second, "sys", spec, "select", tmp_path, concurrency=4)
    assert (retried.status, retried.n_called) == ("complete", 2)
    assert set(second.answered) == failing
    assert all(r.ok for r in load_records(retried.path).values())
    assert _record_ids(retried.path) == [item.item_id for item in select]


async def test_resuming_with_another_system_is_refused(tmp_path: Path) -> None:
    class OtherClient(FakeClient):
        pass

    spec = _choice_spec()
    await run_split(FakeClient(_always("A")), "sys", spec, "select", tmp_path, concurrency=2)
    with pytest.raises(ValueError, match="different system"):
        await run_split(OtherClient(_always("A")), "sys", spec, "select", tmp_path, concurrency=2)


def test_choice_scores_renormalize_and_count_idk_as_wrong() -> None:
    rounded = ChoiceAnswer(choice="A", probabilities={"A": 0.51, "B": 0.5}, confidence=0.0)
    score = score_answer(rounded, "B")
    assert not score.correct
    assert score.p_gold == pytest.approx(0.5 / 1.01)
    assert score.nll == pytest.approx(-math.log(0.5 / 1.01))
    assert score.brier == pytest.approx((0.51 / 1.01) ** 2 + (0.5 / 1.01 - 1) ** 2)
    assert score.confidence == pytest.approx(0.51 / 1.01)

    idk = ChoiceAnswer(choice="E", probabilities={"A": 0.3, "B": 0.0, "E": 0.7}, confidence=0.0)
    abstained = score_answer(idk, "B", idk_key="E")
    assert (abstained.correct, abstained.abstained) == (False, True)
    assert abstained.nll == pytest.approx(-math.log(1e-6))
    assert score_answer(idk, "E", idk_key=None).abstained is False


def test_noul_scores_threshold_at_one_half() -> None:
    half = score_answer(NoulAnswer(noul=0.5), True)
    assert (half.correct, half.p_gold, half.confidence) == (True, 0.5, 0.5)
    no = score_answer(NoulAnswer(noul=0.2), False)
    assert no.correct
    assert no.p_gold == pytest.approx(0.8)
    assert no.brier == pytest.approx(2 * 0.2**2)
    assert no.confidence == pytest.approx(0.8)
    assert not score_answer(NoulAnswer(noul=0.2), True).correct


def test_score_scores_use_the_argmax_level_and_the_reported_score() -> None:
    answer = ScoreAnswer(
        score=1.5,
        legend={str(i): None for i in range(4)},
        probabilities={"0": 0.1, "1": 0.4, "2": 0.4, "3": 0.1},
        confidence=0.0,
    )
    tied = score_answer(answer, 2)
    assert (tied.correct, tied.within_one) == (False, True)  # tie goes to the lower level
    assert tied.abs_error == pytest.approx(0.5)
    assert tied.nll == pytest.approx(-math.log(0.4))
    assert tied.brier == pytest.approx(0.01 + 0.16 + 0.36 + 0.01)
    assert score_answer(answer, 1).correct
    assert score_answer(answer, 3).within_one is False


async def test_macro_average_weighs_benchmarks_equally_and_compare_pairs_shared_items(
    tmp_path: Path,
) -> None:
    choice, noul = _choice_spec(n=40), _noul_spec(n=8)
    gold = {**_gold(choice), **_gold(noul)}

    def oracle(state: str) -> Answer:
        g = gold[state]
        if isinstance(g, bool):
            return NoulAnswer(noul=0.9 if g else 0.1)
        return _choice_answer(str(g))

    def contrarian(state: str) -> Answer:
        g = gold[state]
        if isinstance(g, bool):
            return NoulAnswer(noul=0.1 if g else 0.9)
        return _choice_answer("E")

    for system_id, policy in (("good", oracle), ("bad", contrarian)):
        for spec in (choice, noul):
            client = FakeClient(policy)
            await run_split(client, system_id, spec, "select", tmp_path, concurrency=4)
    # "bad" misses one noul item, which pairing must drop.
    bad_noul = records_path(tmp_path, "bad", "toynoul", "select")
    lines = bad_noul.read_text().splitlines()
    bad_noul.write_text("\n".join(lines[1:]) + "\n")

    good = summarize(tmp_path / "good", n_resamples=200)
    assert good.macro_accuracy.estimate == pytest.approx(1.0)
    bad = summarize(tmp_path / "bad", n_resamples=200)
    assert bad.benchmarks["toychoice"].scores.idk_rate == pytest.approx(1.0)
    assert bad.benchmarks["toynoul"].scores.n == 3

    # Accuracy of "good" on toychoice only is 1; mixed with a benchmark where it is 0 the
    # macro average must be 0.5 whatever the item counts.
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    for source, name in (("good", "toychoice"), ("bad", "toynoul")):
        records = records_path(tmp_path, source, name, "select")
        (mixed / records.name).write_text(records.read_text())
    macro = summarize(mixed, n_resamples=500).macro_accuracy
    assert macro.estimate == pytest.approx(0.5)
    assert macro.low <= 0.5 <= macro.high

    comparison = compare(tmp_path / "good", tmp_path / "bad", n_resamples=200)
    assert comparison.benchmarks["toynoul"].n == 3
    assert comparison.benchmarks["toychoice"].n == 20
    assert comparison.benchmarks["toynoul"].mcnemar.only_a == 3
    assert comparison.macro_delta_accuracy.estimate == pytest.approx(1.0)
    assert comparison.benchmarks["toychoice"].delta_nll.estimate < 0

    # A benchmark filter leaves the others out of the macro as well as the table.
    both = compare(mixed, tmp_path / "bad", n_resamples=200)
    assert both.macro_delta_accuracy.estimate == pytest.approx(0.5)
    only = compare(mixed, tmp_path / "bad", n_resamples=200, benchmarks=["toynoul"])
    assert list(only.benchmarks) == ["toynoul"]
    assert only.macro_delta_accuracy.estimate == pytest.approx(0.0)


async def test_a_dropped_benchmark_leaves_the_macro_unless_named(tmp_path: Path) -> None:
    kept, dropped = _choice_spec("toychoice", n=20), _choice_spec("yelp_stars", n=20)
    for spec, answer in ((kept, "A"), (dropped, "E")):  # sometimes right, never right
        await run_split(FakeClient(_always(answer)), "sys", spec, "select", tmp_path, concurrency=2)

    summary = summarize(tmp_path / "sys", n_resamples=100)
    assert list(summary.benchmarks) == ["toychoice"]
    kept_accuracy = summary.benchmarks["toychoice"].scores.accuracy.estimate
    assert kept_accuracy > 0
    assert summary.macro_accuracy.estimate == pytest.approx(kept_accuracy)
    named = compare(tmp_path / "sys", tmp_path / "sys", n_resamples=100, benchmarks=["yelp_stars"])
    assert list(named.benchmarks) == ["yelp_stars"]
