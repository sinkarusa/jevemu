"""Cross-fitting: fold assignment, out-of-fold calibration and out-of-fold debiasing priors."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import pytest

from jevemu.calibrate import FloatArray, IdentityCalibrator, IntArray
from jevemu.debias import Presentation, cyclic_orders
from jevemu.eval.crossfit import (
    CalibratorSpec,
    CalItem,
    assign_folds,
    crossfit_predict,
    debias_offline,
)


def item(
    item_id: str,
    probs: FloatArray,
    gold: int,
    *,
    benchmark: str = "bench",
    question_id: str | None = None,
    stratum: str = "",
    signature: str = "choice:4:x",
    raw: FloatArray | None = None,
    presentations: tuple[Presentation, ...] = (),
) -> CalItem:
    k = probs.shape[0]
    return CalItem(
        item_id=item_id,
        benchmark=benchmark,
        question_id=question_id or item_id,
        stratum=stratum,
        signature=signature,
        prior_key=signature,
        free=(True,) * k,
        keys=tuple("ABCDEFGHIJ"[:k]),
        gold=gold,
        probs=probs,
        returned=probs,
        raw=raw,
        presentations=presentations,
    )


def softmax(z: FloatArray) -> FloatArray:
    e = np.exp(z - z.max())
    out: FloatArray = e / e.sum()
    return out


# --- folds -------------------------------------------------------------------------------------


def test_folds_keep_a_question_together_and_spread_each_stratum_evenly() -> None:
    items = [
        item(
            f"{stratum}{q}:{perm}",
            np.full(4, 0.25),
            0,
            question_id=f"{stratum}{q}",
            stratum=stratum,
        )
        for stratum, size in (("a", 7), ("b", 12), ("c", 3))
        for q in range(size)
        for perm in range(2)  # two permutations of every question
    ]
    folds = assign_folds(items, n_folds=5, seed=0)
    by_question: dict[str, set[int]] = {}
    for it in items:
        by_question.setdefault(it.question_id, set()).add(folds[it.item_id])
    assert all(len(f) == 1 for f in by_question.values())
    for stratum in "abc":
        counts = np.bincount(
            [folds[f"{q}:0"] for q in by_question if q.startswith(stratum)], minlength=5
        )
        assert counts.max() - counts.min() <= 1
    overall = np.bincount([next(iter(f)) for f in by_question.values()], minlength=5)
    assert overall.max() - overall.min() <= 1
    shuffled = items[:]
    random.Random(1).shuffle(shuffled)
    assert assign_folds(shuffled, n_folds=5, seed=0) == folds
    assert assign_folds(items, n_folds=5, seed=1) != folds


# --- calibration -------------------------------------------------------------------------------


class RecordingIdentity(IdentityCalibrator):
    """Identity that remembers its training rows and refuses to transform one of them."""

    def __init__(self) -> None:
        self.seen: set[bytes] = set()

    def fit(self, logp: FloatArray, y: IntArray) -> RecordingIdentity:
        self.seen = {row.tobytes() for row in np.asarray(logp)}
        return self

    def transform(self, logp: FloatArray) -> FloatArray:
        leaked = [row for row in np.asarray(logp) if row.tobytes() in self.seen]
        assert not leaked, "a calibrator transformed an item it was fitted on"
        return super().transform(logp)


@pytest.mark.parametrize("scope", ["global", "benchmark", "signature"])
def test_crossfit_never_fits_on_the_evaluated_fold(scope: Any) -> None:
    rng = np.random.default_rng(0)
    items = [
        item(f"{b}:{i}", rng.dirichlet(np.ones(4)), int(rng.integers(4)), benchmark=b)
        for b in ("one", "two")
        for i in range(60)
    ]
    folds = assign_folds(items)
    out, fits = crossfit_predict(
        items,
        {it.item_id: it.probs for it in items},
        folds,
        CalibratorSpec("identity", scope),
        factory=lambda name: RecordingIdentity(),
        min_group=1,
    )
    assert set(out) == {it.item_id for it in items}
    for it in items:  # identity on floored logits: the input, up to the 1e-4 floor
        assert out[it.item_id] == pytest.approx(it.probs, abs=1e-3)
    assert {fit.fold for fit in fits} == set(range(5))
    assert all(fit.n_train < len(items) for fit in fits)


def test_temperature_crossfit_recovers_a_known_temperature() -> None:
    rng = np.random.default_rng(7)
    true_t = 2.0
    items = []
    for i in range(3000):
        z = rng.normal(0.0, 1.5, size=4)
        gold = int(rng.choice(4, p=softmax(z / true_t)))
        items.append(item(str(i), softmax(z), gold))  # reported probs: overconfident, T = 1
    folds = assign_folds(items)
    probs = {it.item_id: it.probs for it in items}
    out, fits = crossfit_predict(items, probs, folds, CalibratorSpec.parse("temperature@global"))
    temps = [fit.params["temperature"] for fit in fits]
    assert len(temps) == 5
    assert all(abs(t - true_t) / true_t < 0.05 for t in temps)
    raw_nll = np.mean([-math.log(it.probs[it.gold]) for it in items])
    cal_nll = np.mean([-math.log(out[it.item_id][it.gold]) for it in items])
    assert cal_nll < raw_nll


def test_small_groups_fall_back_to_the_folds_global_temperature() -> None:
    rng = np.random.default_rng(1)
    items = [item(f"big{i}", rng.dirichlet(np.ones(4)), 0, signature="big") for i in range(100)]
    items += [item(f"small{i}", rng.dirichlet(np.ones(4)), 1, signature="small") for i in range(5)]
    folds = assign_folds(items)
    _, fits = crossfit_predict(
        items,
        {it.item_id: it.probs for it in items},
        folds,
        CalibratorSpec.parse("vector@signature"),
        min_group=30,
    )
    assert {fit.fallback for fit in fits if fit.group == "small"} == {True}
    assert {fit.fallback for fit in fits if fit.group == "big"} == {False}
    assert all(
        "temperature" in fit.params and "bias" not in fit.params
        for fit in fits
        if fit.group == "small"
    )


# --- offline debiasing -------------------------------------------------------------------------


def biased(weights: FloatArray, bias: FloatArray) -> FloatArray:
    p = weights * bias
    out: FloatArray = p / p.sum()
    return out


def presentations_of(weights: FloatArray, bias: FloatArray) -> tuple[Presentation, ...]:
    out = []
    for order in cyclic_orders((True,) * 4):
        shown = weights[list(order)]
        out.append(Presentation(order, tuple(np.log(biased(shown, bias)).tolist())))
    return tuple(out)


def test_offline_pride_divides_by_a_prior_from_the_other_folds_only() -> None:
    rng = np.random.default_rng(3)
    bias = np.array([6.0, 1.0, 1.0, 1.0])
    other_bias = np.array([1.0, 1.0, 1.0, 9.0])  # what an in-fold estimate would divide out
    items = [item(str(i), np.full(4, 0.25), 0) for i in range(100)]
    folds = assign_folds(items)
    built = []
    truths = {}
    for i, it in enumerate(items):
        weights = rng.dirichlet(np.ones(4))
        truths[it.item_id] = weights
        estimating = i % 2 == 0
        own = other_bias if folds[it.item_id] == 0 else bias
        built.append(
            item(
                it.item_id,
                biased(weights, bias),
                0,
                raw=biased(weights, bias),
                presentations=presentations_of(weights, own) if estimating else (),
            )
        )
    out = debias_offline(built, "pride", folds)
    checked = 0
    for it in built:
        if folds[it.item_id] == 0 and not it.presentations:
            assert out[it.item_id] == pytest.approx(truths[it.item_id], abs=1e-9)
            checked += 1
    assert checked > 0
    none = debias_offline(built, "none", folds)
    assert all(np.array_equal(none[it.item_id], it.raw) for it in built)
