"""``auto`` with echo switched off (the selection runs' policy) on an echo-capable backend."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
from scoring_helpers import RENDERER, STATE, VLLM_LIKE, choice, lp

from jevemu.backends.base import RenderedPrompt
from jevemu.backends.fake import FakeBackend
from jevemu.scoring import NoEchoAutoStrategy

KEYS = [f"opt-{i}" for i in range(40)]
VOCAB = [f" opt-{i}" for i in range(40)] + ["\n"]
"""Every option key is one token after the space, so the trie has one decision point."""


def _root(prompt: RenderedPrompt, allowed: Sequence[str] | None) -> dict[str, float]:
    # Only three of the 40 keys come back, as from a top-k that cannot hold them all.
    return {" opt-7": lp(0.6), " opt-3": lp(0.3), " opt-12": lp(0.1)}


async def test_overflowing_choice_uses_the_trie_and_never_echoes() -> None:
    # The backend can echo, so plain ``auto`` would echo-score these 40 options.
    backend = FakeBackend(capabilities=VLLM_LIKE, next_token=_root, vocab=VOCAB)
    result = await NoEchoAutoStrategy().score(backend, RENDERER, STATE, choice(40))
    assert result.strategy == "trie"
    assert result.keys == tuple(KEYS)
    assert not [name for name, _ in backend.calls if name == "sequence_logprobs"]
    # The 37 unseen keys each get the upper bound (smallest returned logprob), not a zero.
    probs = [math.exp(v) for v in result.logprobs]
    assert max(result.probabilities, key=result.probabilities.__getitem__) == "opt-7"
    seen = ("opt-7", "opt-3", "opt-12")
    unseen = [p for key, p in zip(KEYS, probs, strict=True) if key not in seen]
    assert unseen == pytest.approx([probs[KEYS.index("opt-12")]] * 37)
    assert result.truncated
    assert "auto: echo off: using S3 trie" in result.warnings
    assert not any("echo unavailable" in w for w in result.warnings)
