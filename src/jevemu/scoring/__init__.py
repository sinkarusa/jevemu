"""Scoring strategies: turn (state, question) into a distribution over the answer keys."""

from __future__ import annotations

from jevemu.scoring.auto import AutoStrategy, Selection, select, select_strategy
from jevemu.scoring.base import ScoreResult, ScoringStrategy
from jevemu.scoring.constrained import ConstrainedStrategy
from jevemu.scoring.echo import EchoMode, EchoStrategy
from jevemu.scoring.first_token import FirstTokenStrategy
from jevemu.scoring.missing_mass import MissingFill, fill_missing
from jevemu.scoring.no_echo import NO_ECHO_TAU, NoEchoAutoStrategy
from jevemu.scoring.single_call import SingleCallStrategy
from jevemu.scoring.trie import TrieStrategy

__all__ = [
    "NO_ECHO_TAU",
    "AutoStrategy",
    "ConstrainedStrategy",
    "EchoMode",
    "EchoStrategy",
    "FirstTokenStrategy",
    "MissingFill",
    "NoEchoAutoStrategy",
    "ScoreResult",
    "ScoringStrategy",
    "Selection",
    "SingleCallStrategy",
    "TrieStrategy",
    "fill_missing",
    "select",
    "select_strategy",
]
