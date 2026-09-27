"""S3 token trie: chain-rule product along each label's token path.

Labels are scored as ``" " + label``, like echo: Choice by option key (any K up to 255), Score
by digit, Noul by ``Yes``/``No``. The label texts are tokenized and their token paths form a
trie. Only decision points are queried: nodes with more than one outcome, where an outcome is
a child token or, when a label ends at a node that longer labels continue through, the end of
that label. A query extends the assistant prefill with the node's text (the label text so far)
and reads the children's first-token logprobs; it is constrained to the children when the
backend supports a choice constraint and the node has no end outcome. Nodes of one depth are
queried concurrently. Digits come out as one query after ``Answer: `` (the shared space is not
a decision point).

**Maths.** Children logprobs are renormalized over the children at each node, so a label's
probability is the product of its path's conditionals (the constrained-decoding distribution
over the label set). A node where a label ends while longer labels continue is read
unconstrained: the end outcome gets ``1 - sum(p(children))``, i.e. the model writes the shorter
label and then anything but a longer label's next token.

**Pruning.** A decision point whose path probability is below ``tau`` (default 1e-4) is not
queried. Every label under it is assigned the node's path probability, an upper bound; the
result is ``truncated`` and the pruned mass is reported in ``warnings``. Children missing from
a top-k go through the missing-label policy (:mod:`jevemu.scoring.missing_mass`).

**Addressing tokens by text.** The prefill is text, so each token's text (decoded alone) must
concatenate to the label text; that holds for byte-level BPE tokenizers such as Qwen3's on
whole characters. A label that violates it, or two sibling tokens with the same text, raise
``ValueError``. Nodes whose text re-tokenizes differently from the label path are queried
anyway and listed in ``warnings``.

``observed_mass`` is the valid-outcome mass at the first decision point, before
renormalization (the shared prefix above it, such as the space before digits, is not queried).
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field

from jevemu.backends.base import Backend, Capabilities, RenderedPrompt
from jevemu.render.labels import sequence_scheme_for
from jevemu.render.renderer import PromptRenderer
from jevemu.scoring.aggregate import (
    logsumexp,
    low_mass_warning,
    observed_mass,
    observed_surfaces,
    renormalize,
)
from jevemu.scoring.base import ScoreResult
from jevemu.scoring.first_token import (
    GAP,
    extend_prefill,
    grammar_first_tokens,
    token_ids,
    token_text,
)
from jevemu.scoring.missing_mass import fill_missing
from jevemu.types import JSONish, Question

__all__ = ["DEFAULT_TAU", "TrieStrategy"]

DEFAULT_TAU = 1e-4
"""Decision points reached with less probability than this are pruned."""


@dataclass(eq=False)
class _Node:
    ids: tuple[int, ...]
    text: str
    children: dict[str, _Node] = field(default_factory=dict)
    """Child token text -> child node."""
    ends: int | None = None
    """Index of the label whose path ends here."""
    labels: list[int] = field(default_factory=list)
    """Every label index in this subtree."""

    @property
    def n_outcomes(self) -> int:
        return len(self.children) + (self.ends is not None)


@dataclass(frozen=True)
class _Branch:
    """One decision point's conditional outcome logprobs (summing to 1) and call accounting."""

    children: dict[str, float]
    end: float | None
    observed_mass: float
    missing: tuple[str, ...]
    truncated: bool
    n_backend_calls: int
    prompt_tokens: int
    cached_tokens: int | None
    warnings: tuple[str, ...]


@dataclass
class _Totals:
    n_backend_calls: int = 0
    prompt_tokens: int = 0
    cached_tokens: int | None = 0
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)

    def add(self, branch: _Branch) -> None:
        self.n_backend_calls += branch.n_backend_calls
        self.prompt_tokens += branch.prompt_tokens
        if self.cached_tokens is not None and branch.cached_tokens is not None:
            self.cached_tokens += branch.cached_tokens
        else:
            self.cached_tokens = None
        self.truncated = self.truncated or branch.truncated
        self.warnings.extend(branch.warnings)


class TrieStrategy:
    """S3: token-trie chain rule over label texts, querying only decision points."""

    name = "trie"

    def __init__(self, *, tau: float = DEFAULT_TAU) -> None:
        if not 0.0 <= tau < 1.0:
            raise ValueError(f"tau must be in [0, 1), got {tau}")
        self.tau = tau

    def supports(self, capabilities: Capabilities, question: Question) -> bool:
        return capabilities.assistant_prefill

    async def score(
        self, backend: Backend, renderer: PromptRenderer, state: JSONish, question: Question
    ) -> ScoreResult:
        scheme = sequence_scheme_for(question)
        prompt = renderer.render(state, question, scheme)
        root = await _build(backend, [GAP + label for label in scheme.labels])
        log_tau = math.log(self.tau) if self.tau > 0 else -math.inf
        label_lp: list[float] = [-math.inf] * len(scheme)
        missing: set[int] = set()
        pruned: list[tuple[_Node, float]] = []
        totals = _Totals()
        first_mass: float | None = None
        frontier: list[tuple[_Node, float]] = [(root, 0.0)]
        while frontier:
            queries: list[tuple[_Node, float]] = []
            next_frontier: list[tuple[_Node, float]] = []
            for node, lp in frontier:
                if lp == -math.inf:
                    continue  # an observed zero: every label below keeps -inf
                if node.n_outcomes == 1:
                    if node.ends is not None:
                        label_lp[node.ends] = lp
                    else:
                        next_frontier.extend((child, lp) for child in node.children.values())
                elif lp < log_tau:
                    pruned.append((node, lp))
                    for index in node.labels:
                        label_lp[index] = lp
                else:
                    queries.append((node, lp))
            branches = await asyncio.gather(
                *(self._branch(backend, prompt, node) for node, _ in queries)
            )
            for (node, lp), branch in zip(queries, branches, strict=True):
                if first_mass is None:
                    first_mass = branch.observed_mass
                totals.add(branch)
                if node.ends is not None and branch.end is not None:
                    label_lp[node.ends] = lp + branch.end
                for text, child in node.children.items():
                    next_frontier.append((child, lp + branch.children[text]))
                for text in branch.missing:
                    missing.update(node.children[text].labels)
            frontier = next_frontier

        warnings = totals.warnings
        if pruned:
            n_labels = sum(len(node.labels) for node, _ in pruned)
            mass = math.exp(logsumexp(lp for _, lp in pruned))
            warnings.append(
                f"trie pruned {len(pruned)} subtrees below tau={self.tau:g}: {n_labels} labels "
                f"under {mass:.3g} total mass, each assigned its subtree's mass (upper bound)"
            )
        mass = 1.0 if first_mass is None else first_mass
        low = low_mass_warning(mass, backend.capabilities.logprobs_mode)
        if low is not None:
            warnings.append(low)
        raw = tuple(label_lp)
        return ScoreResult(
            keys=scheme.keys,
            logprobs=renormalize(raw),
            raw_logprobs=raw,
            observed_mass=mass,
            missing=tuple(key for i, key in enumerate(scheme.keys) if i in missing),
            truncated=totals.truncated or bool(pruned),
            strategy=self.name,
            label_scheme=scheme.name,
            n_backend_calls=totals.n_backend_calls,
            prompt_tokens=totals.prompt_tokens,
            cached_tokens=totals.cached_tokens,
            warnings=tuple(warnings),
        )

    async def _branch(self, backend: Backend, prompt: RenderedPrompt, node: _Node) -> _Branch:
        """Conditional outcome logprobs at ``node``: one first-token query plus any fills."""
        query = extend_prefill(prompt, node.text)
        caps = backend.capabilities
        texts = list(node.children)
        allowed: list[str] | None = None
        top_k = caps.top_logprobs_max
        if caps.structured_choice and node.ends is None:
            allowed = texts
            if caps.mask_reflected_in_logprobs:
                top_k = min(top_k, grammar_first_tokens(texts))
        dist, retokenized = await asyncio.gather(
            backend.next_token_logprobs(query, allowed=allowed, top_k=top_k),
            token_ids(backend, node.text),
        )
        seen = observed_surfaces(dist.top, texts)
        missing = {text: [text] for text in texts if text not in seen}
        fill = await fill_missing(backend, query, dist, missing, observed=seen)
        raw = {text: seen[text] if text in seen else fill.logprobs[text] for text in texts}
        where = f"trie node {node.text!r}"
        notes = [f"{where}: {note}" for note in fill.warnings]
        if node.text and retokenized != node.ids:
            notes.append(f"{where}: the prefill text re-tokenizes differently from the label path")
        seen_mass = observed_mass(seen.values())
        if node.ends is None:
            children = dict(zip(texts, renormalize([raw[t] for t in texts]), strict=True))
            end = None
            mass = seen_mass
        else:
            children, end, note = _split_end(raw)
            if note is not None:
                notes.append(f"{where}: {note}")
            mass = min(1.0, seen_mass + math.exp(end))
        return _Branch(
            children=children,
            end=end,
            observed_mass=mass,
            missing=tuple(missing),
            truncated=fill.truncated,
            n_backend_calls=1 + fill.n_backend_calls,
            prompt_tokens=dist.prompt_tokens + fill.prompt_tokens,
            cached_tokens=dist.cached_tokens,
            warnings=tuple(notes),
        )


def _split_end(raw: dict[str, float]) -> tuple[dict[str, float], float, str | None]:
    """Children logprobs and the end outcome ``1 - sum(p(children))`` from unconstrained reads."""
    total = logsumexp(raw.values())
    if total >= 0.0:
        children = dict(zip(raw, renormalize(list(raw.values())), strict=True))
        return children, -math.inf, "children took all the mass; the shorter label gets 0"
    return dict(raw), math.log1p(-math.exp(total)), None


async def _build(backend: Backend, texts: list[str]) -> _Node:
    """Trie over the token paths of ``texts``; validates that tokens are addressable by text."""
    paths = await asyncio.gather(*(token_ids(backend, text) for text in texts))
    unique = sorted({token for path in paths for token in path})
    decoded = dict(
        zip(unique, await asyncio.gather(*(token_text(backend, t) for t in unique)), strict=True)
    )
    root = _Node((), "")
    for index, (text, path) in enumerate(zip(texts, paths, strict=True)):
        if not path:
            raise ValueError(f"label text {text!r} tokenizes to nothing")
        pieces = [decoded[token] for token in path]
        if "".join(pieces) != text:
            raise ValueError(
                f"label text {text!r} does not decode token by token ({pieces}); "
                "the trie cannot address its tokens through the prefill text"
            )
        node = root
        node.labels.append(index)
        for token, piece in zip(path, pieces, strict=True):
            child = node.children.get(piece)
            if child is None:
                child = _Node((*node.ids, token), node.text + piece)
                node.children[piece] = child
            elif child.ids[-1] != token:
                raise ValueError(
                    f"tokens {child.ids[-1]} and {token} both decode to {piece!r} after "
                    f"{node.text!r}; the trie cannot tell them apart"
                )
            node = child
            node.labels.append(index)
        if node.ends is not None:
            raise ValueError(f"labels {texts[node.ends]!r} and {text!r} share one token path")
        node.ends = index
    return root
