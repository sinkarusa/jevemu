"""Render one question over one state into a byte-stable chat prompt ending in ``Answer:``.

Layouts are Jinja2 templates under ``templates/``. A layout template exports one top-level
variable per chat role it fills (``system``, ``user``); the renderer adds them in that order and
appends the assistant prefill ``Answer:`` (no trailing space, so the model generates the space
together with the label, e.g. ``" A"``).

``PromptRenderer(prefill=False)`` renders the same messages without the prefill, for backends
that cannot continue an assistant message (hosted chat APIs): the prompt ends in the user turn,
whose last line already asks for the label only, and the answer is the first token of the new
assistant turn. Its ``template_id`` carries a ``-noprefill`` suffix on the layout name, so
calibrators and caches never mix the two prompt forms.

Value rendering: strings verbatim; objects and arrays as JSON with 2-space indent, non-ASCII
kept as-is, and keys in the caller's order. Keys are not sorted because the caller's order may
carry meaning (a rubric's fields, context before the question) and Jev receives it as sent;
the output is still byte-stable because the same input always has the same key order. Null
descriptions and instructions are omitted, never rendered as ``null``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from typing import Literal, get_args

import jinja2
from jinja2.meta import find_referenced_templates

from jevemu.backends.base import ChatMessage, RenderedPrompt, Role
from jevemu.render.labels import NOUL_KEYS, LabelScheme, SchemeName
from jevemu.types import ChoiceQuestion, Description, JSONish, Question, ScoreQuestion

__all__ = ["DEFAULT_LAYOUT", "LAYOUTS", "PREFILL", "Layout", "PromptRenderer", "render_value"]

Layout = Literal["question_first", "state_first"]
LAYOUTS: tuple[Layout, ...] = get_args(Layout)
DEFAULT_LAYOUT: Layout = "state_first"
"""``PromptRenderer()``'s layout: +0.05-0.06 macro accuracy over ``question_first`` on every
preset tested (``docs/research/selection.md``)."""

PREFILL = "Answer:"
"""Assistant prefill every prompt ends with; the next token is the label."""

_ROLES: tuple[Role, ...] = ("system", "user")
"""Roles a layout template may export, in message order."""


def render_value(value: Description) -> str:
    """A Jev text-or-structure value as prompt text: strings verbatim, else pretty JSON."""
    if value is None:
        raise ValueError("null values are omitted from prompts, never rendered")
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


@dataclass(frozen=True)
class _Option:
    label: str
    description: Description
    name: str | None = None
    """Choice option key; ``None`` for Score levels and Noul criteria."""


@dataclass(frozen=True)
class _QuestionView:
    kind: str
    scheme: SchemeName
    instructions: Description
    options: tuple[_Option, ...]


class PromptRenderer:
    """Renders prompts for one layout; ``template_id`` = layout name (``-noprefill`` without the
    prefill) + template source hash."""

    def __init__(self, layout: Layout = DEFAULT_LAYOUT, *, prefill: bool = True) -> None:
        if layout not in LAYOUTS:
            raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")
        env = _environment()
        name = f"{layout}.j2"
        self.layout: Layout = layout
        self.prefill = prefill
        mode = "" if prefill else "-noprefill"
        self.template_id = f"{layout}{mode}-{_source_hash(env, name)[:12]}"
        self._template = env.get_template(name)

    def render(self, state: JSONish, question: Question, scheme: LabelScheme) -> RenderedPrompt:
        """Prompt asking ``question`` about ``state``, options labelled in ``scheme`` order.

        Raises ``ValueError`` if ``scheme``'s keys are not exactly the question's options.
        """
        module = self._template.make_module(
            {"state": state, "question": _question_view(question, scheme)}
        )
        messages = [
            ChatMessage(role, str(getattr(module, role)))
            for role in _ROLES
            if hasattr(module, role)
        ]
        if self.prefill:
            messages.append(ChatMessage("assistant", PREFILL))
        return RenderedPrompt(messages=tuple(messages), template_id=self.template_id)


def _question_view(question: Question, scheme: LabelScheme) -> _QuestionView:
    pairs = zip(scheme.labels, scheme.keys, strict=True)
    if isinstance(question, ChoiceQuestion):
        _check_keys(scheme, question.criteria)
        options = tuple(_Option(label, question.criteria[key], name=key) for label, key in pairs)
    elif isinstance(question, ScoreQuestion):
        _check_keys(scheme, (str(level) for level in range(len(question.criteria))))
        options = tuple(_Option(label, question.criteria[int(key)]) for label, key in pairs)
    else:
        _check_keys(scheme, NOUL_KEYS)
        criteria: dict[str, Description] = {
            str(key): value for key, value in (question.criteria or {}).items()
        }
        options = tuple(_Option(label, criteria.get(key)) for label, key in pairs)
    # Codes are rendered like letters ("AB) name", "Respond with the letter only."), so the
    # templates, and every prompt's template_id, stay as they were before the codes scheme.
    shown: SchemeName = "letters" if scheme.name == "codes" else scheme.name
    return _QuestionView(question.type, shown, question.instructions, options)


def _check_keys(scheme: LabelScheme, expected: Iterable[str]) -> None:
    wanted = list(expected)
    if len(scheme.keys) != len(wanted) or set(scheme.keys) != set(wanted):
        raise ValueError(
            f"label scheme keys {list(scheme.keys)} do not match the question's options {wanted}"
        )


@cache
def _environment() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("jevemu.render", "templates"),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
        undefined=jinja2.StrictUndefined,
    )
    env.filters["jsonish"] = render_value
    return env


def _source_hash(env: jinja2.Environment, name: str) -> str:
    """SHA-256 over ``name`` and every template it imports or includes, transitively."""
    loader = env.loader
    if loader is None:
        raise RuntimeError("jinja environment has no loader")
    sources: dict[str, str] = {}
    pending = [name]
    while pending:
        current = pending.pop()
        if current in sources:
            continue
        source, _, _ = loader.get_source(env, current)
        sources[current] = source
        for referenced in find_referenced_templates(env.parse(source)):
            if referenced is None:
                raise ValueError(f"template {current!r} references a template dynamically")
            pending.append(referenced)
    digest = hashlib.sha256()
    for current in sorted(sources):
        digest.update(current.encode() + b"\0" + sources[current].encode() + b"\0")
    return digest.hexdigest()
