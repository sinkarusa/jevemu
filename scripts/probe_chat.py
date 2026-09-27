"""Probe what a Chat Completions API does for logprob scoring (OpenAI or OpenRouter).

    uv run --with tiktoken python scripts/probe_chat.py --system openai --max-usd 1
    uv run python scripts/probe_chat.py --system openrouter --provider makora --max-usd 1
    uv run python scripts/probe_chat.py --system openai --sections basic,structured \\
        --compare-run runs/archive_noprefill/select/gpt-6-luna.first_token.state_first \\
        --out runs/probe_openai_structured

Measures what logprob scoring can rely on (results: ``docs/research/openai_probe_report.md`` and
``docs/research/openrouter_probe_report.md``) with the backend the runs use (the request bodies
are the backend's own), and writes ``<out>/report.json``
(aggregates only: no question text; default ``runs/probe_<system>``) plus a markdown summary on
stdout:

- ``basic``: the backend's request (reasoning off, ``logprobs``, its ``top_logprobs`` and output
  cap) succeeds; how many alternatives come back; finite values; billed reasoning tokens; the
  reported model and provider; the billed cost against the price table; whether one more
  ``top_logprobs`` is refused.
- ``prefill``: whether a final assistant message is continued or a new turn is opened.
- ``temperature``: whether the returned logprobs change with ``temperature`` (0 vs 1).
- ``surfaces``: per benchmark, ``--items`` screen items rendered like the run (``state_first``,
  no prefill): which token surfaces carry each label's mass, observed label mass (bare and
  space-prefixed surfaces, as scoring reads them) and labels missing from the top 5. These are
  the exact requests the screen run sends, so the run replays them from the response cache.
- ``structured``: the structured read (Structured Outputs, a JSON ``enum`` of the labels, read
  where the value starts) on ``--items`` screen items per benchmark the backend can score, with
  the run's exact requests (the screen run replays them from the response cache): whether the
  value token is located on every reply, what its top list holds besides labels (masked
  entries, anything else), whether the reply took its likeliest value, reasoning tokens,
  observed label mass, missing labels, completion tokens and cost per 1k items. With
  ``--compare-run`` (an earlier free-read run directory with the screen split), the argmax's
  agreement with that run's read of the same items; ``--repeat-items`` of the ARC and AG News
  items are sent again (cache bypassed) for the structured read's own argmax flip rate.
- ``determinism``: ``--repeats`` fresh calls per item (cache bypassed), sequential and
  concurrent, against the cached call: total variation between label distributions and argmax
  flips. This is the noise floor of every paired comparison.
- ``caching``: a prompt above 1,024 tokens sent twice with the API's default caching (OpenAI:
  and twice with ``prompt_cache_options.mode="explicit"``, no breakpoints): reported
  ``cached_tokens`` / ``cache_write_tokens``, and whether a cache hit changes the logprobs.
- ``tokenizer`` (OpenAI): with ``tiktoken`` installed, whether GPT-6 maps to an encoding and how
  ``o200k_base`` counts compare with billed ``prompt_tokens``.

Spend is capped by ``--max-usd`` (the backend's budget guard). ``--sections`` runs a subset.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import itertools
import json
import logging
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, get_args

from jevemu.backends import openai_chat, openrouter_chat
from jevemu.backends.base import ChatMessage, NextTokenDist, RenderedPrompt
from jevemu.backends.chat_api import (
    MASKED_LOGPROB,
    AnswerFormat,
    ChatCompletionsBackend,
    reasoning_tokens,
    response_usage,
)
from jevemu.backends.openai_chat import OpenAIChatBackend
from jevemu.backends.openrouter_chat import OpenRouterChatBackend, fetch_endpoints
from jevemu.bench.runner import frozen_split, load_records, records_path
from jevemu.errors import (
    ChatAPIAuthError,
    ChatAPIBudgetExceeded,
    ChatAPIEmptyReply,
    ChatAPIHTTPError,
    ChatAPIQuotaExceeded,
    ChatAPIResponseError,
)
from jevemu.jev_client import load_env_file
from jevemu.render import LabelScheme, PromptRenderer, scheme_for
from jevemu.scoring.aggregate import logsumexp
from jevemu.scoring.first_token import fits_top_k
from jevemu.scoring.missing_mass import upper_bound

BENCHMARKS = (
    "gpqa_diamond_idk",
    "lexam_en_idk",
    "mmlu_pro",
    "arc_challenge",
    "ag_news",
    "boolq",
    "sst5",
)
"""Benchmarks with at most 10 options (banking77 and clinc150 have too many for any top-k;
Yelp is dropped)."""
AGREEMENT_BENCHMARKS = ("arc_challenge", "ag_news")
"""Where the structured read's argmax is compared with the old read's and redrawn."""
SECTIONS = (
    "basic",
    "prefill",
    "temperature",
    "surfaces",
    "structured",
    "determinism",
    "caching",
    "tokenizer",
)

MCQ = RenderedPrompt(
    (
        ChatMessage(
            "user",
            "Question:\nWhich planet is known as the Red Planet?\n\nOptions:\nA) Venus\nB) Mars\n"
            "C) Jupiter\nD) Saturn\n\nRespond with the letter only.",
        ),
    ),
    template_id="probe/mcq",
)
DECORATIONS = "*()[].:_`'\"- \n\t"


def _top(dist: NextTokenDist) -> list[tuple[str, float]]:
    return [(t.token, t.logprob) for t in dist.top]


def _top_k(backend: ChatCompletionsBackend) -> int:
    return backend.capabilities.top_logprobs_max


def _body(backend: ChatCompletionsBackend, prompt: RenderedPrompt) -> dict[str, Any]:
    """The request the backend sends for ``prompt`` at its full top-k."""
    return backend.build_body(prompt, _top_k(backend))


def _with_max_tokens(body: Mapping[str, Any], n: int) -> dict[str, Any]:
    """``body`` with its output-token cap (whichever field the API uses) set to ``n``."""
    field = "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"
    return dict(body, **{field: n})


def _surface_class(token: str, label: str) -> str | None:
    """How ``token`` writes ``label`` (``bare``, ``space``, ``decorated``), or None."""
    if token == label:
        return "bare"
    if token == " " + label:
        return "space"
    if token.strip(DECORATIONS) == label:
        return "decorated"
    return None


def label_logprobs(dist: NextTokenDist, scheme: LabelScheme) -> list[float]:
    """Label logprobs as scoring reads them: bare and space surfaces merged, a missing label at
    the upper bound, renormalized."""
    found: dict[str, list[float]] = defaultdict(list)
    for token in dist.top:
        label = scheme.label_of_surface(token.token)
        if label is not None:
            found[label].append(token.logprob)
    bound = upper_bound(dist, leftover=True)
    raw = [logsumexp(found[label]) if label in found else bound for label in scheme.labels]
    total = logsumexp(raw)
    return [v - total for v in raw]


def total_variation(a: Sequence[float], b: Sequence[float]) -> float:
    return 0.5 * math.fsum(abs(math.exp(x) - math.exp(y)) for x, y in zip(a, b, strict=True))


def _pct(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


# --- probes --------------------------------------------------------------------------------------


async def probe_basic(backend: ChatCompletionsBackend) -> dict[str, Any]:
    body = _body(backend, MCQ)
    reply = await backend.request(body, bypass_cache=True)
    usage = reply.json.get("usage", {})
    choice = reply.json["choices"][0]
    content = choice["logprobs"]["content"]
    alternatives = content[0]["top_logprobs"]
    result: dict[str, Any] = {
        "status": "ok",
        "request": {k: v for k, v in body.items() if k != "messages"},
        "response_model": reply.json.get("model"),
        "provider": reply.json.get("provider"),
        "system_fingerprint": reply.json.get("system_fingerprint"),
        "finish_reason": choice.get("finish_reason"),
        "text": (choice.get("message") or {}).get("content"),
        "n_generated_tokens": len(content),
        "n_top_logprobs": len(alternatives),
        "all_finite": all(math.isfinite(a["logprob"]) for a in alternatives),
        "top": [(a["token"], a["logprob"]) for a in alternatives],
        "mass_top20": math.fsum(math.exp(a["logprob"]) for a in alternatives),
        "usage": usage,
        "billed_usd": reply.cost_usd,
        "table_usd": backend.price.cost(response_usage(reply.json)),
        "latency_ms": reply.latency_ms,
    }
    over_k = _top_k(backend) + 1
    over = dict(body, top_logprobs=over_k)
    try:
        answer = await backend.request(over, bypass_cache=True)
        returned = answer.json["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        result[f"top_logprobs_{over_k}"] = f"accepted, {len(returned)} returned"
    except ChatAPIHTTPError as exc:
        result[f"top_logprobs_{over_k}"] = f"HTTP {exc.status_code}: {str(exc.body)[:300]}"
    except ChatAPIResponseError as exc:  # OpenRouter: HTTP 200 with an upstream error body
        result[f"top_logprobs_{over_k}"] = f"refused: {str(exc)[:300]}"
    return result


async def probe_prefill(backend: ChatCompletionsBackend) -> dict[str, Any]:
    prompt = RenderedPrompt(
        (
            ChatMessage("user", "What is the capital of France?"),
            ChatMessage("assistant", "The capital of France is"),
        ),
        template_id="probe/prefill",
    )
    body = _with_max_tokens(backend.build_body(prompt, min(5, _top_k(backend))), 16)
    reply = await backend.request(body, bypass_cache=True)
    choice = reply.json["choices"][0]
    first = choice["logprobs"]["content"][0]
    text = choice["message"]["content"] or ""
    return {
        "text": text,
        "first_top": [(a["token"], a["logprob"]) for a in first["top_logprobs"]],
        "continued": text.lstrip().startswith("Paris") and not text.startswith("The"),
    }


async def probe_temperature(backend: ChatCompletionsBackend) -> dict[str, Any]:
    base = _body(backend, MCQ)
    zero = await backend.complete(base, bypass_cache=True)
    one = await backend.complete(dict(base, temperature=1), bypass_cache=True)
    a, b = dict(_top(zero)), dict(_top(one))
    shared = sorted(set(a) & set(b))
    return {
        "shared_tokens": len(shared),
        "max_abs_diff": max((abs(a[t] - b[t]) for t in shared), default=math.nan),
        "t0_top3": _top(zero)[:3],
        "t1_top3": _top(one)[:3],
    }


async def probe_surfaces(
    backend: ChatCompletionsBackend, out: Path, n_items: int
) -> tuple[dict[str, Any], list[tuple[str, RenderedPrompt, LabelScheme, NextTokenDist]]]:
    renderer = PromptRenderer("state_first", prefill=False)
    per: dict[str, Any] = {}
    kept: list[tuple[str, RenderedPrompt, LabelScheme, NextTokenDist]] = []
    for benchmark in BENCHMARKS:
        items = frozen_split(out, benchmark, "screen").items[:n_items]
        prompts = []
        for item in items:
            scheme = scheme_for(item.question)
            prompts.append((scheme, renderer.render(item.state, item.question, scheme)))
        dists = await asyncio.gather(
            *(
                backend.next_token_logprobs(p, allowed=None, top_k=_top_k(backend))
                for _, p in prompts
            )
        )
        mass_by_class: defaultdict[str, float] = defaultdict(float)
        observed, broad, n_missing, sampled_label = [], [], [], 0
        other_tokens: Counter[str] = Counter()
        for (scheme, prompt), dist in zip(prompts, dists, strict=True):
            item_mass: defaultdict[str, float] = defaultdict(float)
            labels_seen = set()
            for token in dist.top:
                p = math.exp(token.logprob)
                classes = [
                    (label, c)
                    for label in scheme.labels
                    if (c := _surface_class(token.token, label)) is not None
                ]
                if classes:
                    label, cls = classes[0]
                    item_mass[cls] += p
                    if cls != "decorated":
                        labels_seen.add(label)
                else:
                    other_tokens[token.token] += 1
            for cls, p in item_mass.items():
                mass_by_class[cls] += p / len(items)
            observed.append(item_mass["bare"] + item_mass["space"])
            broad.append(sum(item_mass.values()))
            n_missing.append(len(scheme.labels) - len(labels_seen))
            sampled_label += scheme.label_of_surface(dist.sampled.token) is not None
            kept.append((benchmark, prompt, scheme, dist))
        per[benchmark] = {
            "n": len(items),
            "n_labels": len(scheme_for(items[0].question)),
            "mean_mass_by_surface": dict(mass_by_class),
            "observed_mass_mean": statistics.fmean(observed),
            "observed_mass_p05": _pct(observed, 0.05),
            "observed_mass_min": min(observed),
            "share_observed_ge_0.99": sum(m >= 0.99 for m in observed) / len(observed),
            "decorated_mass_mean": statistics.fmean(
                b - o for b, o in zip(broad, observed, strict=True)
            ),
            "sampled_is_label": sampled_label / len(items),
            "mean_labels_missing_top20": statistics.fmean(n_missing),
            "items_with_missing_labels": sum(m > 0 for m in n_missing),
            "frequent_non_label_tokens": other_tokens.most_common(8),
        }
    return per, kept


def inspect_value(raw: Mapping[str, Any], answer: AnswerFormat) -> dict[str, Any]:
    """Where a structured reply's value starts, located independently of the backend's parser,
    and what is listed at that token: labels (a piece of a value after the token's prefix),
    masked entries (at or below ``MASKED_LOGPROB``) and anything else."""
    content = raw["choices"][0]["logprobs"]["content"]
    tokens = [str(entry["token"]) for entry in content]
    start = answer.value_start("".join(tokens))
    if start is None:
        return {"found": False}
    ends = list(itertools.accumulate(len(token) for token in tokens))
    index = bisect.bisect_left(ends, start)  # the token holding the value's opening quote
    offset = ends[index] - len(tokens[index])
    prefix = tokens[index][: start - offset]
    merged = tokens[index] != prefix
    if not merged:
        index, prefix = index + 1, ""
    if index == len(tokens):
        return {"found": False}
    entry = content[index]
    labels, masked, other = [], 0, []
    for alt in entry.get("top_logprobs") or []:
        token, logprob = str(alt["token"]), float(alt["logprob"])
        if logprob <= MASKED_LOGPROB:
            masked += 1
        elif token.startswith(prefix) and answer.value_of(token[len(prefix) :]) is not None:
            labels.append(logprob)
        else:
            other.append((token, logprob))
    finite = [*labels, *(lp for _, lp in other)]
    text = "".join(tokens)
    return {
        "found": answer.value_of(tokens[index][len(prefix) :]) is not None,
        "index": index,
        "merged": merged,
        # The stream may end in a special token the text lacks, but must hold the whole text.
        "unreported_tokens": not text.startswith(raw["choices"][0]["message"]["content"]),
        "unreported_value_quote": text[start - 1] != '"',
        "label_mass": math.fsum(math.exp(v) for v in labels),
        "masked": masked,
        "other": other,
        "masked_floor": min(
            (float(a["logprob"]) for a in entry.get("top_logprobs") or []), default=math.nan
        ),
        "greedy": not finite or float(entry["logprob"]) >= max(finite) - 1e-6,
    }


def structured_label_logprobs(dist: NextTokenDist, scheme: LabelScheme) -> list[float]:
    """Label logprobs as the structured read's scoring gives them: listed labels as read, a
    missing one at the leftover mass, renormalized."""
    found = {t.token: t.logprob for t in dist.top}
    bound = upper_bound(dist, masked=True)
    raw = [found.get(label, bound) for label in scheme.labels]
    total = logsumexp(raw)
    return [v - total for v in raw]


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


async def probe_structured(
    backend: ChatCompletionsBackend,
    splits: Path,
    n_items: int,
    compare_run: Path | None,
    repeat_items: int,
) -> dict[str, Any]:
    """The structured read on ``n_items`` screen items per benchmark the backend can score, with
    the run's exact requests (the screen run replays them from the response cache).

    Per benchmark: whether the value token is located and parsed on every reply, what its top
    list holds besides labels, whether the reply took its most likely value, reasoning tokens,
    observed label mass, missing labels, completion tokens and cost. With ``compare_run`` (the
    old free-read run directory), the agreement of the argmax with that run's recorded read of
    the same items; ``repeat_items`` items of those benchmarks are sent again (cache bypassed)
    for the structured read's own argmax flip rate.
    """
    renderer = PromptRenderer("state_first", prefill=False)
    top_k = _top_k(backend)
    per: dict[str, Any] = {}
    repeatable: list[tuple[str, dict[str, Any], LabelScheme, AnswerFormat, NextTokenDist]] = []
    for benchmark in BENCHMARKS:
        items = frozen_split(splits, benchmark, "screen").items[:n_items]
        if not all(fits_top_k(backend.capabilities, item.question) for item in items):
            per[benchmark] = {"skipped": f"more options than the top {top_k}"}
            continue
        jobs = []
        for item in items:
            scheme = scheme_for(item.question)
            answer = AnswerFormat(scheme.labels)
            prompt = renderer.render(item.state, item.question, scheme)
            jobs.append((item.item_id, backend.request_body(prompt, top_k, answer), scheme, answer))

        async def read(
            body: dict[str, Any], answer: AnswerFormat
        ) -> tuple[NextTokenDist | str, dict[str, Any] | None]:
            try:
                dist = await backend.complete(body)
            except ChatAPIResponseError as exc:
                return str(exc).split(":")[0][:120], None
            return dist, (await backend.request(body)).json  # a cache hit: the reply read

        replies = await asyncio.gather(*(read(body, answer) for _, body, _, answer in jobs))
        old = (
            load_records(records_path(compare_run.parent, compare_run.name, benchmark, "screen"))
            if compare_run is not None
            else {}
        )
        failures: Counter[str] = Counter()
        values: list[dict[str, Any]] = []
        masses, missing, completion, costs, reasoning = [], [], [], [], Counter[str]()
        agree = compared = 0
        for (item_id, body, scheme, answer), (dist, raw) in zip(jobs, replies, strict=True):
            if isinstance(dist, str) or raw is None:
                failures[str(dist)] += 1
                continue
            values.append(inspect_value(raw, answer))
            masses.append(math.fsum(math.exp(t.logprob) for t in dist.top))
            missing.append(len(scheme.labels) - len(dist.top))
            completion.append(response_usage(raw).output_tokens)
            costs.append(backend.call_cost(raw))
            reasoning[str(reasoning_tokens(raw))] += 1
            lps = structured_label_logprobs(dist, scheme)
            record = old.get(item_id)
            if record is not None and record.diagnostics is not None:
                before = record.diagnostics.raw_probabilities or {}
                if before:
                    compared += 1
                    agree += scheme.keys[_argmax(lps)] == max(before, key=before.__getitem__)
            if benchmark in AGREEMENT_BENCHMARKS:
                repeatable.append((benchmark, body, scheme, answer, dist))
        n = len(values)
        others = Counter(t for v in values for t, _ in v.get("other", []))
        per[benchmark] = {
            "n": len(jobs),
            "n_labels": len(jobs[0][2].labels),
            "read_failures": dict(failures),
            "value_found": sum(v["found"] for v in values),
            "value_token_index": dict(Counter(v.get("index") for v in values)),
            "value_merged_with_quote": sum(v.get("merged", False) for v in values),
            "replies_with_unreported_tokens": sum(v.get("unreported_tokens", 0) for v in values),
            "value_quote_unreported": sum(v.get("unreported_value_quote", 0) for v in values),
            "items_with_non_label_alternatives": sum(bool(v.get("other")) for v in values),
            "non_label_alternatives": others.most_common(6),
            "raw_label_mass_at_value_min": min(
                (v.get("label_mass", 0.0) for v in values), default=math.nan
            ),
            "masked_floor": sorted({v.get("masked_floor") for v in values if v.get("masked")}),
            "greedy": sum(v.get("greedy", False) for v in values),
            "reasoning_tokens": dict(reasoning),
            "observed_mass_mean": statistics.fmean(masses) if n else math.nan,
            "observed_mass_p05": _pct(masses, 0.05),
            "observed_mass_min": min(masses, default=math.nan),
            "items_with_missing_labels": sum(m > 0 for m in missing),
            "missing_labels_per_item": statistics.fmean(missing) if n else math.nan,
            "completion_tokens": {
                "mean": statistics.fmean(completion) if n else math.nan,
                "min": min(completion, default=0),
                "max": max(completion, default=0),
            },
            "usd_per_1k_items": 1000 * statistics.fmean(costs) if n else math.nan,
            "argmax_agrees_with_old_read": f"{agree}/{compared}",
        }

    async def again(body: dict[str, Any]) -> NextTokenDist | None:
        try:
            return await backend.complete(body, bypass_cache=True)
        except ChatAPIResponseError:
            return None

    picked = repeatable[: max(0, repeat_items)]
    redrawn = await asyncio.gather(*(again(body) for _, body, _, _, _ in picked))
    flips = drawn = 0
    for (_, _, scheme, _, reference), redraw in zip(picked, redrawn, strict=True):
        if redraw is None:
            continue
        drawn += 1
        flips += _argmax(structured_label_logprobs(redraw, scheme)) != _argmax(
            structured_label_logprobs(reference, scheme)
        )
    return {
        "benchmarks": per,
        "repeat": {
            "benchmarks": list(AGREEMENT_BENCHMARKS),
            "sent": len(picked),
            "read": drawn,
            "argmax_flips": flips,
        },
    }


async def probe_determinism(
    backend: ChatCompletionsBackend,
    kept: Sequence[tuple[str, RenderedPrompt, LabelScheme, NextTokenDist]],
    n_items: int,
    repeats: int,
) -> dict[str, Any]:
    by_benchmark: dict[str, list[tuple[str, RenderedPrompt, LabelScheme, NextTokenDist]]] = (
        defaultdict(list)
    )
    for entry in kept:
        by_benchmark[entry[0]].append(entry)
    picked: list[tuple[str, RenderedPrompt, LabelScheme, NextTokenDist]] = []
    rows = list(by_benchmark.values())
    while len(picked) < n_items and any(rows):
        for row in rows:
            if row and len(picked) < n_items:
                picked.append(row.pop(0))

    async def fresh(prompt: RenderedPrompt) -> NextTokenDist | None:
        try:
            return await backend.complete(_body(backend, prompt), bypass_cache=True)
        except ChatAPIEmptyReply:
            return None

    results: dict[str, Any] = {}
    for mode in ("sequential", "concurrent"):
        tvs: list[float] = []
        flips = 0
        max_lp_spread: list[float] = []
        identities: Counter[str] = Counter()
        empties_before, unanswered = backend.empty_replies, 0
        for _, prompt, scheme, reference in picked:
            if mode == "sequential":
                drawn = [await fresh(prompt) for _ in range(repeats)]
            else:
                drawn = list(await asyncio.gather(*(fresh(prompt) for _ in range(repeats))))
            samples = [s for s in drawn if s is not None]
            unanswered += len(drawn) - len(samples)
            ref = label_logprobs(reference, scheme)
            ref_top = max(range(len(ref)), key=ref.__getitem__)
            for sample in samples:
                lps = label_logprobs(sample, scheme)
                tvs.append(total_variation(ref, lps))
                flips += max(range(len(lps)), key=lps.__getitem__) != ref_top
                if sample.api is not None:
                    identities[f"{sample.api.system_fingerprint} @ {sample.api.provider}"] += 1
            top_token = reference.top[0].token
            values = [
                lp
                for s in (reference, *samples)
                for lp in [next((t.logprob for t in s.top if t.token == top_token), None)]
                if lp is not None
            ]
            max_lp_spread.append(max(values) - min(values))
        results[mode] = {
            "items": len(picked),
            "repeats": repeats,
            "samples": len(tvs),
            "empty_replies_resent": backend.empty_replies - empties_before,
            "unanswered_after_resends": unanswered,
            "tv_mean": statistics.fmean(tvs),
            "tv_p95": _pct(tvs, 0.95),
            "tv_max": max(tvs),
            "share_tv_zero": sum(tv == 0.0 for tv in tvs) / len(tvs),
            "share_tv_below_0.01": sum(tv < 0.01 for tv in tvs) / len(tvs),
            "argmax_flips": flips,
            "top_token_logprob_spread_max": max(max_lp_spread),
            "fingerprint_at_provider": dict(identities),
        }
    return results


async def probe_caching(backend: ChatCompletionsBackend) -> dict[str, Any]:
    paragraph = (
        "The committee reviewed each quarterly filing, compared the reported figures with the "
        "audited statements, and recorded every discrepancy in a shared ledger. "
    )
    state = f"{time.time_ns()} " + paragraph * 60  # unique prefix: no earlier cache entry
    prompt = RenderedPrompt(
        (
            ChatMessage(
                "user",
                f"State:\n{state}\n\nQuestion:\nIs the ledger shared?\n\nRespond with Yes or No "
                "only.",
            ),
        ),
        template_id="probe/cache",
    )
    base = _body(backend, prompt)
    variants = [("default", base)]
    if isinstance(backend, OpenAIChatBackend):
        variants.append(("explicit", dict(base, prompt_cache_options={"mode": "explicit"})))
    results: dict[str, Any] = {}
    for name, body in variants:
        try:
            first = await backend.request(body, bypass_cache=True)
            await asyncio.sleep(2.0)
            second = await backend.request(body, bypass_cache=True)
        except ChatAPIHTTPError as exc:
            results[name] = {"status": f"HTTP {exc.status_code}: {str(exc.body)[:300]}"}
            continue
        tops = []
        for reply in (first, second):
            content = reply.json["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            tops.append({a["token"]: a["logprob"] for a in content})
        shared = set(tops[0]) & set(tops[1])
        results[name] = {
            "status": "ok",
            "usage": [first.json.get("usage"), second.json.get("usage")],
            "cost_usd": [first.cost_usd, second.cost_usd],
            "latency_ms": [first.latency_ms, second.latency_ms],
            "logprob_max_abs_diff": max(
                (abs(tops[0][t] - tops[1][t]) for t in shared), default=math.nan
            ),
        }
    return results


async def probe_tokenizer(
    backend: ChatCompletionsBackend,
    kept: Sequence[tuple[str, RenderedPrompt, LabelScheme, NextTokenDist]],
) -> dict[str, Any]:
    """Whether tiktoken maps the model, and whether ``o200k_base`` is its encoding: if it is,
    billed ``prompt_tokens`` minus the ``o200k_base`` count of the message contents is the same
    chat-format overhead for every single-message prompt."""
    if not isinstance(backend, OpenAIChatBackend):
        return {"skipped": "tiktoken encodings are OpenAI's"}
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError:
        return {"tiktoken": "not installed"}
    result: dict[str, Any] = {"tiktoken": tiktoken.__version__}
    try:
        result["encoding_for_model"] = tiktoken.encoding_for_model(backend.model).name
    except KeyError as exc:
        result["encoding_for_model"] = f"none ({exc})"
    encoding = tiktoken.get_encoding("o200k_base")
    overheads = Counter(
        dist.prompt_tokens - sum(len(encoding.encode(m.content)) for m in prompt.messages)
        for _, prompt, _, dist in kept
    )
    result["o200k_overhead_counts"] = dict(overheads.most_common())
    result["o200k_matches"] = len(overheads) == 1
    result["single_token_labels_o200k"] = {
        s: len(encoding.encode(s)) == 1
        for s in ("A", " A", "Z", " Z", "0", " 9", "Yes", " Yes", "No", " No", "**")
    }
    return result


# --- main ----------------------------------------------------------------------------------------


def summary(report: Mapping[str, Any]) -> str:
    lines = [f"# {report['service']} probe: `{report['model']}`", ""]
    for key, value in report.items():
        if key == "surfaces" and "error" not in value:
            lines.append("## surfaces")
            lines.append("")
            lines.append(
                "| Benchmark | n | labels | obs. mass mean | p05 | min | >=0.99 | decorated | "
                "missing/item | sampled=label |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for name, row in value.items():
                lines.append(
                    f"| {name} | {row['n']} | {row['n_labels']} | "
                    f"{row['observed_mass_mean']:.4f} | {row['observed_mass_p05']:.4f} | "
                    f"{row['observed_mass_min']:.4f} | {row['share_observed_ge_0.99']:.2f} | "
                    f"{row['decorated_mass_mean']:.4f} | {row['mean_labels_missing_top20']:.2f} | "
                    f"{row['sampled_is_label']:.2f} |"
                )
            lines.append("")
        elif key == "structured" and "error" not in value:
            lines.append("## structured")
            lines.append("")
            lines.append(
                "| Benchmark | n | failed | value found | merged | non-label alts | greedy | "
                "reasoning | mass mean | p05 | min | items missing | missing/item | "
                "out tok | $/1k | argmax = old |"
            )
            lines.append("| --- |" + " ---: |" * 15)
            for name, row in value["benchmarks"].items():
                if "skipped" in row:
                    lines.append(f"| {name} | {row['skipped']} |" + " |" * 14)
                    continue
                lines.append(
                    f"| {name} | {row['n']} | {sum(row['read_failures'].values())} | "
                    f"{row['value_found']} | {row['value_merged_with_quote']} | "
                    f"{row['items_with_non_label_alternatives']} | {row['greedy']} | "
                    f"{row['reasoning_tokens']} | {row['observed_mass_mean']:.4f} | "
                    f"{row['observed_mass_p05']:.4f} | {row['observed_mass_min']:.4f} | "
                    f"{row['items_with_missing_labels']} | "
                    f"{row['missing_labels_per_item']:.2f} | "
                    f"{row['completion_tokens']['mean']:.1f} | {row['usd_per_1k_items']:.4f} | "
                    f"{row['argmax_agrees_with_old_read']} |"
                )
            lines.append("")
            lines.append(f"Redrawn: `{json.dumps(value['repeat'])}`")
            lines.append("")
        elif isinstance(value, dict):
            lines.append(f"## {key}")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(value, indent=1, ensure_ascii=False)[:4000])
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


async def make_backend(args: argparse.Namespace) -> ChatCompletionsBackend:
    if args.system == "openai":
        return OpenAIChatBackend(
            args.model,
            max_usd=args.max_usd,
            max_concurrency=16,
            prompt_cache_mode=args.prompt_cache_mode,
        )
    return OpenRouterChatBackend(
        args.model,
        endpoints=await fetch_endpoints(args.model),
        providers=args.provider,
        top_logprobs=args.top_logprobs,
        max_tokens=args.max_tokens,
        max_usd=args.max_usd,
        max_concurrency=16,
    )


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file()
    backend = await make_backend(args)
    report: dict[str, Any] = {
        "service": backend.service,
        "model": args.model,
        "price": backend.price.describe(),
        "health_flags": dict((await backend.health()).flags),
    }
    async with backend:
        started = time.monotonic()

        async def section(name: str, probe: Awaitable[dict[str, Any]]) -> None:
            """One probe; a failure is recorded and the others still run."""
            try:
                report[name] = await probe
            except (ChatAPIBudgetExceeded, ChatAPIQuotaExceeded, ChatAPIAuthError):
                raise
            except Exception as exc:
                logging.exception("probe %s failed", name)
                report[name] = {"error": f"{type(exc).__name__}: {exc}"[:1000]}

        kept: list[tuple[str, RenderedPrompt, LabelScheme, NextTokenDist]] = []

        async def surfaces() -> dict[str, Any]:
            per, found = await probe_surfaces(backend, args.splits, args.items)
            kept.extend(found)
            return per

        probes: dict[str, Callable[[], Awaitable[dict[str, Any]]]] = {
            "basic": lambda: probe_basic(backend),
            "prefill": lambda: probe_prefill(backend),
            "temperature": lambda: probe_temperature(backend),
            "surfaces": surfaces,
            "structured": lambda: probe_structured(
                backend, args.splits, args.items, args.compare_run, args.repeat_items
            ),
            "determinism": lambda: probe_determinism(
                backend, kept, args.determinism_items, args.repeats
            ),
            "caching": lambda: probe_caching(backend),
            "tokenizer": lambda: probe_tokenizer(backend, kept),
        }
        try:
            for name in args.sections:
                await section(name, probes[name]())
        finally:
            report["spend"] = {
                "spent_usd": backend.spent_usd,
                "network_calls": backend.network_calls,
                "cache_hits": backend.cache_hits,
                "empty_replies": backend.empty_replies,
                "identities": [
                    {"model": m, "system_fingerprint": f, "provider": p, "responses": n}
                    for (m, f, p), n in backend.identities.most_common()
                ],
                "elapsed_s": time.monotonic() - started,
            }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--system", choices=("openai", "openrouter"), required=True)
    parser.add_argument(
        "--model",
        help=f"default: openai {openai_chat.DEFAULT_MODEL}, "
        f"openrouter {openrouter_chat.DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--provider",
        type=lambda v: [s.strip() for s in v.split(",") if s.strip()],
        default=[],
        help="openrouter: pinned provider slugs",
    )
    parser.add_argument("--top-logprobs", type=int, default=openrouter_chat.DEFAULT_TOP_LOGPROBS)
    parser.add_argument("--max-tokens", type=int, default=openrouter_chat.DEFAULT_MAX_TOKENS)
    parser.add_argument("--out", type=Path, help="default: runs/probe_<system>")
    parser.add_argument(
        "--splits", type=Path, default=Path("runs/select"), help="dir holding splits/"
    )
    parser.add_argument("--items", type=int, default=50, help="screen items per benchmark")
    parser.add_argument("--determinism-items", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--prompt-cache-mode",
        choices=get_args(openai_chat.PromptCacheMode),
        help="openai: prompt_cache_options.mode (the runs use explicit)",
    )
    parser.add_argument(
        "--compare-run", type=Path, help="structured: a free-read run dir with the screen split"
    )
    parser.add_argument(
        "--repeat-items", type=int, default=100, help="structured: ARC/AG News items redrawn"
    )
    parser.add_argument(
        "--sections",
        type=lambda v: [s.strip() for s in v.split(",") if s.strip()],
        default=list(SECTIONS),
        help=f"comma-separated subset of {','.join(SECTIONS)}",
    )
    parser.add_argument("--max-usd", type=float, default=1.0)
    args = parser.parse_args()
    unknown = sorted(set(args.sections) - set(SECTIONS))
    if unknown:
        parser.error(f"unknown sections {unknown}")
    if args.model is None:
        args.model = (openai_chat if args.system == "openai" else openrouter_chat).DEFAULT_MODEL
    if args.out is None:
        args.out = Path(f"runs/probe_{args.system}")
    report = asyncio.run(main_async(args))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary(report))
    print(f"\nwrote {args.out / 'report.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
