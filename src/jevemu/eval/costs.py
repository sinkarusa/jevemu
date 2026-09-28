"""Prices of the evaluated systems and the cost of each answered item.

Two kinds of price:

- :class:`TokenPrice`, for APIs: $ per 1M input, cached-input, cache-write and output tokens, a
  rate multiplier (Batch/Flex processing) and a long-context tier (a request with more input
  tokens than the threshold pays multiplied input/cache and output rates for the whole request).
  :data:`TOKEN_PRICES` holds the built-in entries, each with its source and as-of date.
- :class:`GpuTimePrice`, for local GPU systems (the vLLM emulator, CLM): $ per GPU-hour. An
  explicit rate (``run_split.py stats --gpu-usd-per-hour`` or :data:`GPU_USD_PER_HOUR_ENV`)
  wins; otherwise the rate is an electricity estimate, average board power x electricity price
  (:meth:`GpuTimePrice.from_power`, W x $/Wh = $/h). Its defaults: :data:`DEFAULT_GPU_WATTS`
  (the RTX 3090 draws about 410 W while vLLM serves a run, as observed by the user) and
  :data:`DEFAULT_USD_PER_WH` ($0.00027/Wh = $0.27/kWh, New Jersey average), $0.1107 per GPU-hour.
  Override with ``--gpu-watts``/``--usd-per-wh`` or :data:`GPU_WATTS_ENV`/:data:`USD_PER_WH_ENV`.
  Only the GPU's electricity is counted: no hardware amortization, rest of the machine or cooling.

:meth:`PriceBook.resolve` picks a run's price from its manifest's ``system``: a run with a
``price_id`` (an emulator over a paid API, such as ``gpt-6-luna``) is token-priced by the book's
entry for it, else by the ``token_price`` the run recorded (OpenRouter's price snapshot, the
:class:`TokenPrice` fields); any other run of a kind in :data:`GPU_TIME_KINDS` (the emulator,
CLM) is time-priced; anything else (Jev) is looked up by ``system["model"]`` in the book's token
prices.

Item costs (:func:`item_costs`, one records file at a time):

- **Token-priced.** :func:`record_usage` reads the item's tokens. ``input_tokens`` counts every
  prompt token, cached and cache-write tokens included (OpenAI's convention); cached-input and
  cache-write counts come from the usage's extra fields :data:`~jevemu.types.CACHED_INPUT_FIELD`
  and :data:`~jevemu.types.CACHE_WRITE_FIELD` (the emulator's ``diagnostics.cached_tokens`` also
  counts as cached input). When the system recorded its own charge for a billed call (Jev's
  ``cost_usd`` on a non-cache-hit, a paid-API emulator's ``diagnostics.api.cost_usd``), that
  charge is the item's cost, and a disagreement of more than
  :data:`MISMATCH_TOLERANCE` with the price table is flagged. A response-cache hit (recorded
  cost 0) is costed at the table price, so replaying a cache does not make a system look
  cheaper. An item without usage (a failed call) costs 0.
- **Time-priced.** A records file's GPU time is the summed wall time of the manifest
  invocations that wrote it (each one's ``started_at`` to ``finished_at``). That excludes server
  startup and any idle time between runs, and assumes nothing else ran on the GPU meanwhile.
  Records do not say which invocation wrote them, so the file's total is split over all its
  items in proportion to input + output tokens. An item without usage gets the mean weight of
  the items with usage (equal weights if none has any), so the items' shares always sum to the
  file's GPU time. Items CLM answered from its response cache (``cached``) took no GPU time, so
  a run replayed from that cache is cheaper than a fresh one.
"""

from __future__ import annotations

import dataclasses
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from jevemu.bench.manifest import Invocation
from jevemu.bench.runner import ItemRecord
from jevemu.jev_client import USD_PER_INPUT_TOKEN
from jevemu.types import CACHE_WRITE_FIELD, CACHED_INPUT_FIELD

__all__ = [
    "DEFAULT_GPU_WATTS",
    "DEFAULT_USD_PER_WH",
    "GPT_6_LUNA",
    "GPT_6_LUNA_BATCH",
    "GPT_6_LUNA_FLEX",
    "GPU_TIME_KINDS",
    "GPU_USD_PER_HOUR_ENV",
    "GPU_WATTS_ENV",
    "JEV_1_13_0",
    "MISMATCH_TOLERANCE",
    "TOKEN_PRICES",
    "USD_PER_WH_ENV",
    "GpuTimePrice",
    "ItemCost",
    "Price",
    "PriceBook",
    "TokenPrice",
    "TokenUsage",
    "allocate",
    "gpu_usd_per_hour_from_env",
    "invocation_seconds",
    "item_costs",
    "record_usage",
]

GPU_USD_PER_HOUR_ENV = "JEVEMU_GPU_USD_PER_HOUR"
GPU_WATTS_ENV = "JEVEMU_GPU_WATTS"
USD_PER_WH_ENV = "JEVEMU_USD_PER_WH"
DEFAULT_GPU_WATTS = 410.0
"""RTX 3090 board power while vLLM serves a run (about 410 W, observed by the user)."""
DEFAULT_USD_PER_WH = 0.00027
"""Placeholder: $0.27 per kWh, the New Jersey average electricity price (user estimate)."""
MISMATCH_TOLERANCE = 0.01
"""Relative difference between a recorded charge and the price table that gets flagged."""
GPU_TIME_KINDS = frozenset({"emulator", "clm"})
"""Manifest ``system["kind"]`` values of local GPU systems, priced by GPU time."""

_PER_MTOK = 1e-6


@dataclass(frozen=True)
class TokenUsage:
    """One item's tokens. ``input_tokens`` includes the cached and cache-write tokens."""

    input_tokens: int
    output_tokens: int
    cached_input_tokens: int | None = None
    """``None``: the system does not report prompt-cache hits."""
    cache_write_tokens: int = 0

    def __post_init__(self) -> None:
        counts = (self.input_tokens, self.output_tokens, self.cache_write_tokens)
        if min(counts) < 0 or (self.cached_input_tokens or 0) < 0:
            raise ValueError(f"token counts must be >= 0: {self}")
        if (self.cached_input_tokens or 0) + self.cache_write_tokens > self.input_tokens:
            raise ValueError(f"cached and cache-write tokens exceed the input tokens: {self}")


@dataclass(frozen=True)
class TokenPrice:
    """A per-token API price (module docstring)."""

    price_id: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cached_input_usd_per_mtok: float | None = None
    """``None``: cached input is billed at the input rate."""
    cache_write_usd_per_mtok: float | None = None
    """``None``: cache writes are billed at the input rate."""
    rate_multiplier: float = 1.0
    """Applied to every rate (Batch and Flex processing: 0.5)."""
    long_context_threshold: int | None = None
    """A request with more input tokens than this pays the long-context multipliers."""
    long_context_input_multiplier: float = 1.0
    """On the input, cached-input and cache-write rates."""
    long_context_output_multiplier: float = 1.0
    source: str = ""
    as_of: str = ""

    def cost(self, usage: TokenUsage) -> float:
        """$ for one request (one item) with ``usage``."""
        long = (
            self.long_context_threshold is not None
            and usage.input_tokens > self.long_context_threshold
        )
        input_mult = self.long_context_input_multiplier if long else 1.0
        output_mult = self.long_context_output_multiplier if long else 1.0
        cached = usage.cached_input_tokens or 0
        uncached = usage.input_tokens - cached - usage.cache_write_tokens
        cached_rate = self.cached_input_usd_per_mtok
        write_rate = self.cache_write_usd_per_mtok
        input_usd = (
            uncached * self.input_usd_per_mtok
            + cached * (self.input_usd_per_mtok if cached_rate is None else cached_rate)
            + usage.cache_write_tokens
            * (self.input_usd_per_mtok if write_rate is None else write_rate)
        )
        output_usd = usage.output_tokens * self.output_usd_per_mtok
        return (
            (input_usd * input_mult + output_usd * output_mult) * self.rate_multiplier * _PER_MTOK
        )

    def describe(self) -> str:
        rates = [f"input ${self.input_usd_per_mtok:g}"]
        if self.cached_input_usd_per_mtok is not None:
            rates.append(f"cached input ${self.cached_input_usd_per_mtok:g}")
        if self.cache_write_usd_per_mtok is not None:
            rates.append(f"cache writes ${self.cache_write_usd_per_mtok:g}")
        rates.append(f"output ${self.output_usd_per_mtok:g}")
        text = f"`{self.price_id}`: {', '.join(rates)} per 1M tokens"
        if self.rate_multiplier != 1.0:
            text += f", all rates x{self.rate_multiplier:g}"
        if self.long_context_threshold is not None:
            text += (
                f"; above {self.long_context_threshold:,} input tokens "
                f"input/cache x{self.long_context_input_multiplier:g}, "
                f"output x{self.long_context_output_multiplier:g}"
            )
        if self.source:
            text += f" ({self.source}, as of {self.as_of})"
        return text


@dataclass(frozen=True)
class GpuTimePrice:
    """The local GPU, priced by wall time."""

    usd_per_gpu_hour: float | None
    """``None``: no rate; costs are reported in GPU-hours only."""
    price_id: str = "local-gpu"
    basis: str | None = None
    """How the rate was derived, e.g. the electricity estimate behind :meth:`from_power`."""

    def __post_init__(self) -> None:
        if self.usd_per_gpu_hour is not None and not (
            math.isfinite(self.usd_per_gpu_hour) and self.usd_per_gpu_hour >= 0
        ):
            raise ValueError(
                f"usd_per_gpu_hour must be finite and >= 0, got {self.usd_per_gpu_hour}"
            )

    @classmethod
    def from_power(cls, watts: float, usd_per_wh: float) -> GpuTimePrice:
        """Electricity cost of running the GPU: ``watts`` x ``usd_per_wh`` $ per hour."""
        for name, value in (("watts", watts), ("usd_per_wh", usd_per_wh)):
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be finite and >= 0, got {value}")
        return cls(
            watts * usd_per_wh,
            basis=f"electricity estimate: {watts:g} W x ${usd_per_wh:g}/Wh",
        )

    def cost(self, gpu_seconds: float) -> float | None:
        if self.usd_per_gpu_hour is None:
            return None
        return gpu_seconds / 3600.0 * self.usd_per_gpu_hour

    def describe(self) -> str:
        if self.usd_per_gpu_hour is None:
            return f"`{self.price_id}`: GPU wall time, no $/GPU-hour rate set (GPU-hours only)"
        text = f"`{self.price_id}`: GPU wall time at ${self.usd_per_gpu_hour:.4g} per GPU-hour"
        return f"{text} ({self.basis})" if self.basis else text


Price = TokenPrice | GpuTimePrice

JEV_1_13_0 = TokenPrice(
    price_id="jev-1.13.0",
    input_usd_per_mtok=USD_PER_INPUT_TOKEN / _PER_MTOK,
    output_usd_per_mtok=0.0,
    source="https://docs.typesafe.ai/models",
    as_of="2026-09-25",
)
GPT_6_LUNA = TokenPrice(
    price_id="gpt-6-luna",
    input_usd_per_mtok=0.10,
    cached_input_usd_per_mtok=0.01,
    cache_write_usd_per_mtok=0.125,
    output_usd_per_mtok=0.50,
    long_context_threshold=272_000,
    long_context_input_multiplier=2.0,
    long_context_output_multiplier=1.5,
    source="https://developers.openai.com/api/docs/models/gpt-6-luna",
    as_of="2026-09-25",
)
"""Standard processing."""
GPT_6_LUNA_BATCH = dataclasses.replace(GPT_6_LUNA, price_id="gpt-6-luna:batch", rate_multiplier=0.5)
GPT_6_LUNA_FLEX = dataclasses.replace(GPT_6_LUNA, price_id="gpt-6-luna:flex", rate_multiplier=0.5)

TOKEN_PRICES: Mapping[str, TokenPrice] = {
    price.price_id: price for price in (JEV_1_13_0, GPT_6_LUNA, GPT_6_LUNA_BATCH, GPT_6_LUNA_FLEX)
}


def _float_env(name: str) -> float | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {value!r}") from None


def gpu_usd_per_hour_from_env() -> float | None:
    """:data:`GPU_USD_PER_HOUR_ENV` as a float, ``None`` when unset or empty."""
    return _float_env(GPU_USD_PER_HOUR_ENV)


@dataclass(frozen=True)
class PriceBook:
    """The prices :func:`jevemu.bench.compare.stats` applies (module docstring)."""

    token_prices: Mapping[str, TokenPrice] = field(default_factory=lambda: dict(TOKEN_PRICES))
    gpu: GpuTimePrice = field(default_factory=lambda: GpuTimePrice(None))

    @classmethod
    def default(
        cls,
        gpu_usd_per_hour: float | None = None,
        *,
        gpu_watts: float | None = None,
        usd_per_wh: float | None = None,
    ) -> PriceBook:
        """The built-in token prices and a GPU rate: an explicit $/GPU-hour (argument, then
        :data:`GPU_USD_PER_HOUR_ENV`) wins; else the electricity estimate from ``gpu_watts``
        (else :data:`GPU_WATTS_ENV`, else :data:`DEFAULT_GPU_WATTS`) and ``usd_per_wh`` (else
        :data:`USD_PER_WH_ENV`, else :data:`DEFAULT_USD_PER_WH`)."""
        rate = gpu_usd_per_hour if gpu_usd_per_hour is not None else gpu_usd_per_hour_from_env()
        if rate is not None:
            return cls(gpu=GpuTimePrice(rate, basis="explicit rate"))
        watts = gpu_watts if gpu_watts is not None else _float_env(GPU_WATTS_ENV)
        price = usd_per_wh if usd_per_wh is not None else _float_env(USD_PER_WH_ENV)
        return cls(
            gpu=GpuTimePrice.from_power(
                DEFAULT_GPU_WATTS if watts is None else watts,
                DEFAULT_USD_PER_WH if price is None else price,
            )
        )

    def resolve(self, system: Mapping[str, Any]) -> Price | None:
        """The price of the system a manifest describes; ``None`` if the book has none."""
        price_id = system.get("price_id")
        if isinstance(price_id, str):
            price = self.token_prices.get(price_id)
            recorded = system.get("token_price")
            if price is None and isinstance(recorded, Mapping):
                price = TokenPrice(**recorded)
            return price
        if system.get("kind") in GPU_TIME_KINDS:
            return self.gpu
        model = system.get("model")
        return self.token_prices.get(model) if isinstance(model, str) else None


@dataclass(frozen=True)
class ItemCost:
    item_id: str
    usage: TokenUsage | None
    cost_usd: float | None
    """The item's cost (module docstring); ``None`` if unpriced or no $/GPU-hour rate is set."""
    gpu_seconds: float | None = None
    """Time-priced: the item's share of its file's GPU wall time."""
    recorded_usd: float | None = None
    """The system's own accounting (Jev: 0 for a response-cache hit)."""
    table_usd: float | None = None
    """Token-priced: the price table applied to :attr:`usage`."""
    billed: bool = False
    """The system recorded a charge for a call it made (not a response-cache hit)."""
    price_mismatch: bool = False
    """Billed, and the recorded charge differs from :attr:`table_usd` by more than
    :data:`MISMATCH_TOLERANCE`."""


def _extra_count(extra: Mapping[str, Any], name: str) -> int | None:
    value = extra.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"usage.{name} must be a non-negative integer, got {value!r}")
    return value


def record_usage(record: ItemRecord) -> TokenUsage | None:
    """The tokens of ``record`` (module docstring); ``None`` without usage."""
    if record.usage is None:
        return None
    extra = record.usage.model_extra or {}
    cached = _extra_count(extra, CACHED_INPUT_FIELD)
    if cached is None and record.diagnostics is not None:
        cached = record.diagnostics.cached_tokens
    return TokenUsage(
        input_tokens=record.usage.input_tokens,
        output_tokens=record.usage.output_tokens,
        cached_input_tokens=cached,
        cache_write_tokens=_extra_count(extra, CACHE_WRITE_FIELD) or 0,
    )


def invocation_seconds(invocations: Sequence[Invocation]) -> float | None:
    """Summed wall time of the finished invocations; ``None`` if none finished."""
    spans = [
        (i.finished_at - i.started_at).total_seconds()
        for i in invocations
        if i.finished_at is not None
    ]
    return math.fsum(spans) if spans else None


def allocate(total: float, weights: Sequence[float | None]) -> list[float]:
    """Split ``total`` in proportion to ``weights``; a ``None`` weight is the mean of the
    others, and all-zero (or all-``None``) weights split it equally."""
    known = [w for w in weights if w is not None]
    fill = fmean(known) if known else 1.0
    filled = [fill if w is None else w for w in weights]
    norm = math.fsum(filled)
    if norm <= 0.0:
        filled, norm = [1.0] * len(weights), float(len(weights))
    return [total * w / norm for w in filled]


def _close(recorded: float, table: float) -> bool:
    return abs(recorded - table) <= MISMATCH_TOLERANCE * abs(table)


def item_costs(
    records: Sequence[ItemRecord],
    price: Price | None,
    *,
    invocations: Sequence[Invocation] = (),
) -> list[ItemCost]:
    """The cost of each of ``records`` (one records file) under ``price`` (module docstring).

    ``invocations`` (the manifest's, for this file) are needed for a :class:`GpuTimePrice`;
    without them the GPU time, and so the cost, is ``None``.
    """
    usages = [record_usage(record) for record in records]
    if isinstance(price, GpuTimePrice):
        total = invocation_seconds(invocations)
        weights = [None if u is None else float(u.input_tokens + u.output_tokens) for u in usages]
        shares: Sequence[float | None] = (
            [None] * len(records) if total is None else allocate(total, weights)
        )
        return [
            ItemCost(
                item_id=record.item_id,
                usage=usage,
                cost_usd=None if share is None else price.cost(share),
                gpu_seconds=share,
                recorded_usd=record.cost_usd,
            )
            for record, usage, share in zip(records, usages, shares, strict=True)
        ]
    costs = []
    for record, usage in zip(records, usages, strict=True):
        table = None if price is None or usage is None else price.cost(usage)
        recorded = record.cost_usd
        billed = recorded is not None and not record.cached
        if recorded is not None and billed:
            cost: float | None = recorded
        elif table is not None:
            cost = table
        else:
            cost = 0.0 if price is not None else None
        costs.append(
            ItemCost(
                item_id=record.item_id,
                usage=usage,
                cost_usd=cost,
                recorded_usd=recorded,
                table_usd=table,
                billed=billed,
                price_mismatch=(
                    billed
                    and recorded is not None
                    and table is not None
                    and not _close(recorded, table)
                ),
            )
        )
    return costs
