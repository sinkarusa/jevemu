#!/usr/bin/env bash
# Local quality gate (no hosted CI): lint, format, types, offline tests.
# Live-server tiers are opt-in: pytest -m gpu / -m cpu / -m live_jev.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest -q -m "not cpu and not gpu and not live_jev" "$@"
