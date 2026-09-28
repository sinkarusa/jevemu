#!/usr/bin/env bash
# Start the pinned CLM-8B server from docker/clm/compose.yaml (Qwen3-8B encoder + clm-serve) and
# wait until clm-serve reports its encoder up.
#
#   eval "$(scripts/serve_clm.sh | grep '^export ')"
#
# Environment (all optional): JEVEMU_CLM_PORT (8700), JEVEMU_CLM_ENCODER_GPU_MEMORY_UTILIZATION
# (0.8), JEVEMU_CLM_WAIT_SECONDS (1800), HF_TOKEN, CLM_API_KEY (unset: no auth).
# Prints the export scripts/run_split.py --system clm reads (JEVEMU_CLM_URL).
# Stop with: docker compose -f docker/clm/compose.yaml down
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose=(docker compose -f "$root/docker/clm/compose.yaml")
url="http://localhost:${JEVEMU_CLM_PORT:-8700}"
wait_seconds="${JEVEMU_CLM_WAIT_SECONDS:-1800}"
auth=()
if [[ -n "${CLM_API_KEY:-}" ]]; then
  auth=(-H "Authorization: Bearer $CLM_API_KEY")
fi

started=$SECONDS
"${compose[@]}" up -d

deadline=$((SECONDS + wait_seconds))
until curl -fsS "$url/health" 2>/dev/null | grep -q '"embedder":true'; do
  if [[ -n "$("${compose[@]}" ps --all --status exited --quiet)" ]] || ((SECONDS >= deadline)); then
    echo "CLM did not become healthy (waited $((SECONDS - started))s); recent logs:" >&2
    "${compose[@]}" logs --tail=60 >&2
    exit 1
  fi
  sleep 5
done

echo "CLM healthy at $url after $((SECONDS - started))s: $(curl -fsS "${auth[@]}" "$url/v1/models")"
printf 'export JEVEMU_CLM_URL=%q\n' "$url"
