#!/usr/bin/env bash
# Start the pinned vLLM server from docker/vllm/compose.yaml and wait until /health is up.
#
#   scripts/serve_vllm.sh [--preset NAME]        # or JEVEMU_VLLM_PRESET=NAME
#   eval "$(scripts/serve_vllm.sh --preset qwen3.8-27b-awq | grep '^export ')"
#
# A preset is docker/vllm/presets/NAME.env (bash syntax). It sets the model configuration and
# wins over the environment: JEVEMU_VLLM_MODEL, JEVEMU_VLLM_MODEL_REVISION,
# JEVEMU_VLLM_MAX_MODEL_LEN, JEVEMU_VLLM_GPU_MEMORY_UTILIZATION,
# JEVEMU_VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS (JSON, no spaces) and JEVEMU_VLLM_EXTRA_ARGS
# (appended to `vllm serve`). Without a preset these come from the environment, defaulting to
# Qwen/Qwen3-0.6B as in the compose file. Always from the environment (all optional):
# JEVEMU_VLLM_PORT (8000), JEVEMU_VLLM_MAX_LOGPROBS (576), JEVEMU_VLLM_LOGPROBS_MODE
# (raw_logprobs), JEVEMU_VLLM_DTYPE (auto), JEVEMU_VLLM_WAIT_SECONDS (900).
# Prints the exports jevemu clients read (URL, image digest, model, resolved commit SHA, and the
# declared flags the capability probe keys on).
# Stop with: docker compose -f docker/vllm/compose.yaml down
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
presets="$root/docker/vllm/presets"
preset="${JEVEMU_VLLM_PRESET:-}"
while (($#)); do
  case "$1" in
    --preset) preset="${2:?--preset needs a name}"; shift 2 ;;
    --preset=*) preset="${1#--preset=}"; shift ;;
    *) echo "usage: $0 [--preset NAME]" >&2; exit 2 ;;
  esac
done
if [[ -n "$preset" ]]; then
  if [[ ! -f "$presets/$preset.env" ]]; then
    echo "unknown preset '$preset'; available: $(cd "$presets" && ls -- *.env | sed 's/\.env$//' | tr '\n' ' ')" >&2
    exit 2
  fi
  set -a
  # shellcheck source=/dev/null
  source "$presets/$preset.env"
  set +a
fi

export JEVEMU_VLLM_MODEL="${JEVEMU_VLLM_MODEL:-Qwen/Qwen3-0.6B}"
export JEVEMU_VLLM_MODEL_REVISION="${JEVEMU_VLLM_MODEL_REVISION:-main}"
export JEVEMU_VLLM_MAX_LOGPROBS="${JEVEMU_VLLM_MAX_LOGPROBS:-576}"
export JEVEMU_VLLM_LOGPROBS_MODE="${JEVEMU_VLLM_LOGPROBS_MODE:-raw_logprobs}"
export JEVEMU_VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS="${JEVEMU_VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
compose=(docker compose -f "$root/docker/vllm/compose.yaml")
url="http://localhost:${JEVEMU_VLLM_PORT:-8000}"
wait_seconds="${JEVEMU_VLLM_WAIT_SECONDS:-900}"
model="$JEVEMU_VLLM_MODEL"
revision="$JEVEMU_VLLM_MODEL_REVISION"

started=$SECONDS
"${compose[@]}" up -d

deadline=$((SECONDS + wait_seconds))
until curl -fsS "$url/health" >/dev/null 2>&1; do
  if [[ -n "$("${compose[@]}" ps --all --status exited --quiet vllm)" ]] || ((SECONDS >= deadline)); then
    echo "vLLM did not become healthy (waited $((SECONDS - started))s); recent logs:" >&2
    "${compose[@]}" logs --tail=60 vllm >&2
    exit 1
  fi
  sleep 3
done

# vLLM does not report the model commit over HTTP; resolve a branch/tag from the shared HF cache.
ref="${HOME}/.cache/huggingface/hub/models--${model//\//--}/refs/${revision}"
if [[ ! "$revision" =~ ^[0-9a-f]{40}$ && -f "$ref" ]]; then
  revision="$(<"$ref")"
fi

image="$("${compose[@]}" config --images)"
echo "vLLM healthy at $url after $((SECONDS - started))s${preset:+ (preset $preset)}: $(curl -fsS "$url/version")"
printf 'export'
for pair in \
  "JEVEMU_VLLM_URL=$url" \
  "JEVEMU_VLLM_IMAGE=$image" \
  "JEVEMU_VLLM_MODEL=$model" \
  "JEVEMU_VLLM_MODEL_REVISION=$revision" \
  "JEVEMU_VLLM_MAX_LOGPROBS=$JEVEMU_VLLM_MAX_LOGPROBS" \
  "JEVEMU_VLLM_LOGPROBS_MODE=$JEVEMU_VLLM_LOGPROBS_MODE" \
  "JEVEMU_VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS=$JEVEMU_VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS"; do
  printf ' %s=%q' "${pair%%=*}" "${pair#*=}"
done
printf '\n'
