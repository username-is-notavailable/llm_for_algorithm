#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-${PROJECT_ROOT}/.third_party/verl}"
VERL_PYTHON="${VERL_ROOT}/.venv/bin/python"
MODEL_SIZE="${1:-}"
SPLIT="${2:-}"
MODE="${3:-}"

if [[ ! "${MODEL_SIZE}" =~ ^(1\.7b|1\.7b-post|4b|4b-post)$ ]] || [[ ! "${SPLIT}" =~ ^(smoke|dev)$ ]] || \
   [[ ! "${MODE}" =~ ^(oneshot|agent|both|oneshot-sharded|agent-sharded)$ ]]; then
  echo "Usage: bash scripts/cloud_eval_agent.sh {1.7b|1.7b-post|4b|4b-post} {smoke|dev} {oneshot|agent|both|oneshot-sharded|agent-sharded} [--model-path CHECKPOINT|--resume OUTPUT_DIR]" >&2
  exit 2
fi
shift 3
if [[ ! -x "${VERL_PYTHON}" ]]; then
  echo "Missing verl environment. Run bash scripts/cloud_setup.sh first." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/cloud_cache.sh"
export VERL_ROOT
if [[ ! -f data/processed/agent_eval_v1/dev_60.jsonl ]]; then
  "${VERL_PYTHON}" scripts/prepare_agent_eval.py
fi

render_config() {
  local kind="$1"
  local input="configs/eval/${kind}_qwen3_1_7b_base_${SPLIT}_v1.yaml"
  local output="/tmp/qwen3-m9-${kind}-${MODEL_SIZE}-${SPLIT}.yaml"
  local -a overrides=()
  if [[ -n "${M9_MAX_NEW_TOKENS:-}" ]]; then
    overrides+=(--max-new-tokens "${M9_MAX_NEW_TOKENS}")
  fi
  if [[ "${kind}" == "agent" && -n "${M9_MAX_TOTAL_GENERATION_TOKENS:-}" ]]; then
    overrides+=(--max-total-generation-tokens "${M9_MAX_TOTAL_GENERATION_TOKENS}")
  fi
  "${VERL_PYTHON}" scripts/render_m9_eval_config.py \
    --input "${input}" --model-size "${MODEL_SIZE}" --output "${output}" \
    "${overrides[@]}" >/dev/null
  echo "${output}"
}

run_oneshot() {
  local config
  config="$(render_config oneshot)"
  "${VERL_PYTHON}" -m src.eval.evaluator --config "${config}" "$@"
}

run_agent() {
  local config
  config="$(render_config agent)"
  "${VERL_PYTHON}" -m src.agent.evaluator --config "${config}" "$@"
}

run_sharded() {
  local kind="$1"
  shift
  local gpu_count="${EVAL_GPU_COUNT:-4}"
  if [[ ! "${gpu_count}" =~ ^[1-4]$ ]]; then
    echo "EVAL_GPU_COUNT must be 1, 2, 3, or 4." >&2
    exit 2
  fi
  local -a model_args=()
  if (($#)); then
    if (($# != 2)) || [[ "$1" != "--model-path" ]] || [[ ! -d "$2" ]]; then
      echo "Sharded modes accept only --model-path CHECKPOINT; resume individual shards directly." >&2
      exit 2
    fi
    model_args=(--model-path "$2")
  fi
  local config timestamp base status index pid shard
  local -a pids=() shards=()
  config="$(render_config "${kind}")"
  timestamp="${AGENT_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
  export EVAL_RUN_TIMESTAMP="${timestamp}"
  export AGENT_RUN_TIMESTAMP="${timestamp}"
  case "${kind}:${MODEL_SIZE}" in
    oneshot:1.7b) base="m9-oneshot-qwen3-1.7b-base-${SPLIT}-v1" ;;
    oneshot:1.7b-post) base="m9-oneshot-qwen3-1.7b-posttrained-${SPLIT}-v1" ;;
    oneshot:4b) base="m9-oneshot-qwen3-4b-base-${SPLIT}-v1" ;;
    oneshot:4b-post) base="m9-oneshot-qwen3-4b-posttrained-${SPLIT}-v1" ;;
    agent:1.7b) base="m9-agent-qwen3-1.7b-base-${SPLIT}-v1" ;;
    agent:1.7b-post) base="m9-agent-qwen3-1.7b-posttrained-${SPLIT}-v1" ;;
    agent:4b) base="m9-agent-qwen3-4b-base-${SPLIT}-v1" ;;
    agent:4b-post) base="m9-agent-qwen3-4b-posttrained-${SPLIT}-v1" ;;
  esac
  local output_root module merge_module
  if [[ "${kind}" == "agent" ]]; then
    output_root="outputs/agent_eval"
    module="src.agent.evaluator"
    merge_module="src.agent.merge_shards"
  else
    output_root="outputs/eval"
    module="src.eval.evaluator"
    merge_module="src.eval.merge_shards"
  fi
  for ((index = 0; index < gpu_count; index++)); do
    printf -v shard '%s/%s-shard-%02d-of-%02d-%s' \
      "${output_root}" "${base}" "$((index + 1))" "${gpu_count}" "${timestamp}"
    shards+=("${shard}")
    CUDA_VISIBLE_DEVICES="${index}" "${VERL_PYTHON}" -m "${module}" \
      --config "${config}" --shard-index "${index}" --num-shards "${gpu_count}" \
      "${model_args[@]}" \
      2>&1 | tee "/tmp/qwen3-m9-${kind}-${MODEL_SIZE}-${SPLIT}-shard-${index}.log" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=$?
  done
  if ((status != 0)); then
    echo "At least one ${kind} shard failed; shard artifacts were retained." >&2
    exit "${status}"
  fi
  "${VERL_PYTHON}" -m "${merge_module}" \
    --config "${config}" \
    --output-dir "${output_root}/${base}-${timestamp}" \
    "${model_args[@]}" \
    "${shards[@]}"
}

case "${MODE}" in
  oneshot) run_oneshot "$@" ;;
  agent) run_agent "$@" ;;
  both)
    if (($#)); then
      echo "both mode does not accept resume arguments." >&2
      exit 2
    fi
    run_oneshot
    run_agent
    ;;
  oneshot-sharded) run_sharded oneshot "$@" ;;
  agent-sharded) run_sharded agent "$@" ;;
esac
