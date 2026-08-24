#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-${PROJECT_ROOT}/.third_party/verl}"
VERL_PYTHON="${VERL_ROOT}/.venv/bin/python"
MODEL_SIZE="${1:-}"
SPLIT="${2:-}"
MODE="${3:-}"

if [[ ! "${MODEL_SIZE}" =~ ^(1\.7b|4b)$ ]] || [[ ! "${SPLIT}" =~ ^(smoke|dev)$ ]] || \
   [[ ! "${MODE}" =~ ^(oneshot|agent|both|agent-sharded)$ ]]; then
  echo "Usage: bash scripts/cloud_eval_agent.sh {1.7b|4b} {smoke|dev} {oneshot|agent|both|agent-sharded} [--resume OUTPUT_DIR]" >&2
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
  "${VERL_PYTHON}" scripts/render_m9_eval_config.py \
    --input "${input}" --model-size "${MODEL_SIZE}" --output "${output}" >/dev/null
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

run_agent_sharded() {
  if (($#)); then
    echo "agent-sharded does not accept --resume; resume individual shard directories directly." >&2
    exit 2
  fi
  local config timestamp base shard0 shard1 status pid0 pid1
  config="$(render_config agent)"
  timestamp="${AGENT_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
  export AGENT_RUN_TIMESTAMP="${timestamp}"
  base="m9-agent-qwen3-${MODEL_SIZE}-base-${SPLIT}-v1"
  shard0="outputs/agent_eval/${base}-shard-01-of-02-${timestamp}"
  shard1="outputs/agent_eval/${base}-shard-02-of-02-${timestamp}"
  CUDA_VISIBLE_DEVICES=0 "${VERL_PYTHON}" -m src.agent.evaluator \
    --config "${config}" --shard-index 0 --num-shards 2 \
    2>&1 | tee "/tmp/qwen3-m9-agent-${MODEL_SIZE}-${SPLIT}-shard-0.log" &
  pid0=$!
  CUDA_VISIBLE_DEVICES=1 "${VERL_PYTHON}" -m src.agent.evaluator \
    --config "${config}" --shard-index 1 --num-shards 2 \
    2>&1 | tee "/tmp/qwen3-m9-agent-${MODEL_SIZE}-${SPLIT}-shard-1.log" &
  pid1=$!
  status=0
  wait "${pid0}" || status=$?
  wait "${pid1}" || status=$?
  if ((status != 0)); then
    echo "At least one Agent shard failed; shard artifacts were retained." >&2
    exit "${status}"
  fi
  "${VERL_PYTHON}" -m src.agent.merge_shards \
    --config "${config}" \
    --output-dir "outputs/agent_eval/${base}-${timestamp}" \
    "${shard0}" "${shard1}"
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
  agent-sharded) run_agent_sharded "$@" ;;
esac
