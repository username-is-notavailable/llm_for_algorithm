#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-${PROJECT_ROOT}/.third_party/verl}"
VERL_PYTHON="${VERL_ROOT}/.venv/bin/python"
MODE="${1:-smoke}"
MODEL_PATH="${2:-}"
case "${MODE}" in
  smoke) CONFIG="configs/eval/livecodebench_m7_sft1k_smoke_v1.yaml" ;;
  full) CONFIG="configs/eval/livecodebench_m7_sft1k_eval_v1.yaml" ;;
  sharded-full) CONFIG="configs/eval/livecodebench_m7_sft1k_eval_v1.yaml" ;;
  *)
    echo "Usage: bash scripts/cloud_eval_m7.sh {smoke|full|sharded-full} OUTPUTS_TRAINING_RUN/final [--resume OUTPUT_DIR]" >&2
    exit 2
    ;;
esac
shift || true
shift || true

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Usage: bash scripts/cloud_eval_m7.sh {smoke|full|sharded-full} OUTPUTS_TRAINING_RUN/final [--resume OUTPUT_DIR]" >&2
  exit 2
fi
if [[ ! -x "${VERL_PYTHON}" ]]; then
  echo "Missing verl environment. Run bash scripts/cloud_setup.sh first." >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing SFT checkpoint: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/data/processed/livecodebench_v1/eval_v1.jsonl" ]]; then
  echo "Missing fixed evaluation data. Run the M3 preparation first." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/cloud_cache.sh"
export VERL_ROOT
if [[ "${MODE}" == "sharded-full" ]]; then
  if (($#)); then
    echo "sharded-full does not accept --resume; resume each shard directly if needed." >&2
    exit 2
  fi
  export EVAL_RUN_TIMESTAMP="${EVAL_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
  BASE_NAME="m7-sft1k-eval-v1"
  SHARD_0="outputs/eval/${BASE_NAME}-shard-01-of-02-${EVAL_RUN_TIMESTAMP}"
  SHARD_1="outputs/eval/${BASE_NAME}-shard-02-of-02-${EVAL_RUN_TIMESTAMP}"
  CUDA_VISIBLE_DEVICES=0 "${VERL_PYTHON}" -m src.eval.evaluator \
    --config "${CONFIG}" --model-path "${MODEL_PATH}" --shard-index 0 --num-shards 2 \
    2>&1 | tee /tmp/qwen3-m7-eval-shard-0.log &
  PID_0=$!
  CUDA_VISIBLE_DEVICES=1 "${VERL_PYTHON}" -m src.eval.evaluator \
    --config "${CONFIG}" --model-path "${MODEL_PATH}" --shard-index 1 --num-shards 2 \
    2>&1 | tee /tmp/qwen3-m7-eval-shard-1.log &
  PID_1=$!
  STATUS=0
  wait "${PID_0}" || STATUS=$?
  wait "${PID_1}" || STATUS=$?
  if ((STATUS != 0)); then
    echo "At least one evaluation shard failed; shard artifacts were retained." >&2
    exit "${STATUS}"
  fi
  exec "${VERL_PYTHON}" -m src.eval.merge_shards \
    --config "${CONFIG}" --model-path "${MODEL_PATH}" \
    --output-dir "outputs/eval/${BASE_NAME}-${EVAL_RUN_TIMESTAMP}" \
    "${SHARD_0}" "${SHARD_1}"
fi
exec "${VERL_PYTHON}" -m src.eval.evaluator \
  --config "${CONFIG}" \
  --model-path "${MODEL_PATH}" \
  "$@"
