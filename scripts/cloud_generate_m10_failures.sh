#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-${PROJECT_ROOT}/.third_party/verl}"
VERL_PYTHON="${VERL_ROOT}/.venv/bin/python"
CONFIG="${M10_FAILURE_CONFIG:-configs/eval/m10_qwen3_4b_posttrained_failure_pilot_v1.yaml}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
NUM_SHARDS="${#GPUS[@]}"
RUN_TIMESTAMP="${M10_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
export EVAL_RUN_TIMESTAMP="${RUN_TIMESTAMP}"

if [[ ! -x "${VERL_PYTHON}" ]]; then
  echo "Missing verl environment. Run bash scripts/cloud_setup.sh first." >&2
  exit 1
fi
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/cloud_cache.sh"

pids=()
for index in "${!GPUS[@]}"; do
  gpu="${GPUS[$index]}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${VERL_PYTHON}" -m src.eval.evaluator \
    --config "${CONFIG}" --shard-index "${index}" --num-shards "${NUM_SHARDS}" \
    2>&1 | tee "/tmp/qwen3-m10-failure-shard-${index}.log" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if ((status != 0)); then
  echo "At least one failure-rollout shard failed; completed artifacts were retained." >&2
  exit "${status}"
fi
echo "All ${NUM_SHARDS} failure-rollout shards completed."

generation_files=()
for index in "${!GPUS[@]}"; do
  shard_number="$((index + 1))"
  printf -v shard_label '%02d' "${shard_number}"
  printf -v count_label '%02d' "${NUM_SHARDS}"
  generation_files+=(
    "outputs/failure_rollout/m10-qwen3-4b-posttrained-failure-pilot-v1-shard-${shard_label}-of-${count_label}-${RUN_TIMESTAMP}/generations.jsonl"
  )
done
"${VERL_PYTHON}" scripts/build_repair_failure_pool.py \
  --problems data/processed/repair_sft_v1/train_agent_pilot.jsonl \
  --generations "${generation_files[@]}" \
  --producer-model Qwen/Qwen3-4B \
  --exclude-manifest data/splits/agent_eval_v1_problem_ids.json \
  --output data/processed/repair_sft_v1/failure_pool_pilot.jsonl \
  --one-shot-output data/processed/repair_sft_v1/one_shot_candidates_pilot.jsonl
echo "M10 rollout timestamp: ${RUN_TIMESTAMP}"
