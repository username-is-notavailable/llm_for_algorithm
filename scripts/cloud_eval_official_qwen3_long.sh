#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-${PROJECT_ROOT}/.third_party/verl}"
VERL_PYTHON="${VERL_ROOT}/.venv/bin/python"
SIZE="${1:-}"

case "${SIZE}" in
  4b) CONFIG="configs/eval/qwen3_official_4b_16k_smoke.yaml" ;;
  8b) CONFIG="configs/eval/qwen3_official_8b_16k_smoke.yaml" ;;
  *)
    echo "Usage: bash scripts/cloud_eval_official_qwen3_long.sh {4b|8b}" >&2
    exit 2
    ;;
esac

if [[ ! -x "${VERL_PYTHON}" ]]; then
  echo "Missing verl environment. Run bash scripts/cloud_setup.sh first." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/cloud_cache.sh"
export VERL_ROOT
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
if [[ ! -f data/processed/livecodebench_v1/smoke_10.jsonl ]]; then
  echo "Missing frozen smoke data: data/processed/livecodebench_v1/smoke_10.jsonl" >&2
  exit 1
fi
exec "${VERL_PYTHON}" -m src.eval.evaluator --config "${CONFIG}"
