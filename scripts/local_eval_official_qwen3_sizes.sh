#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-${PROJECT_ROOT}/.third_party/verl}"
VERL_PYTHON="${VERL_ROOT}/.venv/bin/python"
SIZE="${1:-}"

case "${SIZE}" in
  1.7b) CONFIG="configs/eval/qwen3_official_1_7b_local_smoke.yaml" ;;
  4b) CONFIG="configs/eval/qwen3_official_4b_local_smoke.yaml" ;;
  *)
    echo "Usage: bash scripts/local_eval_official_qwen3_sizes.sh {1.7b|4b}" >&2
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
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
if [[ ! -f data/processed/livecodebench_v1/smoke_10.jsonl ]]; then
  "${VERL_PYTHON}" scripts/prepare_eval.py
fi
exec "${VERL_PYTHON}" -m src.eval.evaluator --config "${CONFIG}"
