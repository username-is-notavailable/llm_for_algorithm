#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-${PROJECT_ROOT}/.third_party/verl}"
VERL_PYTHON="${VERL_ROOT}/.venv/bin/python"
MODE="${1:-}"
MODEL_PATH="${2:-}"

case "${MODE}" in
  short) CONFIG="configs/eval/livecodebench_m7_ab_short_smoke_v2.yaml" ;;
  code) CONFIG="configs/eval/livecodebench_m7_ab_code_smoke_v2.yaml" ;;
  *)
    echo "Usage: bash scripts/cloud_eval_sft_ab.sh {short|code} OUTPUTS_TRAINING_RUN/{checkpoint-N|final}" >&2
    exit 2
    ;;
esac

if [[ -z "${MODEL_PATH}" || ! -d "${MODEL_PATH}" ]]; then
  echo "Missing SFT checkpoint: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -x "${VERL_PYTHON}" ]]; then
  echo "Missing verl environment. Run bash scripts/cloud_setup.sh first." >&2
  exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/data/processed/livecodebench_v1/smoke_10.jsonl" ]]; then
  echo "Missing fixed smoke data. Run the M3 preparation first." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/cloud_cache.sh"
export VERL_ROOT
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
exec "${VERL_PYTHON}" -m src.eval.evaluator \
  --config "${CONFIG}" \
  --model-path "${MODEL_PATH}"
