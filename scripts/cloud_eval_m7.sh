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
  *)
    echo "Usage: bash scripts/cloud_eval_m7.sh {smoke|full} OUTPUTS_TRAINING_RUN/final [--resume OUTPUT_DIR]" >&2
    exit 2
    ;;
esac
shift || true
shift || true

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Usage: bash scripts/cloud_eval_m7.sh {smoke|full} OUTPUTS_TRAINING_RUN/final [--resume OUTPUT_DIR]" >&2
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
exec "${VERL_PYTHON}" -m src.eval.evaluator \
  --config "${CONFIG}" \
  --model-path "${MODEL_PATH}" \
  "$@"
