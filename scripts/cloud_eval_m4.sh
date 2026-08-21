#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-${PROJECT_ROOT}/.third_party/verl}"
VERL_PYTHON="${VERL_ROOT}/.venv/bin/python"
MODE="${1:-smoke}"

if [[ ! -x "${VERL_PYTHON}" ]]; then
  echo "Missing verl environment. Run bash scripts/cloud_setup.sh first." >&2
  exit 1
fi

case "${MODE}" in
  local-smoke)
    CONFIG="configs/eval/livecodebench_m4_local_smoke_v1.yaml"
    DATA="data/processed/livecodebench_v1/smoke_10.jsonl"
    export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
    export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
    ;;
  smoke)
    CONFIG="configs/eval/livecodebench_m4_smoke_v1.yaml"
    DATA="data/processed/livecodebench_v1/smoke_10.jsonl"
    ;;
  full)
    CONFIG="configs/eval/livecodebench_m4_eval_v1.yaml"
    DATA="data/processed/livecodebench_v1/eval_v1.jsonl"
    ;;
  *)
    echo "Usage: bash scripts/cloud_eval_m4.sh {local-smoke|smoke|full} [--resume OUTPUT_DIR]" >&2
    exit 2
    ;;
esac
shift || true

if [[ ! -f "${PROJECT_ROOT}/${DATA}" ]]; then
  echo "Missing fixed evaluation data: ${DATA}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/cloud_cache.sh"
export VERL_ROOT
exec "${VERL_PYTHON}" -m src.eval.evaluator --config "${CONFIG}" "$@"
