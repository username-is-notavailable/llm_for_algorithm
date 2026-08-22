#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-${PROJECT_ROOT}/.third_party/verl}"
VERL_PYTHON="${VERL_ROOT}/.venv/bin/python"
MODE="${1:-smoke}"
GPU_COUNT="${SFT_GPU_COUNT:-1}"

case "${MODE}" in
  local-smoke) CONFIG="configs/training/m6_sft_local_smoke.yaml" ;;
  smoke) CONFIG="configs/training/m6_sft_smoke.yaml" ;;
  throughput) CONFIG="configs/training/m6_sft_throughput.yaml" ;;
  sft1k) CONFIG="configs/training/m7_sft_1k.yaml" ;;
  sft1k-short-pilot) CONFIG="configs/training/m7_sft_1k_short_pilot.yaml" ;;
  *)
    echo "Usage: SFT_GPU_COUNT=N bash scripts/cloud_train_sft.sh {local-smoke|smoke|throughput|sft1k|sft1k-short-pilot} [--resume CHECKPOINT]" >&2
    exit 2
    ;;
esac
shift || true

if [[ ! "${GPU_COUNT}" =~ ^[1-4]$ ]]; then
  echo "SFT_GPU_COUNT must be 1, 2, 3, or 4." >&2
  exit 2
fi
if [[ ! -x "${VERL_PYTHON}" ]]; then
  echo "Missing verl environment. Run bash scripts/cloud_setup.sh first." >&2
  exit 1
fi
if [[ "${MODE}" == "sft1k-short-pilot" ]]; then
  REQUIRED_DATA="data/processed/sft_1k_short_v1.jsonl"
else
  REQUIRED_DATA="data/processed/sft_1k.jsonl"
fi
if [[ ! -f "${PROJECT_ROOT}/${REQUIRED_DATA}" ]]; then
  echo "Missing training data: ${REQUIRED_DATA}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/cloud_cache.sh"
export VERL_ROOT
export TOKENIZERS_PARALLELISM=false
# Long, variable-length SFT samples can leave large inactive CUDA allocator
# segments between steps. Expandable segments let PyTorch reuse that reserved
# memory instead of failing a later long-sequence backward allocation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SFT_RUN_TIMESTAMP="${SFT_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"

if ((GPU_COUNT == 1)); then
  exec "${VERL_PYTHON}" scripts/train_sft.py --config "${CONFIG}" "$@"
fi
exec "${VERL_PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${GPU_COUNT}" \
  scripts/train_sft.py --config "${CONFIG}" "$@"
