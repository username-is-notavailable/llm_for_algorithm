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
  sft1k-compact-pilot) CONFIG="configs/training/m7_sft_1k_compact_pilot.yaml" ;;
  compact-4epoch) CONFIG="configs/training/m7_sft_compact_4epoch.yaml" ;;
  ab-short-pilot) CONFIG="configs/training/m7_sft_ab_short_pilot.yaml" ;;
  ab-code-pilot) CONFIG="configs/training/m7_sft_ab_code_pilot.yaml" ;;
  ab-short-1k) CONFIG="configs/training/m7_sft_ab_short_1k.yaml" ;;
  ab-code-1k) CONFIG="configs/training/m7_sft_ab_code_1k.yaml" ;;
  ab-short-1k-4epoch) CONFIG="configs/training/m7_sft_ab_short_1k_4epoch.yaml" ;;
  weighted-smoke) CONFIG="configs/training/m7_sft_weighted_smoke.yaml" ;;
  weighted-short-1k-4epoch) CONFIG="configs/training/m7_sft_weighted_short_1k_4epoch.yaml" ;;
  agent-smoke) CONFIG="configs/training/m11_agent_sft_42_smoke.yaml" ;;
  agent-4b-smoke) CONFIG="configs/training/m11_agent_sft_4b_smoke.yaml" ;;
  agent-4b-1epoch) CONFIG="configs/training/m11_agent_sft_4b_1epoch.yaml" ;;
  agent-4b-post-smoke) CONFIG="configs/training/m11_agent_sft_4b_post_smoke.yaml" ;;
  agent-4b-post-1epoch) CONFIG="configs/training/m11_agent_sft_4b_post_1epoch.yaml" ;;
  *)
    echo "Usage: SFT_GPU_COUNT=N bash scripts/cloud_train_sft.sh {local-smoke|smoke|throughput|sft1k|sft1k-short-pilot|sft1k-compact-pilot|compact-4epoch|ab-short-pilot|ab-code-pilot|ab-short-1k|ab-code-1k|ab-short-1k-4epoch|weighted-smoke|weighted-short-1k-4epoch|agent-smoke|agent-4b-smoke|agent-4b-1epoch|agent-4b-post-smoke|agent-4b-post-1epoch} [--resume CHECKPOINT]" >&2
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
if [[ "${MODE}" == "agent-smoke" ]]; then
  REQUIRED_DATA="data/processed/agent_sft_v3/train.jsonl"
elif [[ "${MODE}" == "agent-4b-smoke" || "${MODE}" == "agent-4b-1epoch" ]]; then
  REQUIRED_DATA="data/processed/agent_sft_v1/train.jsonl"
elif [[ "${MODE}" == "agent-4b-post-smoke" || "${MODE}" == "agent-4b-post-1epoch" ]]; then
  REQUIRED_DATA="data/processed/agent_sft_v2/train.jsonl"
elif [[ "${MODE}" == "ab-short-pilot" || "${MODE}" == "ab-short-1k" || "${MODE}" == "ab-short-1k-4epoch" || "${MODE}" == "weighted-smoke" || "${MODE}" == "weighted-short-1k-4epoch" ]]; then
  REQUIRED_DATA="data/processed/sft_1k_short_reasoning_v2.jsonl"
elif [[ "${MODE}" == "ab-code-pilot" || "${MODE}" == "ab-code-1k" ]]; then
  REQUIRED_DATA="data/processed/sft_1k_code_only_v2.jsonl"
elif [[ "${MODE}" == "sft1k-short-pilot" ]]; then
  REQUIRED_DATA="data/processed/sft_1k_short_v1.jsonl"
elif [[ "${MODE}" == "sft1k-compact-pilot" || "${MODE}" == "compact-4epoch" ]]; then
  REQUIRED_DATA="data/processed/sft_1k_compact_v1.jsonl"
else
  REQUIRED_DATA="data/processed/sft_1k.jsonl"
fi
if [[ ! -f "${PROJECT_ROOT}/${REQUIRED_DATA}" ]]; then
  echo "Missing training data: ${REQUIRED_DATA}" >&2
  exit 1
fi
if [[ "${MODE}" == "agent-smoke" && ! -f "${PROJECT_ROOT}/data/processed/agent_sft_v3/dev_8.jsonl" ]]; then
  echo "Missing evaluation data: data/processed/agent_sft_v3/dev_8.jsonl" >&2
  exit 1
fi
if [[ "${MODE}" == "agent-4b-1epoch" && ! -f "${PROJECT_ROOT}/data/processed/agent_sft_v1/dev.jsonl" ]]; then
  echo "Missing evaluation data: data/processed/agent_sft_v1/dev.jsonl" >&2
  exit 1
fi
if [[ "${MODE}" == "agent-4b-post-1epoch" && ! -f "${PROJECT_ROOT}/data/processed/agent_sft_v2/dev.jsonl" ]]; then
  echo "Missing evaluation data: data/processed/agent_sft_v2/dev.jsonl" >&2
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
