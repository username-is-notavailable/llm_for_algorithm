#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-${PROJECT_ROOT}/.third_party/verl}"
VERL_PYTHON="${VERL_ROOT}/.venv/bin/python"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/cloud_cache.sh"

if [[ ! -x "${VERL_PYTHON}" ]]; then
  echo "Missing verl environment. Run bash scripts/cloud_setup.sh first." >&2
  exit 1
fi
if [[ ! -f data/processed/codecontests_plus_repair_300_v2/problems.index.json ]]; then
  echo "Missing the frozen 300-problem compact pool used for incremental exclusion." >&2
  exit 1
fi

CHECKPOINT="data/processed/codecontests_plus_repair_1000_v2/prepare.checkpoint.json"
RESUME_ARGS=()
if [[ -f "${CHECKPOINT}" ]]; then
  RESUME_ARGS+=(--resume)
fi

echo "Selecting and streaming 1000 new checker-backed problems (excluding the frozen 300)"
"${VERL_PYTHON}" scripts/prepare_codecontests_plus_repair_resumable.py \
  --config configs/data/m12_codecontests_plus_repair_1000_v1.yaml \
  "${RESUME_ARGS[@]}"

echo "Compact pool ready: data/processed/codecontests_plus_repair_1000_v2"
echo "Next: export DASHSCOPE_API_KEY=..."
echo "  bash scripts/cloud_generate_repair_api.sh configs/data/m12_repair_api_codecontests_plus_1000_8b.yaml"
