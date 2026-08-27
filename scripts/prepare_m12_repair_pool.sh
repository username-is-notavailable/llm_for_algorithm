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

RAW_DIR="data/processed/codecontests_plus_repair_1000_v1_raw"
COMPACT_DIR="data/processed/codecontests_plus_repair_1000_v2"

echo "[1/2] Selecting 1000 new checker-backed problems (excluding the frozen 300)"
"${VERL_PYTHON}" scripts/prepare_codecontests_plus_repair.py \
  --config configs/data/m12_codecontests_plus_repair_1000_v1.yaml

echo "[2/2] Compacting tests/checkers and building the byte-offset problem index"
"${VERL_PYTHON}" scripts/compact_codecontests_plus_repair.py \
  --problems "${RAW_DIR}/problems_1000.jsonl" \
  --failure-pool "${RAW_DIR}/failure_pool_1000.jsonl" \
  --one-shot-seeds "${RAW_DIR}/one_shot_seeds_1000.jsonl" \
  --output-dir "${COMPACT_DIR}" \
  --manifest data/splits/codecontests_plus_repair_1000_v2_manifest.json

echo "Compact pool ready. The raw directory can be deleted after checking the manifest:"
echo "  ${RAW_DIR}"
echo "Next: export DASHSCOPE_API_KEY=..."
echo "  bash scripts/cloud_generate_repair_api.sh configs/data/m12_repair_api_codecontests_plus_1000_8b.yaml"
