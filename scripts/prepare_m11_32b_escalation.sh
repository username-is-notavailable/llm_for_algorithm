#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:-}"
if [[ -z "${RUN}" ]]; then
  RUN="$(find "${PROJECT_ROOT}/outputs/data_generation" -mindepth 1 -maxdepth 1 -type d \
    -name 'm11-repair-api-codecontests-plus-300-8b-v1-*' | sort | tail -n 1)"
fi
if [[ -z "${RUN}" || ! -f "${RUN}/metrics.json" ]]; then
  echo "Usage: bash scripts/prepare_m11_32b_escalation.sh [8B_RUN_DIRECTORY]" >&2
  exit 1
fi
cd "${PROJECT_ROOT}"
VERL_PYTHON="${VERL_ROOT:-${PROJECT_ROOT}/.third_party/verl}/.venv/bin/python"

"${VERL_PYTHON}" scripts/prepare_m11_repair_escalation.py \
  --run "${RUN}" \
  --failure-pool data/processed/codecontests_plus_repair_300_v2/failure_pool.jsonl \
  --canonical-output data/processed/agent_sft_source_v1/repair_8b_canonical.jsonl \
  --escalation-output data/processed/agent_sft_source_v1/repair_escalation_32b.jsonl \
  --manifest data/splits/m11_repair_8b_escalation_v1_manifest.json

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "Escalation data is ready. Set DASHSCOPE_API_KEY, then rerun the API command below:" >&2
  echo "bash scripts/cloud_generate_repair_api.sh configs/data/m11_repair_api_codecontests_plus_escalation_32b.yaml" >&2
  exit 2
fi
bash scripts/cloud_generate_repair_api.sh \
  configs/data/m11_repair_api_codecontests_plus_escalation_32b.yaml
