#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-${PROJECT_ROOT}/.third_party/verl}"
VERL_PYTHON="${VERL_ROOT}/.venv/bin/python"
STAGE="${1:-}"
RUN="${2:-}"
cd "${PROJECT_ROOT}"

if [[ ! -x "${VERL_PYTHON}" ]]; then
  echo "Missing verl environment. Run bash scripts/cloud_setup.sh first." >&2
  exit 1
fi
if [[ ! -d "${RUN}" || ! -f "${RUN}/metrics.json" ]]; then
  echo "Usage: bash scripts/prepare_m12_repair_stage.sh {8b|32b} RUN_DIRECTORY" >&2
  exit 1
fi

mkdir -p data/processed/agent_sft_source_m12
case "${STAGE}" in
  8b)
    "${VERL_PYTHON}" scripts/prepare_m11_repair_escalation.py \
      --run "${RUN}" \
      --failure-pool data/processed/codecontests_plus_repair_1000_v2/failure_pool.jsonl \
      --canonical-output data/processed/agent_sft_source_m12/repair_8b_canonical.jsonl \
      --escalation-output data/processed/agent_sft_source_m12/repair_escalation_32b.jsonl \
      --manifest data/splits/m12_repair_8b_escalation_v1_manifest.json
    echo "Run the remaining tasks with:"
    echo "  bash scripts/cloud_generate_repair_api.sh configs/data/m12_repair_api_codecontests_plus_escalation_32b.yaml"
    ;;
  32b)
    "${VERL_PYTHON}" scripts/prepare_m11_repair_escalation.py \
      --run "${RUN}" \
      --failure-pool data/processed/agent_sft_source_m12/repair_escalation_32b.jsonl \
      --canonical-output data/processed/agent_sft_source_m12/repair_32b_canonical.jsonl \
      --escalation-output data/processed/agent_sft_source_m12/repair_unresolved.jsonl \
      --manifest data/splits/m12_repair_32b_final_v1_manifest.json
    ;;
  *)
    echo "Usage: bash scripts/prepare_m12_repair_stage.sh {8b|32b} RUN_DIRECTORY" >&2
    exit 1
    ;;
esac
