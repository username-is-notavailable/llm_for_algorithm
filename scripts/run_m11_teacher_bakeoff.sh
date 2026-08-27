#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_PYTHON="${VERL_ROOT:-${PROJECT_ROOT}/.third_party/verl}/.venv/bin/python"
cd "${PROJECT_ROOT}"
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is not set." >&2
  exit 1
fi

"${VERL_PYTHON}" scripts/prepare_m11_teacher_bakeoff.py \
  --input data/processed/agent_sft_source_v1/repair_escalation_32b.jsonl \
  --output data/processed/agent_sft_source_v1/repair_teacher_bakeoff_20.jsonl \
  --manifest data/splits/m11_teacher_bakeoff_20_v1_manifest.json \
  --size 20

echo "[1/2] qwen3-32b"
bash scripts/cloud_generate_repair_api.sh configs/data/m11_repair_api_bakeoff_32b.yaml
echo "[2/2] qwen3-coder-next"
bash scripts/cloud_generate_repair_api.sh configs/data/m11_repair_api_bakeoff_coder_next.yaml
