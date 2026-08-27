#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-${PROJECT_ROOT}/.third_party/verl}"
VERL_PYTHON="${VERL_ROOT}/.venv/bin/python"
ONE_SHOT_CONFIG="${M11_ONE_SHOT_CONFIG:-configs/data/m11_agent_one_shot_300_v1.yaml}"
REPAIR_CONFIG="${M11_REPAIR_CONFIG:-configs/data/m11_repair_api_codecontests_plus_300_8b.yaml}"

if [[ ! -x "${VERL_PYTHON}" ]]; then
  echo "Missing verl environment. Run bash scripts/cloud_setup.sh first." >&2
  exit 1
fi
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/cloud_cache.sh"

echo "[1/2] Verifying and building one-shot Agent trajectories"
"${VERL_PYTHON}" scripts/prepare_agent_one_shot.py --config "${ONE_SHOT_CONFIG}"

echo "[2/2] Generating qwen3-8b repair trajectories"
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "One-shot data is complete. Set DASHSCOPE_API_KEY and rerun this script for API repair." >&2
  exit 2
fi
bash scripts/cloud_generate_repair_api.sh "${REPAIR_CONFIG}" "$@"
