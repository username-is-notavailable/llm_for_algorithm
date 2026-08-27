#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/cloud_cache.sh"

VERL_PYTHON="${PROJECT_ROOT}/.third_party/verl/.venv/bin/python"
if [[ ! -x "${VERL_PYTHON}" ]]; then
  echo "Missing verl Python environment: ${VERL_PYTHON}" >&2
  exit 1
fi

"${VERL_PYTHON}" scripts/prepare_agent_sft_dataset.py "$@"
