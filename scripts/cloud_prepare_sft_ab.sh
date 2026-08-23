#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-${PROJECT_ROOT}/.third_party/verl}"
VERL_PYTHON="${VERL_ROOT}/.venv/bin/python"

if [[ ! -x "${VERL_PYTHON}" ]]; then
  echo "Missing verl environment. Run bash scripts/cloud_setup.sh first." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/cloud_cache.sh"
export VERL_ROOT
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "${VERL_PYTHON}" scripts/prepare_sft_ab.py "$@"
