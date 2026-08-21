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
if [[ ! -f data/processed/livecodebench_v1/smoke_10.jsonl ]]; then
  "${VERL_PYTHON}" scripts/prepare_eval.py
fi
exec "${VERL_PYTHON}" -m src.eval.evaluator --config configs/eval/livecodebench_smoke_v1.yaml "$@"
