#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-${PROJECT_ROOT}/.third_party/verl}"
VERL_PYTHON="${VERL_ROOT}/.venv/bin/python"

if [[ ! -x "${VERL_PYTHON}" ]]; then
  echo "Missing verl environment. Run bash scripts/cloud_setup.sh first." >&2
  exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/data/splits/eval_v1_fingerprints.json" ]]; then
  echo "Missing committed Eval fingerprints." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/cloud_cache.sh"
export VERL_ROOT

MAX_ATTEMPTS="${SFT_PREPARE_MAX_ATTEMPTS:-20}"
RETRY_DELAY_SECONDS="${SFT_PREPARE_RETRY_DELAY_SECONDS:-5}"
if [[ ! "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SFT_PREPARE_MAX_ATTEMPTS must be a positive integer." >&2
  exit 2
fi
if [[ ! "${RETRY_DELAY_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "SFT_PREPARE_RETRY_DELAY_SECONDS must be a non-negative integer." >&2
  exit 2
fi

for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
  if "${VERL_PYTHON}" scripts/prepare_sft.py "$@"; then
    exit 0
  else
    status=$?
  fi
  if ((attempt == MAX_ATTEMPTS)); then
    echo "SFT preparation failed after ${MAX_ATTEMPTS} attempts (last exit ${status})." >&2
    exit "${status}"
  fi
  echo "SFT preparation exited with ${status}; restarting from its checkpoint in ${RETRY_DELAY_SECONDS}s (${attempt}/${MAX_ATTEMPTS})." >&2
  sleep "${RETRY_DELAY_SECONDS}"
done
