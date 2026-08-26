#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/scripts/cloud_cache.sh"
VERL_PYTHON="${PROJECT_ROOT}/.third_party/verl/.venv/bin/python"

"${VERL_PYTHON}" scripts/audit_codecontests_plus.py "$@"
