#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_PYTHON="${PROJECT_ROOT}/.third_party/verl/.venv/bin/python"
PROJECT_HF_CACHE="${PROJECT_ROOT}/cache/huggingface"
AUDIT_HF_CACHE="${PROJECT_ROOT}/cache/m10_source_audit"

if [[ "${1:-}" != "--purge" ]]; then
  echo "Usage: bash scripts/redownload_audit_m10_sources.sh --purge" >&2
  echo "This removes only the cached OCR2/TACO repository metadata and the M10 audit cache." >&2
  exit 2
fi
if [[ ! -x "${VERL_PYTHON}" ]]; then
  echo "Missing verl Python: ${VERL_PYTHON}" >&2
  exit 1
fi
case "${PROJECT_HF_CACHE}" in
  "${PROJECT_ROOT}"/cache/huggingface) ;;
  *) echo "Refusing unexpected cache root: ${PROJECT_HF_CACHE}" >&2; exit 1 ;;
esac
case "${AUDIT_HF_CACHE}" in
  "${PROJECT_ROOT}"/cache/m10_source_audit) ;;
  *) echo "Refusing unexpected audit cache: ${AUDIT_HF_CACHE}" >&2; exit 1 ;;
esac

targets=(
  "${PROJECT_HF_CACHE}/hub/datasets--nvidia--OpenCodeReasoning-2"
  "${PROJECT_HF_CACHE}/hub/.locks/datasets--nvidia--OpenCodeReasoning-2"
  "${PROJECT_HF_CACHE}/hub/datasets--BAAI--TACO"
  "${PROJECT_HF_CACHE}/hub/.locks/datasets--BAAI--TACO"
  "${AUDIT_HF_CACHE}"
)
echo "Removing only these cache paths:"
printf '  %s\n' "${targets[@]}"
rm -rf -- "${targets[@]}"

mkdir -p "${AUDIT_HF_CACHE}"
export HF_HOME="${AUDIT_HF_CACHE}"
export HF_HUB_CACHE="${AUDIT_HF_CACHE}/hub"
echo "Fresh isolated HF cache: ${HF_HOME}"

cd "${PROJECT_ROOT}"
"${VERL_PYTHON}" scripts/audit_m10_sources.py
