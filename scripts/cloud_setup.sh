#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
UV_VERSION="0.11.16"
VERL_REPO="https://github.com/verl-project/verl.git"
VERL_REF="b256ebf83b304d83be5c1207fdf6867c04a0d077"
VERL_ROOT="${VERL_ROOT:-${PROJECT_ROOT}/.third_party/verl}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"Python 3.12 is required, got {sys.version.split()[0]}. "
        "Activate the bootstrap Conda environment first."
    )
PY

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v g++ >/dev/null || { echo "g++ with C++17 support is required" >&2; exit 1; }

# Conda only supplies the bootstrap Python. uv and verl's committed lock own
# the actual training environment and its CUDA dependency graph.
"${PYTHON_BIN}" -m pip install "uv==${UV_VERSION}"
UV_BIN="$("${PYTHON_BIN}" -c 'import shutil; print(shutil.which("uv") or "")')"
if [[ -z "${UV_BIN}" ]]; then
  echo "uv was installed but is not available on PATH" >&2
  exit 1
fi

mkdir -p .third_party
if [[ ! -d "${VERL_ROOT}/.git" ]]; then
  git clone --filter=blob:none "${VERL_REPO}" "${VERL_ROOT}"
fi
git -C "${VERL_ROOT}" fetch --tags origin
git -C "${VERL_ROOT}" checkout --detach "${VERL_REF}"
ACTUAL_VERL_REF="$(git -C "${VERL_ROOT}" rev-parse HEAD)"
if [[ "${ACTUAL_VERL_REF}" != "${VERL_REF}" ]]; then
  echo "VERL_REF must resolve exactly to a full commit: ${VERL_REF}" >&2
  exit 1
fi

(
  cd "${VERL_ROOT}"
  PATH="$(dirname "${UV_BIN}"):${PATH}" \
    "${PYTHON_BIN}" manage_envs.py sync vllm fsdp -- --frozen
)

VERL_PYTHON="${VERL_ROOT}/.venv/bin/python"
if [[ ! -x "${VERL_PYTHON}" ]]; then
  echo "verl environment was not created: ${VERL_PYTHON}" >&2
  exit 1
fi

"${UV_BIN}" pip install \
  --python "${VERL_PYTHON}" \
  --no-deps \
  --editable "${PROJECT_ROOT}"

echo "Cloud environment ready."
echo "Python: ${VERL_PYTHON}"
echo "Next: bash scripts/cloud_smoke_test.sh"
