#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
UV_VERSION="0.11.16"
VERL_REPO="https://github.com/verl-project/verl.git"
VERL_REF="b256ebf83b304d83be5c1207fdf6867c04a0d077"
VERL_ROOT="${VERL_ROOT:-${PROJECT_ROOT}/.third_party/verl}"
ALIYUN_PYPI="https://mirrors.aliyun.com/pypi/simple/"

# Prefer the Aliyun mirror. Keep official PyPI as the default fallback because
# newly released PyTorch/CUDA wheels may not have reached the mirror yet.
export UV_INDEX="${UV_INDEX:-aliyun=${ALIYUN_PYPI}}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.org/simple}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"

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
"${PYTHON_BIN}" -m pip install \
  --root-user-action=ignore \
  --index-url "${ALIYUN_PYPI}" \
  "uv==${UV_VERSION}"
UV_BIN="$("${PYTHON_BIN}" -c 'import shutil; print(shutil.which("uv") or "")')"
if [[ -z "${UV_BIN}" ]]; then
  echo "uv was installed but is not available on PATH" >&2
  exit 1
fi

mkdir -p .third_party
ACTUAL_VERL_REF="$(git -C "${VERL_ROOT}" rev-parse HEAD 2>/dev/null || true)"
if [[ "${ACTUAL_VERL_REF}" != "${VERL_REF}" || ! -f "${VERL_ROOT}/manage_envs.py" ]]; then
  VERL_STAGING="$(mktemp -d "${PROJECT_ROOT}/.third_party/verl.staging.XXXXXX")"
  git -C "${VERL_STAGING}" init --quiet
  git -C "${VERL_STAGING}" remote add origin "${VERL_REPO}"

  FETCHED=0
  for ATTEMPT in 1 2 3; do
    echo "Fetching pinned verl commit (attempt ${ATTEMPT}/3)..."
    if git -c http.version=HTTP/1.1 -C "${VERL_STAGING}" \
      fetch --no-tags --depth=1 origin "${VERL_REF}"; then
      FETCHED=1
      break
    fi
  done
  if [[ "${FETCHED}" -ne 1 ]]; then
    echo "Failed to fetch verl after 3 attempts; staging directory: ${VERL_STAGING}" >&2
    exit 1
  fi

  git -C "${VERL_STAGING}" checkout --detach FETCH_HEAD
  STAGED_VERL_REF="$(git -C "${VERL_STAGING}" rev-parse HEAD)"
  if [[ "${STAGED_VERL_REF}" != "${VERL_REF}" || ! -f "${VERL_STAGING}/manage_envs.py" ]]; then
    echo "Fetched verl checkout failed validation: ${VERL_STAGING}" >&2
    exit 1
  fi

  if [[ -e "${VERL_ROOT}" ]]; then
    FAILED_VERL_ROOT="${VERL_ROOT}.failed-$(date +%Y%m%d-%H%M%S)"
    mv "${VERL_ROOT}" "${FAILED_VERL_ROOT}"
    echo "Archived incomplete verl checkout at: ${FAILED_VERL_ROOT}"
  fi
  mv "${VERL_STAGING}" "${VERL_ROOT}"
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
