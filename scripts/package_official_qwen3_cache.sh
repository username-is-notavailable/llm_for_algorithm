#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HUB_ROOT="${PROJECT_ROOT}/cache/huggingface/hub"
ARCHIVE="${1:-${PROJECT_ROOT}/qwen3-official-4b-8b-cache.tar.zst}"
MODELS=(models--Qwen--Qwen3-4B models--Qwen--Qwen3-8B)

for model in "${MODELS[@]}"; do
  if [[ ! -d "${HUB_ROOT}/${model}" ]]; then
    echo "Missing cache directory: ${HUB_ROOT}/${model}" >&2
    exit 1
  fi
done
if ! command -v zstd >/dev/null 2>&1; then
  echo "zstd is required to create ${ARCHIVE}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
tar -I 'zstd -T0 -1' -cf "${ARCHIVE}" \
  "cache/huggingface/hub/${MODELS[0]}" \
  "cache/huggingface/hub/${MODELS[1]}"
ARCHIVE_DIR="$(dirname "${ARCHIVE}")"
ARCHIVE_NAME="$(basename "${ARCHIVE}")"
(
  cd "${ARCHIVE_DIR}"
  sha256sum "${ARCHIVE_NAME}" > "${ARCHIVE_NAME}.sha256"
)
du -h "${ARCHIVE}" "${ARCHIVE}.sha256"
