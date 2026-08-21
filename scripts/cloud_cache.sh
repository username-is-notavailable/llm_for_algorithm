#!/usr/bin/env bash

# This file is sourced by the cloud entry points. A cache uploaded alongside
# the project takes precedence; otherwise each tool keeps its normal default.
PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT:-${PROJECT_ROOT}/cache}"

if [[ -d "${PROJECT_CACHE_ROOT}/uv" ]]; then
  export UV_CACHE_DIR="${PROJECT_CACHE_ROOT}/uv"
  echo "Using project uv cache: ${UV_CACHE_DIR}"

  # Exported uv caches can contain absolute links back to the machine that
  # created them (for example /root/.cache/uv/archive-v0/...). Rewrite only
  # links whose target is present in this project cache, making it portable.
  REWRITTEN_UV_LINKS=0
  while IFS= read -r -d '' LINK_PATH; do
    LINK_TARGET="$(readlink "${LINK_PATH}")"
    case "${LINK_TARGET}" in
      */.cache/uv/*)
        CACHE_RELATIVE_TARGET="${LINK_TARGET#*/.cache/uv/}"
        LOCAL_TARGET="${UV_CACHE_DIR}/${CACHE_RELATIVE_TARGET}"
        if [[ -e "${LOCAL_TARGET}" ]]; then
          RELATIVE_TARGET="$(realpath --relative-to="$(dirname "${LINK_PATH}")" "${LOCAL_TARGET}")"
          ln -sfn "${RELATIVE_TARGET}" "${LINK_PATH}"
          REWRITTEN_UV_LINKS=$((REWRITTEN_UV_LINKS + 1))
        fi
        ;;
    esac
  done < <(find "${UV_CACHE_DIR}" -type l -print0)
  if [[ "${REWRITTEN_UV_LINKS}" -gt 0 ]]; then
    echo "Rewrote ${REWRITTEN_UV_LINKS} non-portable uv cache links."
  fi
else
  echo "Project uv cache not found; using uv default cache."
fi

if [[ -d "${PROJECT_CACHE_ROOT}/huggingface" ]]; then
  export HF_HOME="${PROJECT_CACHE_ROOT}/huggingface"
  echo "Using project Hugging Face cache: ${HF_HOME}"
else
  echo "Project Hugging Face cache not found; using Hugging Face default cache."
fi
