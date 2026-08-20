#!/usr/bin/env bash

# This file is sourced by the cloud entry points. A cache uploaded alongside
# the project takes precedence; otherwise each tool keeps its normal default.
PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT:-${PROJECT_ROOT}/cache}"

if [[ -d "${PROJECT_CACHE_ROOT}/uv" ]]; then
  export UV_CACHE_DIR="${PROJECT_CACHE_ROOT}/uv"
  echo "Using project uv cache: ${UV_CACHE_DIR}"
else
  echo "Project uv cache not found; using uv default cache."
fi

if [[ -d "${PROJECT_CACHE_ROOT}/huggingface" ]]; then
  export HF_HOME="${PROJECT_CACHE_ROOT}/huggingface"
  echo "Using project Hugging Face cache: ${HF_HOME}"
else
  echo "Project Hugging Face cache not found; using Hugging Face default cache."
fi
