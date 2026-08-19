# syntax=docker/dockerfile:1.7

# Reproducible NVIDIA training image for Qwen3-0.6B + verl (FSDP + vLLM).
# The host must provide an NVIDIA driver compatible with CUDA 13 (>= 580)
# and NVIDIA Container Toolkit. GPU drivers are intentionally not installed
# in the image.
ARG CUDA_VERSION=13.0.2
ARG UV_VERSION=0.11.16

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12
ARG VERL_REPO=https://github.com/verl-project/verl.git
ARG VERL_REF=b256ebf83b304d83be5c1207fdf6867c04a0d077

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CUDA_HOME=/usr/local/cuda \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=500 \
    UV_NO_PROGRESS=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    RAY_DEDUP_LOGS=0 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        git-lfs \
        libibverbs-dev \
        libnuma-dev \
        librdmacm-dev \
        ninja-build \
        numactl \
        openssh-client \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*

# Use a pinned uv binary rather than downloading an installer script.
COPY --from=uv /uv /uvx /usr/local/bin/

# verl owns the CUDA/PyTorch/vLLM/Transformers dependency graph. Its committed
# lock installs the validated vLLM rollout + FSDP training combination.
RUN git clone --filter=blob:none "${VERL_REPO}" /opt/verl \
    && git -C /opt/verl checkout --detach "${VERL_REF}" \
    && test "$(git -C /opt/verl rev-parse HEAD)" = "${VERL_REF}"

WORKDIR /opt/verl
RUN python${PYTHON_VERSION} manage_envs.py sync vllm fsdp -- --frozen

# Make the verl-managed environment the only Python environment used by Ray
# workers and project commands.
ENV VIRTUAL_ENV=/opt/verl/.venv
ENV PATH="/opt/verl/.venv/bin:${PATH}"

WORKDIR /workspace/project
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

# Do not resolve project cloud dependencies here: verl's lock is authoritative
# for torch, transformers, vLLM, Ray, and their native CUDA dependencies.
RUN uv pip install --no-deps --editable . \
    && python scripts/docker_smoke_test.py --skip-gpu

CMD ["bash"]
