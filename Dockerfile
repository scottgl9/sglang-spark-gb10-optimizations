# syntax=docker/dockerfile:1
#
# Dockerfile — SGLang GB10 (Spark / ASUS Ascent GX10, NVIDIA GB10 Grace Blackwell,
# SM 12.1, aarch64) build, containerized.
#
# This intentionally does NOT reuse docker/Dockerfile's upstream multi-stage,
# prebuilt-wheel pipeline: this fork's actual validated build path is
# `sglang.sh build`, which installs a pinned torch 2.9.1+cu130, a
# SM_121a-enabled sgl-kernel cu130 wheel (the PyPI wheel lacks SM_121a), and
# then patches the installed flashinfer site-packages for GB10 compatibility
# (see sglang.sh's cmd_build for the full, commented step list and the
# torch/triton/flashinfer/sgl-kernel version-pin rationale). Running that same
# script inside the image keeps the container byte-for-byte the same build
# this fork validates on bare metal.
#
# Build (from repo root):
#   docker build -t sglang-gb10:latest .
#
# Run — mirrors the cache-volume + GPU pattern already used by this fork's
# systemd service unit (~/.config/systemd/user/vllm-laguna.service) and the
# volume layout documented in sglang.sh's setup_runtime_env():
#   docker run -d --name sglang-laguna \
#     --gpus all --ipc=host \
#     --ulimit memlock=-1 \
#     -p 8000:8000 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -v ~/.cache/sglang_compilers:/root/.cache/sglang_compilers \
#     -e SGLANG_PRESET=laguna-s-2.1 \
#     -e CONTEXT_LENGTH=262144 \
#     sglang-gb10:latest
#
# Override the model path / any sglang.sh preset env var with -e, e.g.:
#   -e LAGUNA_MODEL=/root/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/snapshots/<rev>
#
# Other presets (see sglang.sh --help for the full list and their env vars):
#   -e SGLANG_PRESET=Qwen3.6-35B-NVFP4
#   -e SGLANG_PRESET=minimax-m27
#
# Drop into a shell instead of launching a server:
#   docker run -it --gpus all --entrypoint bash sglang-gb10:latest

FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Python 3.12 ships in Ubuntu 24.04 main -- no deadsnakes PPA needed.
RUN --mount=type=cache,target=/var/cache/apt,id=sglang-gb10-apt \
    apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      ca-certificates git curl wget vim \
      build-essential cmake ninja-build \
      libnuma1 libnuma-dev numactl \
      locales \
    && locale-gen en_US.UTF-8 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain: sglang's setuptools-rust extensions (sglang-grpc, sglang-mm)
# discover/build their crates at `pip install -e` time and fail without cargo
# on PATH (see docker/Dockerfile's torch_deps stage for the same requirement).
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal \
    && rustc --version && cargo --version

WORKDIR /sgl-workspace/sglang

# Copy source last-ordering would normally help caching, but sglang.sh build
# installs from this checkout in editable mode (`pip install -e python[all]`),
# so the source has to be present before the build step regardless. The
# .dockerignore (symlinked to .gitignore) already excludes .sglang/, .git/
# history bloat, and other build artifacts from the context.
COPY . .

# Build the GB10 venv into .sglang/ (torch 2.9.1+cu130, sglang[all], SM_121a
# sgl-kernel wheel, flashinfer GB10 compatibility patches). Runs as root here
# since the container itself is the isolation boundary.
RUN --mount=type=cache,target=/root/.cache/pip,id=sglang-gb10-pip \
    CUDA_HOME=/usr/local/cuda bash sglang.sh build

# Mount these at `docker run` time to persist model snapshots and JIT/compiler
# caches across container restarts (same dirs sglang.sh's setup_runtime_env()
# creates under $HOME on bare metal):
#   /root/.cache/huggingface        HF model snapshots
#   /root/.cache/sglang_compilers   triton / nv / flashinfer / torch caches
VOLUME ["/root/.cache/huggingface", "/root/.cache/sglang_compilers"]

EXPOSE 8000

# Default preset + context length; override either with `docker run -e`.
# See sglang.sh's usage() for the full preset list and per-preset env vars
# (LAGUNA_MODEL, QWEN36_35B_MODEL, MINIMAX_MODEL, DISABLE_DFLASH, ...).
ENV SGLANG_PRESET=laguna-s-2.1 \
    CONTEXT_LENGTH=262144

# Launches `sglang.sh "$SGLANG_PRESET" <extra CMD args>`. Any args passed to
# `docker run sglang-gb10:latest ...` are forwarded to the preset (or to
# `sglang.sh` directly for "build"/"launch"/"shell").
ENTRYPOINT ["/sgl-workspace/sglang/docker-entrypoint.sh"]
CMD []
