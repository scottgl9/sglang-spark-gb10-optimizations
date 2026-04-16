#!/bin/bash
# entrypoint-gb10.sh — Docker entrypoint for SGLang on GB10
#
# Usage:
#   docker run ... sglang-gb10 Qwen3.5-NVFP4           # model preset
#   docker run ... sglang-gb10 launch --model-path ...  # raw sglang args
#   docker run ... sglang-gb10 bash                     # shell access

set -euo pipefail

# ── Compiler cache directories ──────────────────────────────────────────────
# Persist via: -v sglang-compilers:/root/.cache/sglang_compilers
CACHE_DIR="/root/.cache/sglang_compilers"
mkdir -p \
    "${CACHE_DIR}/triton" \
    "${CACHE_DIR}/nv/ComputeCache" \
    "${CACHE_DIR}/flashinfer" \
    "${CACHE_DIR}/torch/inductor"

export CUDA_CACHE_PATH="${CACHE_DIR}/nv/ComputeCache"
export CUDA_CACHE_MAXSIZE=4294967296
export TRITON_CACHE_DIR="${CACHE_DIR}/triton"
export FLASHINFER_WORKSPACE_DIR="${CACHE_DIR}/flashinfer"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR}/torch/inductor"
export TORCH_COMPILE_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=4
export CUDA_NVCC_FLAGS="${CUDA_NVCC_FLAGS:---threads 4}"

# ── Defaults ────────────────────────────────────────────────────────────────
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
SERVER_ARGS=(--host 0.0.0.0 --port 8000 --kv-cache-dtype "${KV_CACHE_DTYPE}")

info() { echo -e "\033[1;34m[sglang-gb10]\033[0m $*"; }

# ── Launch helper ───────────────────────────────────────────────────────────
do_launch() {
    info "Launching SGLang server"
    info "  KV_CACHE_DTYPE = ${KV_CACHE_DTYPE}"
    info "  CONTEXT_LENGTH = ${CONTEXT_LENGTH}"
    exec python3 -m sglang.launch_server "${SERVER_ARGS[@]}" "$@"
}

# ── Model presets ───────────────────────────────────────────────────────────
# Each preset mirrors the corresponding cmd_* function in sglang.sh
# Model paths default to HuggingFace Hub IDs; override with env vars.

preset_qwen35_nvfp4() {
    local model="${QWEN35_MODEL:-Sehyo/Qwen3.5-122B-A10B-NVFP4}"
    local spec_args=()
    if [[ "${DISABLE_MTP:-}" != "1" ]]; then
        spec_args=(
            --speculative-algorithm NEXTN
            --speculative-num-steps 2
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens 2
            --mamba-scheduler-strategy extra_buffer
        )
        export SGLANG_ENABLE_SPEC_V2=1
    fi
    info "Preset: Qwen3.5-122B-A10B-NVFP4 | Model: ${model}"
    do_launch \
        --model-path "${model}" \
        --served-model-name qwen3-coder-next \
        --quantization compressed-tensors \
        --speculative-draft-model-quantization compressed-tensors \
        --mem-fraction-static 0.85 \
        --context-length "${CONTEXT_LENGTH}" \
        --max-running-requests 3 \
        --attention-backend flashinfer \
        --linear-attn-backend triton \
        --linear-attn-prefill-backend triton \
        --chunked-prefill-size 16384 \
        --mamba-full-memory-ratio auto \
        --disable-piecewise-cuda-graph \
        --disable-multimodal \
        "${spec_args[@]}" \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --trust-remote-code \
        "$@"
}

preset_qwen35_35b_nvfp4() {
    local model="${QWEN35_35B_MODEL:-Sehyo/Qwen3.5-35B-A3B-NVFP4}"
    local spec_args=()
    if [[ "${DISABLE_MTP:-}" != "1" ]]; then
        spec_args=(
            --speculative-algorithm NEXTN
            --speculative-num-steps 2
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens 2
            --mamba-scheduler-strategy extra_buffer
        )
        export SGLANG_ENABLE_SPEC_V2=1
    fi
    info "Preset: Qwen3.5-35B-A3B-NVFP4 | Model: ${model}"
    do_launch \
        --model-path "${model}" \
        --served-model-name qwen3-coder-next \
        --quantization compressed-tensors \
        --mem-fraction-static 0.75 \
        --context-length "${CONTEXT_LENGTH}" \
        --max-running-requests 3 \
        --attention-backend flashinfer \
        --linear-attn-prefill-backend triton \
        --chunked-prefill-size 16384 \
        --mamba-full-memory-ratio auto \
        --disable-multimodal \
        "${spec_args[@]}" \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --trust-remote-code \
        "$@"
}

preset_qwen3_coder_next_nvfp4() {
    local model="${QWEN3_CODER_NVFP4_MODEL:-GadflyII/Qwen3-Coder-Next-NVFP4}"
    local ctx="${CONTEXT_LENGTH:-131072}"
    info "Preset: Qwen3-Coder-Next-NVFP4 | Model: ${model}"
    do_launch \
        --model-path "${model}" \
        --served-model-name qwen3-coder-next \
        --quantization compressed-tensors \
        --mem-fraction-static 0.85 \
        --context-length "${ctx}" \
        --attention-backend triton \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --trust-remote-code \
        "$@"
}

preset_qwen3_coder_next_fp8() {
    local model="${QWEN3_CODER_MODEL:-Qwen/Qwen3-Coder-Next-FP8}"
    local ctx="${CONTEXT_LENGTH:-131072}"
    info "Preset: Qwen3-Coder-Next-FP8 | Model: ${model}"
    do_launch \
        --model-path "${model}" \
        --quantization fp8 \
        --mem-fraction-static 0.85 \
        --max-running-requests 8 \
        --context-length "${ctx}" \
        --chunked-prefill-size 8192 \
        --attention-backend triton \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --trust-remote-code \
        "$@"
}

preset_minimax() {
    local model="${MINIMAX_MODEL:-saricles/MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10}"
    local ctx="${CONTEXT_LENGTH:-65536}"
    local spec_args=()
    if [[ "${DISABLE_NGRAM:-}" != "1" ]]; then
        spec_args=(
            --speculative-algorithm NGRAM
            --speculative-num-draft-tokens 5
            --speculative-ngram-max-match-window-size 16
        )
    fi
    info "Preset: MiniMax M2.5 REAP 139B NVFP4 | Model: ${model}"
    do_launch \
        --model-path "${model}" \
        --served-model-name MiniMax-M2.5 \
        --quantization modelopt_fp4 \
        --mem-fraction-static 0.88 \
        --max-running-requests 8 \
        --context-length "${ctx}" \
        --attention-backend triton \
        --moe-runner-backend flashinfer_cutlass \
        --enable-eplb \
        --ep-num-redundant-experts 8 \
        --reasoning-parser minimax \
        --tool-call-parser minimax-m2 \
        --trust-remote-code \
        --disable-cuda-graph \
        "${spec_args[@]}" \
        "$@"
}

preset_nemotron() {
    local model="${NEMOTRON_MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"
    local spec_args=()
    if [[ "${DISABLE_MTP:-}" != "1" ]]; then
        spec_args=(
            --speculative-algorithm NEXTN
            --speculative-num-steps 1
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens 1
        )
        export SGLANG_ENABLE_SPEC_V2=1
    fi
    export SGLANG_NEMOTRON_FP8_POST_QUANT=1
    info "Preset: Nemotron-3-Super-120B-A12B-NVFP4 | Model: ${model}"
    do_launch \
        --model-path "${model}" \
        --served-model-name nemotron \
        --quantization modelopt_fp8 \
        --mem-fraction-static 0.88 \
        --context-length "${CONTEXT_LENGTH}" \
        --max-running-requests 4 \
        --attention-backend flashinfer \
        --linear-attn-backend triton \
        --linear-attn-prefill-backend triton \
        --chunked-prefill-size 16384 \
        --moe-runner-backend triton \
        --disable-radix-cache \
        --disable-multimodal \
        "${spec_args[@]}" \
        --reasoning-parser nemotron_3 \
        --tool-call-parser qwen3_coder \
        --trust-remote-code \
        "$@"
}

preset_mistral_small4() {
    local model="${MISTRAL_MODEL:-mistralai/Mistral-Small-4-119B-2603-NVFP4}"
    local eagle="${MISTRAL_EAGLE_MODEL:-mistralai/Mistral-Small-4-119B-2603-eagle}"
    local spec_args=()
    if [[ "${ENABLE_EAGLE:-}" == "1" ]]; then
        spec_args=(
            --speculative-algorithm EAGLE
            --speculative-draft-model-path "${eagle}"
            --speculative-draft-model-quantization fp8
            --speculative-num-steps 3
            --speculative-eagle-topk 4
            --speculative-num-draft-tokens 16
        )
        export SGLANG_QUANTIZE_LM_HEAD_FP8=0
    fi
    info "Preset: Mistral-Small-4-119B NVFP4 | Model: ${model}"
    do_launch \
        --model-path "${model}" \
        --served-model-name mistral-small-4 \
        --quantization compressed-tensors \
        --mem-fraction-static 0.88 \
        --context-length "${CONTEXT_LENGTH}" \
        --max-running-requests 4 \
        --attention-backend triton \
        --chunked-prefill-size 16384 \
        --chat-template mistral \
        --tool-call-parser mistral \
        --disable-multimodal \
        --trust-remote-code \
        "${spec_args[@]}" \
        "$@"
}

# ── Dispatch ────────────────────────────────────────────────────────────────
CMD="${1:-}"
shift || true

case "${CMD}" in
    # Model presets
    Qwen3.5-NVFP4|qwen3.5-nvfp4|qwen35-nvfp4)
        preset_qwen35_nvfp4 "$@" ;;
    Qwen3.5-35B-NVFP4|qwen3.5-35b-nvfp4|qwen35-35b-nvfp4)
        preset_qwen35_35b_nvfp4 "$@" ;;
    Qwen3-Coder-Next-NVFP4|qwen3-coder-next-nvfp4)
        preset_qwen3_coder_next_nvfp4 "$@" ;;
    Qwen3-Coder-Next-FP8|qwen3-coder-next-fp8)
        preset_qwen3_coder_next_fp8 "$@" ;;
    minimax|MiniMax)
        preset_minimax "$@" ;;
    nemotron|Nemotron|nemotron-3-super)
        preset_nemotron "$@" ;;
    mistral-small-4|Mistral-Small-4|mistral-small4)
        preset_mistral_small4 "$@" ;;

    # Raw launch mode
    launch)
        do_launch "$@" ;;

    # Help
    ""|help|-h|--help)
        cat <<'EOF'
SGLang GB10 Docker — Model presets:

  Qwen3.5-NVFP4           Qwen3.5-122B MoE NVFP4 + MTP
  Qwen3.5-35B-NVFP4       Qwen3.5-35B-A3B NVFP4 + MTP
  Qwen3-Coder-Next-NVFP4  GadflyII Qwen3-Coder-Next NVFP4
  Qwen3-Coder-Next-FP8    Qwen3-Coder-Next dense FP8
  minimax                  MiniMax M2.5 REAP 139B NVFP4
  nemotron                 Nemotron-3-Super-120B NVFP4 + MTP
  mistral-small-4          Mistral-Small-4-119B NVFP4

  launch [args]            Raw sglang.launch_server args
  bash / python / ...      Run any command

Environment overrides:
  CONTEXT_LENGTH=N         Context window (default: 65536)
  KV_CACHE_DTYPE=X         KV cache dtype (default: fp8_e4m3)
  DISABLE_MTP=1            Disable speculative decoding
  DISABLE_NGRAM=1          Disable NGRAM speculation (minimax)
  ENABLE_EAGLE=1           Enable EAGLE (mistral-small-4)
  HF_TOKEN=...             HuggingFace token for gated models

Model path overrides:
  QWEN35_MODEL, QWEN35_35B_MODEL, MINIMAX_MODEL,
  NEMOTRON_MODEL, MISTRAL_MODEL, MISTRAL_EAGLE_MODEL,
  QWEN3_CODER_NVFP4_MODEL, QWEN3_CODER_MODEL
EOF
        ;;

    # Arbitrary command (bash, python, etc.)
    *)
        exec "${CMD}" "$@" ;;
esac
