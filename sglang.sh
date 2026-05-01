#!/bin/bash
# sglang.sh — Build and launch SGLang on DGX GB10 (SM 12.1 / CUDA 13.0)
#
# Commands:
#   build [--skip-*]               Build SGLang into .sglang/ venv (run once)
#   launch [args]                  Start the OpenAI-compatible API server (raw args)
#   shell                          Drop into an activated venv shell
#   Qwen3.5-NVFP4 [args]          Qwen3.5-122B MoE NVFP4 + speculative decoding
#   Qwen3.5-35B-NVFP4 [args]      Sehyo/Qwen3.5-35B-A3B-NVFP4 + speculative decoding
#   Qwen3-Coder-Next-NVFP4 [args] GadflyII Qwen3-Coder-Next NVFP4
#   Qwen3-Coder-Next-FP8 [args]   Qwen/Qwen3-Coder-Next dense FP8
#   minimax-m27 [args]             MiniMax M2.7 REAP 172B NVFP4-GB10 (compressed-tensors)
#   nemotron [args]                NVIDIA Nemotron-3-Super 120B-A12B NVFP4 + MTP
#   mistral-small-4 [args]        Mistral-Small-4-119B NVFP4 + EAGLE
#
# Context window (default 65536 — override with CONTEXT_LENGTH):
#   CONTEXT_LENGTH=32768 ./sglang.sh Qwen3.5-NVFP4
#
# Build:
#   ./sglang.sh build
#   ./sglang.sh build --skip-venv --skip-torch   # partial rebuild
#   rm -rf .sglang && ./sglang.sh build           # clean rebuild
#
# Build options:
#   --skip-venv       Skip venv creation (if already exists)
#   --skip-torch      Skip torch installation
#   --skip-sglang     Skip sglang[all] installation
#   --skip-sgl-kernel Skip sgl-kernel wheel installation
#   --skip-fixes      Skip flashinfer GB10 compatibility patches
#
# Model path overrides:
#   QWEN35_MODEL=/path/to/snapshot               ./sglang.sh Qwen3.5-NVFP4
#   QWEN35_35B_MODEL=Sehyo/...                   ./sglang.sh Qwen3.5-35B-NVFP4
#   QWEN3_CODER_NVFP4_MODEL=GadflyII/...         ./sglang.sh Qwen3-Coder-Next-NVFP4
#   QWEN3_CODER_MODEL=Qwen/Qwen3-Coder-Next-FP8  ./sglang.sh Qwen3-Coder-Next-FP8
#   MINIMAX_MODEL=/path/to/model                  ./sglang.sh minimax-m27
#   NEMOTRON_MODEL=/path/to/model                 ./sglang.sh nemotron
#   MISTRAL_MODEL=/path/to/model                  ./sglang.sh mistral-small-4
#   MISTRAL_EAGLE_MODEL=/path/to/eagle             ./sglang.sh mistral-small-4
#
# Key environment overrides:
#   CONTEXT_LENGTH             Context window tokens (default: 65536)
#   DISABLE_MTP=1              Disable speculative decoding for Qwen3.5-NVFP4 / nemotron
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SGLANG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SGLANG_DIR}/.sglang"
PYTHON="python3.12"

# ── Context window default ────────────────────────────────────────────────────
# Override before calling: CONTEXT_LENGTH=32768 ./sglang.sh <preset>
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"

# ── KV cache dtype default ────────────────────────────────────────────────────
# Default is fp8_e4m3 — validated on SM121 (GB10). CUTLASS FP8 works correctly
# on SM121 and halves KV cache bandwidth vs BF16.
# User can override: KV_CACHE_DTYPE=auto ./sglang.sh ...  (forces BF16)
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"

# ── Shared launch arg groups ────────────────────────────────────────────────

# Standard server binding (all presets)
SERVER_ARGS=(--host 0.0.0.0 --port 8000)

# Common to all Qwen3 presets: model name + tool calling
QWEN3_ARGS=(
    --served-model-name qwen3-coder-next
    --tool-call-parser qwen3_coder
)

# ── Runtime env vars ──────────────────────────────────────────────────────────
setup_runtime_env() {
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export TRITON_PTXAS_PATH="${CUDA_HOME}/bin/ptxas"

    # ── Compiler / JIT caches ─────────────────────────────────────────────────
    # Dedicated per-tool cache dirs under ~/.cache/sglang_compilers/, mirroring
    # the docker volume layout used in the vllm container scripts:
    #   vllm docker: -v ~/.cache/vllm_compilers/triton:/root/.triton
    #                -v ~/.cache/vllm_compilers/nv:/root/.nv
    #                -v ~/.cache/vllm_compilers/flashinfer:/root/.cache/flashinfer
    #                -v ~/.cache/vllm_compilers/torch:/root/.cache/torch
    # Keeping sglang and vllm caches separate avoids stale-cache cross-contamination.
    local SGLANG_COMPILERS_DIR="${HOME}/.cache/sglang_compilers"
    mkdir -p \
        "${SGLANG_COMPILERS_DIR}/triton" \
        "${SGLANG_COMPILERS_DIR}/nv/ComputeCache" \
        "${SGLANG_COMPILERS_DIR}/flashinfer" \
        "${SGLANG_COMPILERS_DIR}/torch"

    # CUDA kernel cache — persists JIT-compiled kernels across restarts
    export CUDA_CACHE_PATH="${SGLANG_COMPILERS_DIR}/nv/ComputeCache"
    export CUDA_CACHE_MAXSIZE=4294967296   # 4 GB

    # Triton JIT cache (default: ~/.triton/cache)
    export TRITON_CACHE_DIR="${SGLANG_COMPILERS_DIR}/triton"

    # FlashInfer + inductor cache dirs
    export FLASHINFER_WORKSPACE_DIR="${SGLANG_COMPILERS_DIR}/flashinfer"
    export TORCHINDUCTOR_CACHE_DIR="${SGLANG_COMPILERS_DIR}/torch/inductor"

    # ── JIT compilation parallelism ──────────────────────────────────────────
    # GB10 unified memory: each nvcc process can use several GB of RAM.
    # Limit parallel compile jobs to avoid OOM during flashinfer JIT builds.
    export MAX_JOBS="${MAX_JOBS:-4}"
    export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}"
    export CUDA_NVCC_FLAGS="${CUDA_NVCC_FLAGS:---threads 4}"
    export TORCH_COMPILE_THREADS="${TORCH_COMPILE_THREADS:-4}"
    export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"

    # Faster safetensors weight loading via GPU pinned memory
    export SAFETENSORS_FAST_GPU=1

    # Post-quantize lm_head from BF16 to FP8 (halves DRAM read cost, ~5% decode speedup)
    export SGLANG_QUANTIZE_LM_HEAD_FP8="${SGLANG_QUANTIZE_LM_HEAD_FP8:-1}"

    # Post-quantize embed_tokens from BF16 to FP8 (saves ~0.6 GB for larger KV cache)
    export SGLANG_QUANTIZE_EMBED_FP8="${SGLANG_QUANTIZE_EMBED_FP8:-1}"

    # Post-quantize MTP draft model layers to FP8 (halves MoE+fc bandwidth per draft step)
    # NOTE: On SM121, this is overridden to 0 below (Triton fp8e4nv dot not supported).
    # Save user's explicit value so SM121 override doesn't clobber it.
    local _user_mtp_fp8="${SGLANG_MTP_FP8:-}"
    export SGLANG_MTP_FP8="${SGLANG_MTP_FP8:-1}"

    # KV cache quantization dtype (default: fp8_e4m3; set to fp4_e2m1 to test NVFP4 KV cache)
    export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"

    # torch.compile / inductor thread count (avoids saturating CPUs during JIT)
    export TORCH_COMPILE_THREADS=4
    export TORCHINDUCTOR_COMPILE_THREADS=4

    # CPU thread count for OMP-parallelized operations
    export OMP_NUM_THREADS=8

    # Expandable segments avoids allocator fragmentation on large models
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # Disable CuDNN version check — torch 2.9.1+cu130 ships CuDNN 9.13 but
    # sglang checks for 9.15+ due to a Conv3d bug (pytorch#168167).
    # Conv3d is not used in LLM inference, so this check is safe to skip.
    export SGLANG_DISABLE_CUDNN_CHECK=1

    # ── SM121 (GB10) specific overrides ────────────────────────────────────
    # Detect SM121 at runtime and apply necessary workarounds:
    #   - KV_CACHE_DTYPE=fp8_e4m3: CUTLASS FP8 validated on SM121 (no override needed)
    #   - SGLANG_ENABLE_JIT_DEEPGEMM=0: DeepGEMM JIT fails on SM121
    #   - SGLANG_MTP_FP8=0: Triton fp8e4nv dot not supported on SM120+
    local gpu_sm
    gpu_sm=$(python -c "
import torch
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability()
    print(f'{cc[0]}.{cc[1]}')
else:
    print('0.0')
" 2>/dev/null || echo "0.0")

    if [[ "${gpu_sm}" == "12.1" ]]; then
        info "Detected SM121 (GB10) — applying workarounds"
        # KV cache: fp8_e4m3 is validated on SM121 (CUTLASS FP8 works, unlike FP4).
        # The script default is already fp8_e4m3 so no override needed.
        export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"
        # Triton fp8e4nv dot not supported on SM120+, so MTP MoE FP8 post-quant
        # is incompatible. Only override if user didn't explicitly set it.
        if [[ -z "${_user_mtp_fp8}" ]]; then
            export SGLANG_MTP_FP8=0
        fi
    fi
}

# ── Helpers ───────────────────────────────────────────────────────────────────
info()    { echo -e "\033[1;34m[sglang]\033[0m $*"; }
success() { echo -e "\033[1;32m[sglang]\033[0m $*"; }
warn()    { echo -e "\033[1;33m[sglang]\033[0m $*"; }
die()     { echo -e "\033[1;31m[sglang]\033[0m ERROR: $*" >&2; exit 1; }

# ── Subcommands ──────────────────────────────────────────────────────────────

cmd_build() {
    # Parse build options
    local SKIP_VENV=0 SKIP_TORCH=0 SKIP_SGLANG=0 SKIP_SGL_KERNEL=0 SKIP_FIXES=0
    for arg in "$@"; do
        case $arg in
            --skip-venv)       SKIP_VENV=1 ;;
            --skip-torch)      SKIP_TORCH=1 ;;
            --skip-sglang)     SKIP_SGLANG=1 ;;
            --skip-sgl-kernel) SKIP_SGL_KERNEL=1 ;;
            --skip-fixes)      SKIP_FIXES=1 ;;
        esac
    done

    local LOG="${SGLANG_DIR}/build_env.log"

    info "Building SGLang on GB10 (SM 12.1, CUDA 13.0)"
    info "  Source : ${SGLANG_DIR}"
    info "  Venv   : ${VENV_DIR}"
    info "  Log    : ${LOG}"
    echo ""

    [[ -f "${SGLANG_DIR}/python/sglang/__init__.py" ]] || die "Run from sglang source root (python/sglang/__init__.py not found)"
    [[ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]] || die "nvcc not found. Is CUDA installed?"

    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

    # ── Step 1: Create virtual environment ──────────────────────────────────
    if [[ $SKIP_VENV -eq 0 ]]; then
        if [[ ! -d "${VENV_DIR}" ]]; then
            info "Creating venv with ${PYTHON}..."
            "${PYTHON}" -m venv "${VENV_DIR}"
        else
            warn "Venv already exists — reusing (delete ${VENV_DIR} to rebuild from scratch)"
        fi
    else
        info "Skipping venv creation (--skip-venv)"
        [[ -d "${VENV_DIR}" ]] || die "Venv not found at ${VENV_DIR}"
    fi

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    # ── Step 2: Install PyTorch 2.9.1+cu130 ─────────────────────────────────
    # IMPORTANT: Do NOT use torch 2.10 — c10_cuda_check_implementation signature
    # changed between 2.9 and 2.10 (int → unsigned int), causing ABI mismatch with
    # sgl_kernel 0.3.x. torch 2.9.1 officially supports SM up to 12.0; GB10 (SM
    # 12.1) works but emits a warning at runtime — this is expected and harmless.
    if [[ $SKIP_TORCH -eq 0 ]]; then
        info "Installing PyTorch 2.9.1+cu130..."
        pip install -q --upgrade pip
        pip install \
            torch==2.9.1+cu130 \
            torchvision==0.24.1 \
            torchaudio==2.9.1 \
            --index-url https://download.pytorch.org/whl/cu130
    else
        info "Skipping torch install (--skip-torch)"
    fi

    # ninja is required for JIT compilation in flashinfer
    pip install -q ninja

    # ── Step 3: Install sglang + flashinfer ─────────────────────────────────
    if [[ $SKIP_SGLANG -eq 0 ]]; then
        info "Installing sglang[all] in editable mode..."
        pushd "${SGLANG_DIR}" > /dev/null
        pip install -e "python[all]" 2>&1 | tee "${LOG}"
        popd > /dev/null

        # Upgrade triton to 3.6.0: sglang[all] installs 3.5.1 which lacks ptxas-blackwell,
        # causing "sm_121a is not defined" errors on GB10 during Triton JIT kernel compilation.
        # triton 3.6.0 ships ptxas-blackwell (CUDA 12.9) that handles SM 100+/121a correctly.
        info "Upgrading triton to 3.6.0 (SM121a ptxas support)..."
        pip install triton==3.6.0
    else
        info "Skipping sglang install (--skip-sglang)"
    fi

    # ── Step 4: Install sgl-kernel cu130 wheel ──────────────────────────────
    # The PyPI sgl-kernel wheel (installed as a sglang[all] dependency) lacks SM_121a.
    # The cu130-tagged wheel from the sgl-project GitHub releases includes SM_121a.
    if [[ $SKIP_SGL_KERNEL -eq 0 ]]; then
        info "Installing sgl-kernel cu130 wheel (SM_121a)..."
        local SGL_KERNEL_WHL="https://github.com/sgl-project/whl/releases/download/v0.3.21/sgl_kernel-0.3.21+cu130-cp310-abi3-manylinux2014_aarch64.whl"
        pip install --force-reinstall --no-deps "${SGL_KERNEL_WHL}"
    else
        info "Skipping sgl-kernel install (--skip-sgl-kernel)"
    fi

    # ── Step 5: Apply GB10 fixes to flashinfer site-packages ────────────────
    # Fix 9d: Register FP4 JIT kernels as torch.library custom ops with fake
    #         implementations so AOTAutograd can infer shapes without calling
    #         TVM FFI. Required for --enable-piecewise-cuda-graph.
    # Fix 10: Use shutil.which("ninja") + abspath for correct ninja path
    #         resolution. Required if ninja is in PATH but not /usr/local/bin.
    if [[ $SKIP_FIXES -eq 0 ]]; then
        info "Applying GB10 fixes to flashinfer..."

        local SITE_PKG
        SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
        local FP4_FILE="${SITE_PKG}/flashinfer/fp4_quantization.py"
        local CPP_EXT_FILE="${SITE_PKG}/flashinfer/jit/cpp_ext.py"

        # Fix 9d: fp4_quantization.py — register torch.library custom ops
        if [[ -f "${FP4_FILE}" ]]; then
            if grep -q "_ensure_fp4_fns_cached" "${FP4_FILE}"; then
                info "Fix 9d: already applied — skipping"
            else
                info "Fix 9d: applying to ${FP4_FILE}"
                python3 - "${FP4_FILE}" <<'PYEOF'
import sys, re

path = sys.argv[1]
with open(path) as f:
    src = f.read()

marker = "\ndef _compute_swizzled_layout_sf_size("
if marker not in src:
    print(f"ERROR: marker not found in {path}", file=sys.stderr)
    sys.exit(1)

if "_ensure_fp4_fns_cached" in src:
    print("Already patched, skipping")
    sys.exit(0)

INJECTION = '''
# Module-level caches for torch.library custom-op wrappers. Populated eagerly
# by _ensure_fp4_fns_cached() called from piecewise_cuda_graph_runner.py
# before torch.compile starts. (Fix 9d — GB10 SM 12.1 compatibility)
_fp4_quantize_fn = None
_block_scale_interleave_fn = None


def _ensure_fp4_fns_cached(arch_key: str) -> None:
    """Register FP4 JIT kernels as torch.library custom ops with fake impls.

    Must be called BEFORE torch.compile / piecewise CUDA graph warmup.
    Creates arch-specific torch custom ops:
      flashinfer_fp4_rt::fp4_quantize_<arch_key>
      flashinfer_fp4_rt::block_scale_interleave_<arch_key>
    Each op has a register_fake impl so AOTAutograd can infer shapes via
    FakeTensors without calling the TVM FFI kernel.
    """
    global _fp4_quantize_fn, _block_scale_interleave_fn
    if _fp4_quantize_fn is not None:
        return  # Already registered.

    jit_mod = get_fp4_quantization_module(arch_key)

    # fp4_quantize custom op
    _fp4_q_name = f"flashinfer_fp4_rt::fp4_quantize_{arch_key}"

    @torch.library.custom_op(_fp4_q_name, mutates_args=())
    def _fp4_quantize_op(
        input: torch.Tensor,
        global_scale,
        sf_vec_size: int,
        sf_use_ue8m0: bool,
        is_sf_swizzled_layout: bool,
        is_sf_8x4_layout: bool,
        enable_pdl: bool,
    ):
        return jit_mod.fp4_quantize_sm100(
            input, global_scale, sf_vec_size, sf_use_ue8m0,
            is_sf_swizzled_layout, is_sf_8x4_layout, enable_pdl,
        )

    @torch.library.register_fake(_fp4_q_name)
    def _fp4_quantize_fake(
        input: torch.Tensor,
        global_scale,
        sf_vec_size: int,
        sf_use_ue8m0: bool,
        is_sf_swizzled_layout: bool,
        is_sf_8x4_layout: bool,
        enable_pdl: bool,
    ):
        out_val = input.new_empty(
            (*input.shape[:-1], input.shape[-1] // 2), dtype=torch.uint8
        )
        m = input.numel() // input.shape[-1]
        k = input.shape[-1]
        if is_sf_swizzled_layout:
            row_size = 8 if is_sf_8x4_layout else 128
            padded_row = (m + row_size - 1) // row_size * row_size
            padded_col = (k // sf_vec_size + 3) // 4 * 4
            out_sf_size = padded_row * padded_col
        else:
            out_sf_size = m * k // sf_vec_size
        out_sf = input.new_empty((out_sf_size,), dtype=torch.uint8)
        return out_val, out_sf

    # block_scale_interleave custom op
    _bsi_name = f"flashinfer_fp4_rt::block_scale_interleave_{arch_key}"

    @torch.library.custom_op(_bsi_name, mutates_args=())
    def _block_scale_interleave_op(unswizzled_sf: torch.Tensor) -> torch.Tensor:
        return jit_mod.block_scale_interleave_sm100(unswizzled_sf)

    @torch.library.register_fake(_bsi_name)
    def _block_scale_interleave_fake(unswizzled_sf: torch.Tensor) -> torch.Tensor:
        num_experts = unswizzled_sf.shape[0] if unswizzled_sf.dim() == 3 else 1
        padded_row = (unswizzled_sf.shape[-2] + 127) // 128 * 128
        padded_col = (unswizzled_sf.shape[-1] + 3) // 4 * 4
        return unswizzled_sf.new_empty(
            (num_experts * padded_row * padded_col,), dtype=unswizzled_sf.dtype
        )

    _fp4_quantize_fn = _fp4_quantize_op
    _block_scale_interleave_fn = _block_scale_interleave_op

'''

src = src.replace(marker, INJECTION + marker)
with open(path, 'w') as f:
    f.write(src)
print(f"Patched {path}")
PYEOF
                info "Fix 9d: applied"
            fi
        else
            info "Fix 9d: ${FP4_FILE} not found — skipping"
        fi

        # Fix 10: cpp_ext.py — use shutil.which + abspath for ninja path
        if [[ -f "${CPP_EXT_FILE}" ]]; then
            if grep -q "shutil.which.*ninja" "${CPP_EXT_FILE}"; then
                info "Fix 10: already applied — skipping"
            else
                info "Fix 10: applying to ${CPP_EXT_FILE}"
                python3 - "${CPP_EXT_FILE}" "${VENV_DIR}" <<'PYEOF'
import sys

path = sys.argv[1]
venv = sys.argv[2]

with open(path) as f:
    src = f.read()

old = '    command = [\n        "ninja",'
new = f'''    import shutil, os
    ninja_exe = shutil.which("ninja")
    ninja_exe = os.path.abspath(ninja_exe) if ninja_exe else "{venv}/bin/ninja"
    command = [
        ninja_exe,'''

if old not in src:
    print(f"ERROR: marker not found in {path}", file=sys.stderr)
    sys.exit(1)
if "shutil.which" in src:
    print("Already patched, skipping")
    sys.exit(0)

src = src.replace(old, new)
with open(path, 'w') as f:
    f.write(src)
print(f"Patched {path}")
PYEOF
                info "Fix 10: applied"
            fi
        else
            info "Fix 10: ${CPP_EXT_FILE} not found — skipping"
        fi
    else
        info "Skipping flashinfer fixes (--skip-fixes)"
    fi

    # ── Step 6: Verify installation ─────────────────────────────────────────
    info "Verifying installation..."

    python -c "
import torch
print(f'  torch              : {torch.__version__}')
print(f'  torch.version.cuda : {torch.version.cuda or \"None\"}')
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f'  GPU                : {torch.cuda.get_device_name(0)} (SM {cap[0]}.{cap[1]})')
"
    python -c "import sgl_kernel; print('  sgl-kernel         : OK')" 2>/dev/null || \
        warn "sgl-kernel import failed"
    python -c "import sglang; print(f'  sglang             : {sglang.__version__}')" 2>/dev/null || \
        warn "sglang import failed"
    python -c "import flashinfer; print(f'  flashinfer         : {flashinfer.__version__}')" 2>/dev/null || \
        warn "flashinfer import failed"
    python -c "import triton; print(f'  triton             : {triton.__version__}')" 2>/dev/null || \
        warn "triton import failed"

    echo ""
    success "Build complete. Log: ${LOG}"
}

cmd_launch() {
    [[ -d "${VENV_DIR}" ]] || die "Venv not found at ${VENV_DIR}. Run: ./sglang.sh build"

    # Kill any existing SGLang server and wait for GPU memory to drain.
    # Launching while the previous server still holds VRAM risks GPU OOM
    # which can cascade into a full system hang on GB10 (unified memory).
    local existing
    existing=$(pgrep -f "python.*sglang" 2>/dev/null || true)
    if [[ -n "${existing}" ]]; then
        info "Stopping existing SGLang processes (server + workers) (PIDs: ${existing})..."
        # shellcheck disable=SC2086
        pkill -TERM -f "python.*sglang" 2>/dev/null || true
        local waited=0
        while pgrep -f "python.*sglang" > /dev/null 2>&1; do
            sleep 1
            (( waited++ )) || true
            if (( waited >= 30 )); then
                info "  Sending SIGKILL after ${waited}s..."
                pkill -9 -f "python.*sglang" 2>/dev/null || true
                break
            fi
        done
        # Give the NVIDIA driver time to reclaim GPU memory
        info "  Waiting 5s for GPU memory to be released..."
        sleep 5
    fi

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    setup_runtime_env

    info "Launching SGLang OpenAI-compatible server"
    info "  SAFETENSORS_FAST_GPU           = ${SAFETENSORS_FAST_GPU}"
    info "  SGLANG_QUANTIZE_LM_HEAD_FP8   = ${SGLANG_QUANTIZE_LM_HEAD_FP8}"
    info "  SGLANG_MTP_FP8                = ${SGLANG_MTP_FP8}"
    info "  KV_CACHE_DTYPE                = ${KV_CACHE_DTYPE}"
    echo ""

    # Pass KV_CACHE_DTYPE as --kv-cache-dtype arg (resolved after SM121 override).
    # Presets should NOT pass --kv-cache-dtype themselves.
    exec python -m sglang.launch_server --kv-cache-dtype "${KV_CACHE_DTYPE}" "$@"
}

cmd_qwen35_nvfp4() {
    local model="${QWEN35_MODEL:-Sehyo/Qwen3.5-122B-A10B-NVFP4}"

    local spec_args=()
    if [[ "${DISABLE_MTP:-}" != "1" ]]; then
        spec_args=(
            --speculative-algorithm NEXTN
            --speculative-num-steps 4
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens 5
            --mamba-scheduler-strategy extra_buffer
        )
        export SGLANG_ENABLE_SPEC_V2=1
        info "Preset: Qwen3.5-122B-A10B-NVFP4 (compressed-tensors, speculative NEXTN)"
    else
        info "Preset: Qwen3.5-122B-A10B-NVFP4 (compressed-tensors, MTP DISABLED)"
    fi
    info "  Model : ${model}"
    info "  CtxLen: ${CONTEXT_LENGTH}"

    cmd_launch \
        --model-path "${model}" \
        --quantization compressed-tensors \
        --speculative-draft-model-quantization compressed-tensors \
        --mem-fraction-static 0.87 \
        --context-length "${CONTEXT_LENGTH}" \
        --max-running-requests 3 \
        --attention-backend flashinfer \
        --linear-attn-backend triton \
        --linear-attn-prefill-backend triton \
        --chunked-prefill-size 16384 \
        --mamba-full-memory-ratio auto \
        --mamba-ssm-dtype bfloat16 \
        --disable-piecewise-cuda-graph \
        --disable-multimodal \
        "${spec_args[@]}" \
        --reasoning-parser qwen3 \
        --trust-remote-code \
        "${SERVER_ARGS[@]}" \
        "${QWEN3_ARGS[@]}" \
        --served-model-name qwen35 \
        "$@"
}

cmd_qwen35_35b_nvfp4() {
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
        info "Preset: Qwen3.5-35B-A3B-NVFP4 (compressed-tensors, speculative NEXTN)"
    else
        info "Preset: Qwen3.5-35B-A3B-NVFP4 (compressed-tensors, MTP DISABLED)"
    fi
    info "  Model : ${model}"
    info "  CtxLen: ${CONTEXT_LENGTH}"

    cmd_launch \
        --model-path "${model}" \
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
        --trust-remote-code \
        "${SERVER_ARGS[@]}" \
        "${QWEN3_ARGS[@]}" \
        "$@"
}

cmd_qwen3_coder_next_nvfp4() {
    local model="${QWEN3_CODER_NVFP4_MODEL:-GadflyII/Qwen3-Coder-Next-NVFP4}"
    local ctx="${CONTEXT_LENGTH:-131072}"

    info "Preset: Qwen3-Coder-Next-NVFP4 (GadflyII, compressed-tensors)"
    info "  Model : ${model}"
    info "  CtxLen: ${ctx}"

    cmd_launch \
        --model-path "${model}" \
        --quantization compressed-tensors \
        --mem-fraction-static 0.85 \
        --context-length "${ctx}" \
        --attention-backend triton \
        --reasoning-parser qwen3 \
        --trust-remote-code \
        "${SERVER_ARGS[@]}" \
        "${QWEN3_ARGS[@]}" \
        "$@"
}

cmd_qwen3_coder_next_fp8() {
    local model="${QWEN3_CODER_MODEL:-Qwen/Qwen3-Coder-Next-FP8}"
    local ctx="${CONTEXT_LENGTH:-131072}"

    info "Preset: Qwen3-Coder-Next-FP8 (dense FP8, chunked-prefill)"
    info "  Model : ${model}"
    info "  CtxLen: ${ctx}"

    cmd_launch \
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
        "${SERVER_ARGS[@]}" \
        "$@"
}

cmd_minimax_m27() {
    # MiniMax M2.7-REAP 172B NVFP4-GB10 (compressed-tensors format).
    # Produced by https://github.com/scottgl9/sglang-spark-gb10-optimizations/
    # tree/main/minimax_reap_reduction (convert.py) from catplusplus/MiniMax-M2.7-REAP-172B-A10B-NVFP4.
    #
    # Measured on Spark GB10 (SM121), batch=1, with CUDA graphs enabled:
    #   4K  decode ≈ 16.9 tok/s
    #   32K decode ≈ 12.8 tok/s
    #
    # The previous `--disable-cuda-graph` flag was attributed to a Marlin FP4
    # `torch.compile` graph break (see model_config.py piecewise-disabled list),
    # but the actual capture blocker on our stack was the MoE NaN-fallback in
    # python/sglang/srt/models/minimax_m2.py — `hidden_states.isnan().any()`
    # forces a D2H sync during `cudaStreamCapture` → cudaErrorStreamCaptureUnsupported.
    # Replacing the conditional with an unconditional `torch.where` lets capture
    # succeed. Piecewise CUDA graphs stay disabled per model_config.py.
    local model="${MINIMAX_MODEL:-scottgl/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10}"
    local ctx="${CONTEXT_LENGTH:-65536}"

    info "Preset: MiniMax M2.7 REAP 172B NVFP4-GB10 (compressed-tensors)"
    info "  Model : ${model}"
    info "  CtxLen: ${ctx}"
    info "  KV    : ${KV_CACHE_DTYPE}  (FP8 KV cache — halves KV bandwidth)"
    info "  Note  : CUDA graphs enabled (regular only; piecewise disabled in model_config)"

    cmd_launch \
        --model-path "${model}" \
        --served-model-name minimax27 \
        --quantization compressed-tensors \
        --mem-fraction-static 0.90 \
        --max-running-requests 2 \
        --context-length "${ctx}" \
        --chunked-prefill-size 16384 \
        --attention-backend triton \
        --reasoning-parser minimax \
        --tool-call-parser minimax-m2 \
        --trust-remote-code \
        --disable-piecewise-cuda-graph \
        "${SERVER_ARGS[@]}" \
        "$@"
}

cmd_nemotron() {
    # Default to the locally cached snapshot; override with NEMOTRON_MODEL=<path>
    local _snap="/home/scottgl/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/snapshots/bd90f177c8c69d8f3969c61e7e8f1afaba57ae61"
    local model="${NEMOTRON_MODEL:-${_snap}}"

    # MTP (NEXTN) — model has num_nextn_predict_layers=1, so 1 draft step is the max.
    # Disable with: DISABLE_MTP=1 ./sglang.sh nemotron
    local spec_args=()
    if [[ "${DISABLE_MTP:-}" != "1" ]]; then
        spec_args=(
            --speculative-algorithm NEXTN
            --speculative-num-steps 1
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens 1
        )
        export SGLANG_ENABLE_SPEC_V2=1
        info "Preset: Nemotron-3-Super-120B-A12B-NVFP4 (modelopt_fp4, MTP NEXTN 1-step, FP8 post-quant)"
    else
        info "Preset: Nemotron-3-Super-120B-A12B-NVFP4 (modelopt_fp4, MTP DISABLED, FP8 post-quant)"
    fi
    export SGLANG_NEMOTRON_FP8_POST_QUANT=1
    info "  Model : ${model}"
    info "  CtxLen: ${CONTEXT_LENGTH}"

    cmd_launch \
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
        "${SERVER_ARGS[@]}" \
        "$@"
}

cmd_mistral_small4() {
    local _snap="/home/scottgl/.cache/huggingface/hub/models--mistralai--Mistral-Small-4-119B-2603-NVFP4/snapshots/043f75a201a226d8e9cbbc3316af437ea25d3912"
    local _eagle="/home/scottgl/.cache/huggingface/hub/models--mistralai--Mistral-Small-4-119B-2603-eagle/snapshots/3ff299733b3dcb701617a22add5ce796304f7f05"
    local model="${MISTRAL_MODEL:-${_snap}}"
    local eagle="${MISTRAL_EAGLE_MODEL:-${_eagle}}"

    # EAGLE is disabled by default: the EAGLE draft model was trained on the
    # unquantized base model. NVFP4 quantization changes output distributions
    # enough that EAGLE predictions diverge (~6% accept rate = no speedup).
    # Enable with: ENABLE_EAGLE=1 ./sglang.sh mistral-small-4
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
        info "Preset: Mistral-Small-4-119B NVFP4 + EAGLE (experimental)"
    else
        info "Preset: Mistral-Small-4-119B NVFP4"
    fi
    info "  Model : ${model}"
    info "  Eagle : ${eagle}"
    info "  CtxLen: ${CONTEXT_LENGTH}"

    cmd_launch \
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
        "${SERVER_ARGS[@]}" \
        "$@"
}

cmd_shell() {
    [[ -d "${VENV_DIR}" ]] || die "Venv not found at ${VENV_DIR}. Run: ./sglang.sh build"
    info "Activating SGLang venv — type 'deactivate' to exit"
    setup_runtime_env
    exec bash --rcfile <(echo "source '${VENV_DIR}/bin/activate'; PS1='(sglang-gb10) \u@\h:\w\$ '")
}

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [args]

Commands:
  build [--skip-*]               Build SGLang into .sglang/ venv
  launch [sglang args]           Start the OpenAI-compatible API server
  shell                          Drop into an activated venv shell

  Qwen3.5-NVFP4 [args]          Qwen3.5-122B MoE NVFP4, speculative decoding
  Qwen3.5-35B-NVFP4 [args]      Sehyo/Qwen3.5-35B-A3B-NVFP4, speculative decoding
  Qwen3-Coder-Next-NVFP4 [args] GadflyII/Qwen3-Coder-Next-NVFP4
  Qwen3-Coder-Next-FP8 [args]   Qwen/Qwen3-Coder-Next-FP8
  minimax-m27 [args]             MiniMax M2.7 REAP 172B NVFP4-GB10 (compressed-tensors)
  nemotron [args]                NVIDIA Nemotron-3-Super 120B-A12B NVFP4 + MTP
  mistral-small-4 [args]        Mistral-Small-4-119B NVFP4 + EAGLE

Build options:
  --skip-venv       Skip venv creation
  --skip-torch      Skip torch installation
  --skip-sglang     Skip sglang[all] installation
  --skip-sgl-kernel Skip sgl-kernel wheel installation
  --skip-fixes      Skip flashinfer GB10 patches

Context window (default: ${CONTEXT_LENGTH}):
  CONTEXT_LENGTH=32768 ./sglang.sh Qwen3.5-NVFP4

Model path overrides:
  QWEN35_MODEL=<path>              Override Qwen3.5-NVFP4 model path
  QWEN35_35B_MODEL=<path>          Override Qwen3.5-35B-NVFP4 model path
  QWEN3_CODER_NVFP4_MODEL=<path>  Override Qwen3-Coder-Next-NVFP4 model
  QWEN3_CODER_MODEL=<path>        Override Qwen3-Coder-Next-FP8 model
  MINIMAX_MODEL=<path>             Override MiniMax model path
  NEMOTRON_MODEL=<path>            Override Nemotron model path
  MISTRAL_MODEL=<path>             Override Mistral-Small-4 model path
  MISTRAL_EAGLE_MODEL=<path>       Override Mistral-Small-4 EAGLE draft path

Environment overrides:
  CONTEXT_LENGTH=N               Context window tokens (default: 65536)
  DISABLE_MTP=1                  Disable speculative decoding (Qwen3.5-NVFP4)
  ENABLE_EAGLE=1                 Enable EAGLE speculative decoding (mistral-small-4, experimental)
  SGLANG_QUANTIZE_LM_HEAD_FP8=0 Disable FP8 lm_head quantization (default: 1)
  SGLANG_QUANTIZE_EMBED_FP8=0   Disable FP8 embed_tokens quantization (default: 1)
  MINIMAX_MODEL=<path>           Override MiniMax model path

Examples:
  ./sglang.sh build
  ./sglang.sh Qwen3.5-NVFP4
  CONTEXT_LENGTH=32768 ./sglang.sh Qwen3.5-NVFP4
  ./sglang.sh minimax-m27                        # MiniMax M2.7 (compressed-tensors NVFP4)
  CONTEXT_LENGTH=4096 ./sglang.sh minimax-m27    # tight-memory mode for 128 GB hosts

EOF
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
CMD="${1:-}"
shift || true

case "${CMD}" in
    build)   cmd_build "$@" ;;
    launch)  cmd_launch "$@" ;;
    shell)   cmd_shell ;;
    Qwen3.5-NVFP4|qwen3.5-nvfp4|qwen35-nvfp4) cmd_qwen35_nvfp4 "$@" ;;
    Qwen3.5-35B-NVFP4|qwen3.5-35b-nvfp4|qwen35-35b-nvfp4) cmd_qwen35_35b_nvfp4 "$@" ;;
    Qwen3-Coder-Next-NVFP4|qwen3-coder-next-nvfp4) cmd_qwen3_coder_next_nvfp4 "$@" ;;
    Qwen3-Coder-Next-FP8|qwen3-coder-next-fp8) cmd_qwen3_coder_next_fp8 "$@" ;;
    minimax-m27|MiniMax-M27|minimax-m2.7|MiniMax-M2.7) cmd_minimax_m27 "$@" ;;
    nemotron|Nemotron|nemotron-3-super|Nemotron-3-Super) cmd_nemotron "$@" ;;
    mistral-small-4|Mistral-Small-4|mistral-small4) cmd_mistral_small4 "$@" ;;
    ""|help|-h|--help) usage ;;
    *) die "Unknown command: ${CMD}. Run './sglang.sh help' for usage." ;;
esac
