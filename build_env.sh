#!/bin/bash
# =============================================================================
# build_env.sh — Full SGLang build script for MiniMax-M2.5 on DGX Spark (GB10)
#
# Target:  NVIDIA GB10 (SM 12.1), CUDA 13.0, aarch64, Python 3.12
# Purpose: Build SGLang + sgl-kernel from source with SM_121a support,
#          install dependencies, and apply GB10 compatibility fixes.
#
# Usage:
#   cd ~/sandbox/sglang
#   bash build_env.sh [--skip-venv] [--skip-torch] [--skip-sglang] [--skip-sgl-kernel]
#
# Options:
#   --skip-venv       Skip venv creation (if already exists)
#   --skip-torch      Skip torch installation (if already installed)
#   --skip-sglang     Skip sglang/flashinfer installation (if already installed)
#   --skip-sgl-kernel Skip sgl-kernel build (if already built)
#   --skip-fixes      Skip applying GB10 fixes to flashinfer site-packages
#
# On success: source .sglang/bin/activate && python run_minimax.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.sglang"
LOG="$SCRIPT_DIR/build_env.log"

# Parse args
SKIP_VENV=0; SKIP_TORCH=0; SKIP_SGLANG=0; SKIP_SGL_KERNEL=0; SKIP_FIXES=0
for arg in "$@"; do
    case $arg in
        --skip-venv)       SKIP_VENV=1 ;;
        --skip-torch)      SKIP_TORCH=1 ;;
        --skip-sglang)     SKIP_SGLANG=1 ;;
        --skip-sgl-kernel) SKIP_SGL_KERNEL=1 ;;
        --skip-fixes)      SKIP_FIXES=1 ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
die() { echo "ERROR: $*" >&2; exit 1; }

exec > >(tee -a "$LOG") 2>&1
log "=== SGLang GB10 build started ==="
log "Script dir: $SCRIPT_DIR"
log "Log: $LOG"

# Verify we're in the right directory
[[ -f "$SCRIPT_DIR/python/sglang/__init__.py" ]] || die "Run from ~/sandbox/sglang (sglang source root)"

# =============================================================================
# Step 1: Create virtual environment
# =============================================================================
if [[ $SKIP_VENV -eq 0 ]]; then
    log "--- Step 1: Creating venv at $VENV ---"
    if [[ -d "$VENV" ]]; then
        log "WARNING: venv already exists. Removing and recreating..."
        rm -rf "$VENV"
    fi
    /usr/bin/python3.12 -m venv "$VENV"
    log "venv created: $VENV"
else
    log "--- Step 1: Skipping venv creation (--skip-venv) ---"
    [[ -d "$VENV" ]] || die "Venv not found at $VENV"
fi

source "$VENV/bin/activate"
log "Python: $(which python) $(python --version)"

# =============================================================================
# Step 2: Install PyTorch 2.9.1+cu130 (must match sgl-kernel ABI)
# =============================================================================
# IMPORTANT: Do NOT use torch 2.10 — c10_cuda_check_implementation signature
# changed between 2.9 and 2.10 (int → unsigned int), causing ABI mismatch with
# sgl_kernel 0.3.x. torch 2.9.1 officially supports SM up to 12.0; GB10 (SM
# 12.1) works but emits a warning at runtime — this is expected and harmless.
if [[ $SKIP_TORCH -eq 0 ]]; then
    log "--- Step 2: Installing PyTorch 2.9.1+cu130 ---"
    pip install \
        torch==2.9.1+cu130 \
        torchvision==0.24.1 \
        torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu130
    log "PyTorch installed: $(python -c 'import torch; print(torch.__version__)')"
else
    log "--- Step 2: Skipping torch install (--skip-torch) ---"
fi

# Install ninja (required for JIT compilation in flashinfer)
pip install ninja
log "ninja: $(which ninja)"

# =============================================================================
# Step 3: Install sglang + flashinfer (editable install for sglang source)
# =============================================================================
if [[ $SKIP_SGLANG -eq 0 ]]; then
    log "--- Step 3: Installing sglang[all] in editable mode ---"
    cd "$SCRIPT_DIR"
    pip install -e "python[all]"
    # Upgrade triton to 3.6.0: sglang[all] installs 3.5.1 which lacks ptxas-blackwell,
    # causing "sm_121a is not defined" errors on GB10 during Triton JIT kernel compilation.
    # triton 3.6.0 ships ptxas-blackwell (CUDA 12.9) that handles SM 100+/121a correctly.
    # torch 2.9.1 pins triton==3.5.1 but the combination with 3.6.0 is known-good on GB10.
    pip install triton==3.6.0
    log "sglang installed"
    log "triton: $(python -c 'import triton; print(triton.__version__)' 2>/dev/null || echo 'not found')"
    log "flashinfer: $(python -c 'import flashinfer; print(flashinfer.__version__)' 2>/dev/null || echo 'not found')"
else
    log "--- Step 3: Skipping sglang install (--skip-sglang) ---"
fi

# =============================================================================
# Step 4: Install sglang-kernel cu130 wheel with SM_121a support
# =============================================================================
# The PyPI sglang-kernel wheel links against libnvrtc.so.12 (CUDA 12) and fails
# on GB10 (CUDA 13). The cu130-tagged wheel from sgl-project GitHub releases
# links against libnvrtc.so.13 and includes SM_121a support.
#
# NOTE: Source build (pip install -e sgl-kernel/) was attempted but OOMs due to
# compiling 7 architectures × 32 parallel nvcc threads on GB10's limited memory.
# The pre-built cu130 wheel is identical in functionality and includes SM_121a.
if [[ $SKIP_SGL_KERNEL -eq 0 ]]; then
    log "--- Step 4: Installing sglang-kernel cu130 wheel (SM_121a) ---"
    SGLANG_KERNEL_WHL="https://github.com/sgl-project/whl/releases/download/v0.4.1.post1/sglang_kernel-0.4.1.post1%2Bcu130-cp310-abi3-manylinux2014_aarch64.whl"
    pip install --force-reinstall --no-deps "$SGLANG_KERNEL_WHL"
    log "sglang-kernel installed: $(python -c 'import sgl_kernel; print("OK")')"
else
    log "--- Step 4: Skipping sglang-kernel build (--skip-sgl-kernel) ---"
fi

cd "$SCRIPT_DIR"

# =============================================================================
# Step 5: Apply GB10 fixes to flashinfer site-packages
# =============================================================================
# These fixes patch flashinfer (installed in site-packages, not source tree):
#   Fix 9d: Register FP4 JIT kernels as torch.library custom ops with fake
#           implementations so AOTAutograd can infer shapes without calling
#           TVM FFI. Required for --enable-piecewise-cuda-graph.
#   Fix 10: Use shutil.which("ninja") + abspath for correct ninja path
#           resolution. Required if ninja is in PATH but not /usr/local/bin.
#
# These are only needed when using --enable-piecewise-cuda-graph. The default
# run script (run_minimax.sh) does NOT use piecewise CUDA graphs, so these
# fixes are applied defensively / for future use.

if [[ $SKIP_FIXES -eq 0 ]]; then
    log "--- Step 5: Applying GB10 fixes to flashinfer ---"

    SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
    log "site-packages: $SITE_PKG"

    FP4_FILE="$SITE_PKG/flashinfer/quantization/fp4_quantization.py"
    # Fallback: older flashinfer versions kept the file at the top-level path
    [[ -f "$FP4_FILE" ]] || FP4_FILE="$SITE_PKG/flashinfer/fp4_quantization.py"
    CPP_EXT_FILE="$SITE_PKG/flashinfer/jit/cpp_ext.py"

    # Fix 9d: fp4_quantization.py — register torch.library custom ops
    if [[ -f "$FP4_FILE" ]]; then
        if grep -q "_ensure_fp4_fns_cached" "$FP4_FILE"; then
            log "Fix 9d: already applied to $FP4_FILE — skipping"
        else
            log "Fix 9d: applying to $FP4_FILE"
            python3 - "$FP4_FILE" <<'PYEOF'
import sys, re

path = sys.argv[1]
with open(path) as f:
    src = f.read()

# Find the insertion point: after the imports block (right before _compute_swizzled_layout_sf_size)
marker = "\ndef _compute_swizzled_layout_sf_size("
if marker not in src:
    print(f"ERROR: marker not found in {path}", file=sys.stderr)
    sys.exit(1)

# Check it's not already patched
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
            log "Fix 9d: applied"
        fi
    else
        log "Fix 9d: $FP4_FILE not found — skipping"
    fi

    # Fix 10: cpp_ext.py — use shutil.which + abspath for ninja path
    if [[ -f "$CPP_EXT_FILE" ]]; then
        if grep -q "shutil.which.*ninja" "$CPP_EXT_FILE"; then
            log "Fix 10: already applied to $CPP_EXT_FILE — skipping"
        else
            log "Fix 10: applying to $CPP_EXT_FILE"
            python3 - "$CPP_EXT_FILE" "$VENV" <<'PYEOF'
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
            log "Fix 10: applied"
        fi
    else
        log "Fix 10: $CPP_EXT_FILE not found — skipping"
    fi
else
    log "--- Step 5: Skipping flashinfer fixes (--skip-fixes) ---"
fi

# =============================================================================
# Step 6: Verify installation
# =============================================================================
log "--- Step 6: Verifying installation ---"

python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f'GPU compute capability: SM {cap[0]}.{cap[1]}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

python -c "import sgl_kernel; print('sgl-kernel: OK')" 2>/dev/null || \
    log "WARNING: sgl-kernel import failed"

python -c "import sglang; print(f'sglang: {sglang.__version__}')" 2>/dev/null || \
    log "WARNING: sglang import failed"

python -c "import flashinfer; print(f'flashinfer: {flashinfer.__version__}')" 2>/dev/null || \
    log "WARNING: flashinfer import failed"

python -c "import ninja; print('ninja python pkg: OK')" 2>/dev/null || true
which ninja && log "ninja binary: $(which ninja)" || log "WARNING: ninja not in PATH"

log "=== Build complete! ==="
log ""
log "To run MiniMax-M2.5:"
log "  cd ~/sandbox/sglang"
log "  bash run_minimax.sh"
log ""
log "To activate the venv manually:"
log "  source ~/.sglang/bin/activate   # or:"
log "  source ~/sandbox/sglang/.sglang/bin/activate"
