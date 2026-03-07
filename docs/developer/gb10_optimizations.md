# GB10 (SM121) Performance Optimizations & NVFP4 KV Cache Support

This document describes the performance optimizations on the `gb10-optimization` branch
targeting the NVIDIA GB10 GPU (DGX Spark, compute capability 12.1).

## Hardware Context

The GB10 is a bandwidth-limited mobile Blackwell GPU:

| Spec | GB10 | B200 (comparison) |
|---|---|---|
| Memory | 128 GB LPDDR5X unified | 192 GB HBM3e |
| Bandwidth | 273 GB/s | 8 TB/s |
| Shared Memory | 99 KB/SM | 228 KB/SM |
| FP4 Tensor Cores | Yes (SM121) | Yes (SM100) |
| PTX `cvt.e2m1x2` | No | Yes |

All optimizations focus on reducing memory bandwidth consumption, since GB10 has
29x less bandwidth than a B200.

---

## P0: Critical Fixes (Unblocks FP4 on SM121)

### 1. Software E2M1 Conversion for SM121

**File:** `python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_quant.cuh`

SM121 lacks the `cvt.rn.satfinite.e2m1x2.f32` PTX instruction that SM100+ provides.
The original code called `__trap()` on unsupported architectures, crashing the GPU.

Added `_sw_float_to_e2m1()` — a software bit-manipulation fallback using 6 threshold
comparisons for IEEE 754 round-to-nearest-even E2M1 encoding. Guarded by
`#if __CUDA_ARCH__ == 1210`. Both `fp32_vec_to_e2m1()` overloads (float[8] and
float2[4]) use the SW path on SM121.

**Impact:** Core enabler. Without this, all runtime FP4 quantization crashes on GB10.

### 2. `nv_fp4_dummy.h` for CUDA 13.0 CCCL Compatibility

**File:** `sgl-kernel/csrc/nv_fp4_dummy.h`

CUDA 13.0 CCCL headers reference `__nv_fp4_e2m1` types not yet shipped. Provides
stub type definitions guarded with `#ifndef __nv_fp4_e2m1` (no-op when official
types arrive).

**Impact:** Fixes build failure on CUDA 13.0 + SM121.

---

## P1: High-Impact Optimizations

### 3. GB10 MoE Triton Config Tuning

**Files:**
- `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=512,...fp8_w8a8.json`
- `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`

Added tuned MoE kernel configs for GB10 with `BLOCK_SIZE_M=16` (vs default 128).
The default is far too large for decode on bandwidth-limited hardware. Added Triton
3.5.0/3.5.1 to the version fallback list.

**Impact:** 2-5x MoE decode speedup. vLLM benchmarks showed +65.7% from tuned configs.

### 4. FlashInfer SM121 Compatibility Patches

**Files:**
- `python/sglang/srt/utils/gb10_flashinfer_compat.py` (new)
- `python/sglang/srt/utils/common.py`

7 idempotent patches applied at startup when SM121 is detected:
1. `_patch_float_subbyte()` — Removes SM121A from PTX-dependent FP4 paths
2. `_patch_quantization_utils()` — Software E2M1 in FlashInfer's quantization
3. `_patch_arch_condition()` — Fixes architecture condition checks
4. `_copy_fp4_header()` — Provides FP4 header for JIT compilation
5. `_patch_trtllm_fused_moe_runtime_checks()` — Fixes MoE runtime SM checks
6. `_patch_trtllm_fused_moe_jit()` — Fixes MoE JIT compile flags
7. `_clear_moe_jit_cache()` — Invalidates stale JIT cache entries

Called from `set_cuda_arch()` in `common.py` when `capability == (12, 1)`.

**Impact:** Without these, FlashInfer FP4 KV cache and FP4 MoE kernels crash on GB10.

### 5. NVFP4 Global Scale Overflow Fix

**File:** `python/sglang/srt/layers/quantization/modelopt_quant.py`

Changed 6 `torch.empty()` calls to `torch.full(..., torch.finfo(torch.float32).min)`
for NVFP4 PerTensorScaleParameter initialization. With MergedColumnParallelLinear,
unloaded slots contain garbage that can overflow to +inf via `max()`.

**Impact:** Prevents NaN/inf output on tensor-parallel NVFP4 configs.

---

## Blackwell Attention & KV Cache

### 6. Blackwell Attention Backend Auto-Selection

**File:** `python/sglang/srt/layers/attention/attention_registry.py`

Added `flashinfer` as a supported backend for Blackwell GDN hybrid models.
Auto-selects FlashInfer linear attention backend on SM120+ for faster CUTLASS kernels.

### 7. NVFP4 KV Cache Auto-Detection

**Files:**
- `python/sglang/srt/layers/quantization/modelopt_quant.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`

Parses `{type: float, num_bits: 4}` kv_cache_scheme in both ModelOptFp8Config and
ModelOptFp4Config to produce `"NVFP4"` quant algo. Auto-detects NVFP4 in
`configure_kv_cache_dtype()` and sets `fp4_e2m1`. Auto-switches attention backend
when the current one is FP4-incompatible. Pre-warms KVFP4QuantizeUtil (`@torch.compile`)
before CUDA graph capture. Fixes FP4 scale overhead in `cell_size` for hybrid SWA models.

**Impact:** NVFP4 KV cache reduces KV read bandwidth by 50% vs FP8 (75% vs BF16).

### 8. FP8 KV Cache Auto-Selection on Blackwell

**File:** `python/sglang/srt/model_executor/model_runner.py`

Auto-selects `fp8_e4m3` KV cache on Blackwell+ (major >= 10) when the model doesn't
specify a KV quantization algo. Eliminates the need for manual `KV_CACHE_DTYPE=fp8_e4m3`
environment variable.

**Impact:** FP8 KV cache halves KV read bandwidth vs BF16 with no user configuration.

---

## FP4 Kernel Pre-warming

### 9. FP4 JIT Kernel Pre-warm Before CUDA Graph Capture

**File:** `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`

Pre-warms FlashInfer FP4 JIT kernels (scaled_fp4_quant, GEMM, MoE grouped GEMM)
before `torch.compile` to avoid tracing failures during CUDA graph capture. The JIT
compilation must complete before graph recording begins.

---

## Post-Quantization for Bandwidth Reduction

### 10. NVFP4 Post-Quantization for GDN Layers

**File:** `python/sglang/srt/layers/quantization/nvfp4_post_quant.py` (new)

Post-quantizes BF16 GDN attention projection layers (in_proj_qkv, in_proj_qkvz,
in_proj_z) to NVFP4 at load time on SM120+. Uses CUTLASS NVFP4 GEMM for inference.

**Models:** Qwen3-Next (`qwen3_next.py`), Qwen3.5 dense & MoE (`qwen3_5.py`)

**Impact:** Halves GDN layer bandwidth. These layers are called every token and
dominate decode time on bandwidth-limited hardware.

### 11. MTP FP8 Post-Quantization for Draft Models

**File:** `python/sglang/srt/layers/quantization/fp8_post_quant.py` (new)

Enabled by `SGLANG_MTP_FP8=1`. Post-quantizes MTP draft model layers from BF16
to FP8 after weight loading, before CUDA graph capture:
- **fc layers** (`nn.Linear`): Replaced with `FP8PostQuantLinear` using `torch._scaled_mm`
- **MoE expert weights** (w13/w2): Per-expert per-tensor quantized to `float8_e4m3fn`
- **lm_head**: Included in layer patterns for FP8 post-quantization

Especially impactful for Qwen3.5 MTP where all MTP layers are BF16 (the MTP module
nullifies quant_config for modelopt_fp4 checkpoints).

**Impact:** +20% from FP8 lm_head alone in vLLM benchmarks. Halves draft model bandwidth.

### 12. lm_head FP8 Quantization

**File:** `python/sglang/srt/layers/quantization/compressed_tensors.py`

`SGLANG_QUANTIZE_LM_HEAD_FP8=1` applies dynamic FP8 to the lm_head layer at load
time. Especially useful when the checkpoint stores lm_head in BF16 but the rest of
the model is quantized.

---

## Speculative Decoding

### 13. Auto-Enable Decode Mode for MTP on Blackwell

**File:** `python/sglang/srt/server_args.py`

Auto-sets `speculative_attention_mode="decode"` on SM100+/SM120+ with FlashInfer.
The default `"prefill"` mode is slower for MTP speculative decoding.

**Impact:** vLLM saw +28.7% (44.83 -> 57.71 tok/s) from this change.

---

## Triton Pipeline Depth Fix

### 14. MXFP8 `num_stages` Fix for SM120

**File:** `python/sglang/srt/layers/quantization/fp8_utils.py`

Changed `num_stages=1` to `num_stages=2` on SM120. The original value of 1 disabled
memory-compute pipelining entirely. GB10 has 99 KB shared memory per SM, sufficient
for 2-stage pipelining.

---

## Triton PTXAS Auto-Configuration

### 15. System PTXAS Path Auto-Detection

**File:** `python/sglang/srt/utils/common.py`

Triton's bundled ptxas (CUDA 12.8) rejects `sm_121a` targets. Auto-detects and sets
`TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` at startup when the env var is unset
and system ptxas exists.

---

## Sparse FP4 GEMV for MoE Decode

### 16. 2:4 Structured Sparsity GEMV Kernel

**Files:**
- `python/sglang/srt/layers/moe/sparse_fp4/sparse_fp4_gemv.cu` (new)
- `python/sglang/srt/layers/moe/sparse_fp4/kernel.py` (new)
- `python/sglang/srt/layers/quantization/modelopt_quant.py`

Enabled by `SGLANG_NVFP4_SPARSE=1`. Exploits natural 2:4 structured sparsity in
NVFP4 MoE weights to reduce DRAM traffic by ~22% during decode. The CUDA kernel:
- Extracts the 2 largest-magnitude values per group-of-4 at weight load time
- Stores compressed weights + 2-bit index metadata + FP8 block scales
- Uses constant-memory LUTs for FP4 and FP8 dequantization (no branching)
- Fuses the complete MoE forward: BF16->FP16 cast, GEMM1, SiLU, GEMM2, weighted reduce

Weight conversion happens once in `process_weights_after_loading`. The sparse path
activates automatically for decode (batch_size <= 2), falling back to CUTLASS for
larger batches (prefill).

**Benchmark** (Qwen3.5-122B-A10B, E=256, N=1024, K=3072, topk=8, BS=1 on GB10):
- 0.44 ms per MoE layer
- 31.5 MB read per call (vs 40.5 MB dense, 22% savings)
- 75 GB/s effective bandwidth

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SGLANG_MTP_FP8` | `0` | Post-quantize MTP draft model to FP8 |
| `SGLANG_QUANTIZE_LM_HEAD_FP8` | `0` | Apply FP8 to lm_head at load time |
| `SGLANG_NVFP4_SPARSE` | `0` | Enable sparse FP4 GEMV for MoE decode |
| `TRITON_PTXAS_PATH` | (auto) | System ptxas path (auto-detected on SM121) |

---

## Usage Example

```bash
python -m sglang.launch_server \
  --model Qwen3.5-122B-A10B-NVFP4 \
  --quantization nvfp4 \
  --kv-cache-dtype auto \
  --speculative-algorithm EAGLE \
  --num-speculative-tokens 2 \
  --env SGLANG_MTP_FP8=1 \
  --env SGLANG_NVFP4_SPARSE=1
```
