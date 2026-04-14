# GB10 (SM121) NVFP4 Inference Optimizations

This document describes the optimizations on the `gb10-optimization` branch
targeting the NVIDIA GB10 GPU (DGX Spark, compute capability 12.1) with
compressed-tensors NVFP4 models (e.g. Qwen3.5-35B-A3B-NVFP4, Qwen3.5-122B-A10B-NVFP4).

## Hardware Context

| Spec | GB10 |
|---|---|
| Memory | 128 GB LPDDR5X unified (CPU+GPU) |
| Bandwidth | ~273 GB/s |
| Shared Memory | 99 KB/SM (opt-in), 48 KB default |
| SM version | 12.1 |
| CUTLASS FP4 GEMM | **Broken** — produces zeros/NaN |
| CUTLASS FP8 GEMM | Working |
| Marlin FP4 GEMM | Working |

All decode optimizations target memory bandwidth, as decode is fully bandwidth-bound
on GB10 (29x less bandwidth than B200).

---

## Root Cause: CUTLASS FP4 Broken on SM121

CUTLASS FP4 GEMM kernels (used by compressed-tensors W4A4 NVFP4 layers) target
SM100/SM120 but produce all-zeros or NaN output on SM121. This affects:
- MoE expert GEMMs (gate_up_proj, down_proj)
- Attention projections (qkv_proj, o_proj)
- Shared expert projections
- GDN post-quantization

The fix for all affected layers is to route through the **Marlin FP4 GEMM kernel**
(`gptq_marlin_gemm`), which is proven working on SM121.

---

## Fix 1: Marlin MoE Kernel — `dequant_fp8_scales` (Critical)

**File:** `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h`

The `dequant_fp8_scales` code block for FP4 weight types was commented out in
sglang's MoE Marlin fork (but active in vLLM and the non-MoE kernel). Without
runtime S0E5M3→BF16 scale conversion, scale bytes were misinterpreted as raw BF16
values, causing SiLU output to overflow to Inf.

**Fix:** Uncommented the `dequant_fp8_scales` block for `w_type == kFE2M1f`.

**Impact:** Core enabler for MoE NVFP4 on SM121. Without this, all MoE layers
produce Inf/NaN output.

---

## Fix 2: `moe_sum_reduce` Missing Argument

**File:** `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py`

`moe_sum_reduce(input, output)` was called with 2 arguments but requires 3
`(input, output, routed_scaling_factor)`. This produced all-zero MoE output.

**Fix:** Added `1.0` as the third argument.

---

## Fix 3: Marlin FP4 Scale Processing + Interleaving

**File:** `python/sglang/srt/layers/quantization/marlin_utils.py`

Three new functions required to correctly transform NVFP4 scales for the Marlin kernel:

### `nvfp4_marlin_process_scales(scales)`
Converts FP8 E4M3 block scales to S0E5M3 format required by the Marlin FP4 kernel.
Applies a sign bit strip and exponent bias correction.

### `nvfp4_marlin_process_global_scale(global_scale)`
Corrects exponent bias for the global dequantization scale.

### `nvfp4_marlin_interleave_scales(scales, K, N, group_size)`
**Critical.** The Marlin FP4 kernel template divides `s_tb_groups` by 2, causing
adjacent K-groups (16 elements each) to be loaded as pairs via `int4` loads where
`warp_row % 2` selects the even/odd `int2` (8-byte) chunk.

Without interleaving, adjacent groups 0 and 1 would both read from scale[0],
producing garbage output. The fix byte-interleaves adjacent fp8 scale rows in
8-byte chunks so each K-group gets its own correct scale.

**Applied to both dense and MoE scale processing.**

---

## Fix 4: Marlin FP4 Dense GEMM for SM121

**File:** `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py`

Replaces the broken CUTLASS FP4 path with Marlin FP4 GEMM for non-MoE NVFP4 layers
(attention qkv_proj/o_proj, shared expert gate_up/down_proj) on SM121.

**Weight preparation** (`process_weights_after_loading`):
1. `gptq_marlin_repack()` — repacks packed FP4 weights to Marlin tile layout
2. `marlin_permute_scales()` → `nvfp4_marlin_process_scales()` — S0E5M3 format
3. `nvfp4_marlin_interleave_scales()` — 8-byte chunk interleaving
4. Invert global scale (1/scale) → `nvfp4_marlin_process_global_scale()`

**Inference** (`apply_weights`): calls `gptq_marlin_gemm()` with `float4_e2m1f` type.

Falls back to BF16 dequant for layers where `N % 64 != 0` or `K % 128 != 0`.

---

## Fix 5: GDN NVFP4 Post-Quantization via Marlin FP4

**Files:**
- `python/sglang/srt/layers/quantization/nvfp4_post_quant.py`
- `python/sglang/srt/models/qwen3_5.py`

GDN linear attention projections (`in_proj_qkv`, `in_proj_z`) are BF16 in the
checkpoint (listed in the model's quantization ignore list). On SM120+, these are
post-quantized to NVFP4 at load time.

On SM121, CUTLASS FP4 is broken, so the post-quantization path uses pure Python
BF16→NVFP4 quantization (producing checkpoint-compatible packed FP4 format) followed
by Marlin repack with interleaved scales.

**`MarlinFp4PostQuantLinearMethod`** — the quant method used for these layers:
calls `gptq_marlin_gemm()` during inference.

Applied to: 60 GDN layers (35B model), 72 GDN layers (122B model).

**Impact:** +13.5% decode speedup for 35B model (cosine similarity vs BF16: 0.9987).

---

## Fix 6: CUTLASS FP8 Post-Quantization for Remaining BF16 GDN Layers

**File:** `python/sglang/srt/layers/quantization/fp8_post_quant.py`

GDN layers `in_proj_a`, `in_proj_b`, and `out_proj` remain BF16 in the checkpoint
and are not suitable for FP4 quantization (accuracy-sensitive with small output dims).
Post-quantized to FP8 at load time using per-channel quantization for CUTLASS FP8 GEMM.

**`Fp8PostQuantLinearMethod`** — uses `fp8_scaled_mm` (CUTLASS FP8, 2x faster than
BF16 matmul on SM121) with per-token activation quantization via `sglang_per_token_quant_fp8`.

Applied to 108 layers in the 122B model.

**Note:** `torch._scaled_mm` is NOT used — it dispatches to `sm89_xmma` which is
not optimized for SM121 and is no faster than BF16. Always use `fp8_scaled_mm`
from sgl-kernel for FP8 on SM121.

---

## Fix 7: lm_head FP8 Routing (`SGLANG_QUANTIZE_LM_HEAD_FP8`)

**Files:**
- `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
- `python/sglang/srt/layers/logits_processor.py`

### Bug 1: `get_quant_method()` never matched `ParallelLMHead`

`ParallelLMHead` extends `VocabParallelEmbedding` (not `LinearBase`). The existing
`get_quant_method()` only checked `isinstance(layer, LinearBase)`, so even when
`lm_head_fp8_config` was set via `SGLANG_QUANTIZE_LM_HEAD_FP8=1`, the lm_head
stayed BF16.

**Fix:** Added `isinstance(layer, ParallelLMHead)` check before the `LinearBase`
branch. `ParallelLMHead` now receives `Fp8LinearMethod`, which quantizes the weight
to FP8 per-channel and uses CUTLASS `fp8_scaled_mm` during inference.

### Bug 2: `LogitsProcessor._compute_lm_head()` bypassed quant_method

`_compute_lm_head()` checked `hasattr(lm_head, "weight")` first — true for
`ParallelLMHead` — routing to `torch.matmul` even when the weight was FP8.

**Fix:** Added a dtype check: if `lm_head.weight.dtype == torch.float8_e4m3fn`,
route through `quant_method.apply()` for CUTLASS FP8 GEMM.

**Impact:** lm_head (largest single layer: 151936×5120 for 122B) now uses FP8,
providing a 2.75x bandwidth reduction vs BF16.

---

## Fix 8: MTP Draft Model `weight_scale` Sharing

**Files:**
- `python/sglang/srt/speculative/eagle_worker_v2.py`
- `python/sglang/srt/speculative/eagle_worker.py`

After Fix 7, MTP acceptance rate dropped from ~97% to ~33% (zero draft token acceptance).

**Root cause:** `set_embed_and_head()` shares only the `weight` tensor from target
to draft model. The draft model's `lm_head` had no checkpoint weights (`mtp.lm_head.weight`
absent from checkpoint), so `process_weights_after_loading()` quantized
random/uninitialized data, producing a meaningless `weight_scale`. After weight sharing,
the draft model used the correct FP8 weight with the wrong scale → garbage logits.

**Fix:** After `set_embed_and_head()`, also copy `weight_scale` and `input_scale`
from the target model's lm_head to the draft model's lm_head.

**Impact:** MTP acceptance rate restored to ~88-97%.

---

## Performance Results

### Qwen3.5-35B-A3B-NVFP4 on GB10

| Config | Decode TPS (with MTP) |
|--------|----------------------|
| BF16 dequant fallback (before fixes) | ~60 |
| Marlin FP4 dense + interleaved scales | ~67 (+12%) |
| + GDN NVFP4 post-quantization | **~76 (+27%)** |

### Qwen3.5-122B-A10B-NVFP4 on GB10

| Config | Decode TPS (with MTP) |
|--------|----------------------|
| Baseline (NVFP4 + GDN post-quant only) | ~36.9 |
| + FP8 GDN layers (torch._scaled_mm — slow) | ~36.9 (no gain) |
| + CUTLASS FP8 GDN layers | ~37.9 (+3%) |
| + FP8 lm_head + MTP scale sharing | **~43-45 (+17-22%)** |

---

## Kernel Tuning Results

### Marlin `pipe_stages` (No Improvement)

Tested `pipe_stages = 4, 5, 6, 7` — all produce identical latency (~10 µs per
FP4 GEMM call). SM121 fully hides memory latency at 4 stages; more stages add
shared memory usage without benefit.

- `pipe_stages = 3`: fails compile-time `static_assert`
- `pipe_stages = 8`: exceeds 99 KB opt-in shared memory limit for primary decode config

The Marlin FP4 kernel is at the memory bandwidth ceiling on SM121.

---

## SM121 Execution Path Summary

| Layer Type | Backend | Status |
|------------|---------|--------|
| MoE expert weights (W4A4 NVFP4) | Marlin FP4 MoE kernel | ✓ Working |
| Attention qkv_proj, o_proj (W4A4 NVFP4) | Marlin FP4 dense GEMM | ✓ Working |
| Shared expert gate_up/down (W4A4 NVFP4) | Marlin FP4 dense GEMM | ✓ Working |
| GDN in_proj_qkv, in_proj_z (post-quant) | Marlin FP4 dense GEMM | ✓ Working |
| GDN in_proj_a, in_proj_b, out_proj (post-quant) | CUTLASS FP8 | ✓ Working |
| lm_head (post-quant) | CUTLASS FP8 | ✓ Working |
| KV cache | FP8 E4M3 | ✓ Working |
| CUTLASS FP4 GEMM | — | ✗ Broken on SM121 |

---

## Environment Variables

| Variable | Default (SM121) | Description |
|---|---|---|
| `SGLANG_QUANTIZE_LM_HEAD_FP8` | `1` | Post-quantize lm_head to FP8 |
| `SGLANG_MTP_FP8` | `0` | MTP draft FP8 post-quant (disabled: Triton fp8e4nv not supported on SM120+) |
| `SGLANG_ENABLE_JIT_DEEPGEMM` | `0` | DeepGEMM JIT disabled (fails on SM121) |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | FP8 KV cache (CUTLASS FP8 validated on SM121) |

---

## Investigated Options (Not Applicable to SM121)

### `--fp4-gemm-backend flashinfer_cudnn`

Described as "optimal on CUDA 13+ with cuDNN 9.15+". GB10 has CUDA 13.0 and cuDNN 9.13.0
(just below the threshold), but more importantly this backend routes **CUTLASS-based** FP4
GEMM through cuDNN — which still uses CUTLASS FP4 under the hood. CUTLASS FP4 is broken on
SM121, so this backend would produce the same corrupt output regardless.

On SM121, all NVFP4 GEMMs are routed through **Marlin FP4** (`gptq_marlin_gemm`), bypassing
`fp4_gemm_runner_backend` entirely. The `--fp4-gemm-backend` flag has no effect on SM121.

### `--mm-attention-backend flashinfer_cudnn`

FlashInfer cuDNN backend for multimodal (vision) attention. Not applicable — we are running
text-only models (Qwen3.5-122B-A10B-NVFP4 has no vision encoder).

---

## Launch Command (Qwen3.5-122B on GB10)

```bash
cd ~/sandbox/sglang
bash sglang.sh Qwen3.5-NVFP4
```

This auto-detects SM121 and applies all necessary overrides. See `sglang.sh` for
the full argument list.
