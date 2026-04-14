# SGLang Spark GB10 Optimizations

Fork of [SGLang](https://github.com/sgl-project/sglang) with optimizations for the **NVIDIA Spark (GB10)** — Grace Blackwell SM 12.1.

These patches fix critical issues that prevent NVFP4-quantized models from running on SM121 and add performance optimizations specific to the GB10's unified memory architecture.

## Hardware

| Component | Spec |
|-----------|------|
| Platform | ASUS Ascent GX10 / NVIDIA Spark (GB10) |
| GPU | Grace Blackwell, SM 12.1 |
| Memory | 128 GB unified (CPU+GPU shared) |
| CUDA | 13.0, `TORCH_CUDA_ARCH_LIST=12.1` |

## Performance

| Model | Quantization | Decode Speed | MTP Accept Rate | MTP Accept Length |
|-------|-------------|-------------|----------------|-------------------|
| Qwen3.5-35B-A3B | NVFP4 + GDN post-quant | ~76 tok/s | ~97% | ~2.9 |
| Qwen3.5-122B-A10B | NVFP4 + FP8 post-quant | ~43-45 tok/s | ~90% | ~2.7 |

## Key Optimizations

### 1. NVFP4 Quantization via Marlin FP4

**Problem:** All CUTLASS FP4 GEMM operations produce corrupt output (zeros/NaN) on SM121.

**Solution:** Route all NVFP4 through the Marlin FP4 backend with a scale interleaving fix.

- **Marlin FP4 scale interleaving** — The Marlin FP4 kernel divides `s_tb_groups` by 2, causing adjacent K-groups (16 elements each) to share one scale. Fix: byte-interleave adjacent FP8 scale rows in 8-byte chunks before storing. Applied to both dense and MoE pathways.
- **Marlin MoE kernel fix** — Uncommented `dequant_fp8_scales` block and fixed missing `moe_sum_reduce` 3rd argument (`routed_scaling_factor=1.0`).
- **GDN post-quantization** — Runtime NVFP4 quantization of BF16 GatedDeltaNet layers via Marlin FP4 repack (avoids CUTLASS entirely).
- **MoE backend routing** — SM121 auto-routes compressed-tensors NVFP4 MoE to Marlin backend instead of broken CUTLASS/TRT-LLM paths.
- **relu2/gelu activation** — Added support for non-standard activations in Marlin FP4 MoE.

Key files:
- `python/sglang/srt/layers/quantization/marlin_utils.py` — `nvfp4_marlin_interleave_scales()`, scale processing
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` — Marlin FP4 dense
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` — Marlin MoE
- `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h` — MoE kernel template

### 2. FP8 Post-Quantization

**Finding:** CUTLASS FP8 (`fp8_scaled_mm` from sgl-kernel) is **2-3x faster** than BF16 matmul at M=1 on SM121. In contrast, `torch._scaled_mm` dispatches to `sm89_xmma` and is *not* faster than BF16 on SM121.

- **CUTLASS FP8 for BF16 GDN layers** — Per-channel FP8 quantization + CUTLASS GEMM for layers that can't use NVFP4 (e.g., `in_proj_qkv`, `o_proj` in linear attention blocks).
- **lm_head FP8 routing** — Fixed `LogitsProcessor._compute_lm_head()` which bypassed FP8 for `ParallelLMHead` (checked `hasattr(lm_head, "weight")` → True → fell through to `torch.matmul` BF16 fallback). Now checks `lm_head.weight.dtype == torch.float8_e4m3fn`.
- **MTP draft model FP8** — FP8 post-quantization for speculative decoding draft models with weight_scale sharing from the target model.

Key files:
- `python/sglang/srt/layers/quantization/fp8_post_quant.py` — FP8 post-quant implementation
- `python/sglang/srt/layers/logits_processor.py` — lm_head FP8 routing fix

### 3. Attention & KV Cache

- **Triton attention for MLA models** — FlashInfer MLA is 30x slower on SM121; auto-select Triton attention backend.
- **FP8 KV cache auto-selection** — Auto-detect Blackwell and enable FP8 E4M3 KV cache (halves KV bandwidth vs BF16).

### 4. Kernel Optimizations

- **Sparse FP4 GEMV** — Custom kernel for MoE decode on GB10.
- **CUTLASS FP8 Blockwise GEMM** — Improvements for SM120/SM121.
- **FP4 kernel pre-warm** — Pre-warm piecewise CUDA graphs to avoid first-request latency.

## Supported Models

| Model | Preset | Notes |
|-------|--------|-------|
| Qwen3.5-122B-A10B-NVFP4 | `./sglang.sh Qwen3.5-NVFP4` | MTP speculative decoding, FP8 KV cache |
| Qwen3.5-35B-A3B-NVFP4 | `./sglang.sh Qwen3.5-35B-NVFP4` | MTP, GDN post-quant |
| Mistral-Small-4-119B NVFP4 | `./sglang.sh mistral-small-4` | Triton attention, EAGLE disabled by default |
| Nemotron-3-Super-120B-A12B-NVFP4 | `./sglang.sh nemotron` | FP8 post-quant, Triton MoE |
| MiniMax M2.5 | `./sglang.sh minimax` | NGRAM speculation |
| Qwen3-Coder-Next NVFP4 | `./sglang.sh Qwen3-Coder-Next-NVFP4` | |
| Qwen3-Coder-Next FP8 | `./sglang.sh Qwen3-Coder-Next-FP8` | Dense FP8 |

## Quick Start

### Build

```bash
./sglang.sh build
```

This creates a `.sglang/` venv with all dependencies compiled for SM 12.1. See `./sglang.sh --help` for partial rebuild options (`--skip-venv`, `--skip-torch`, etc.).

### Launch

```bash
# Qwen3.5-122B MoE NVFP4 with MTP speculative decoding
./sglang.sh Qwen3.5-NVFP4

# Qwen3.5-35B with MTP
./sglang.sh Qwen3.5-35B-NVFP4

# Override context length
CONTEXT_LENGTH=32768 ./sglang.sh Qwen3.5-NVFP4

# Disable speculative decoding
DISABLE_MTP=1 ./sglang.sh Qwen3.5-NVFP4
```

### Environment Overrides

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTEXT_LENGTH` | 65536 | Context window size |
| `KV_CACHE_DTYPE` | fp8_e4m3 | KV cache dtype (`auto` for BF16) |
| `DISABLE_MTP` | 0 | Disable MTP speculative decoding |
| `DISABLE_NGRAM` | 0 | Disable NGRAM speculation (minimax) |

Model paths can be overridden with `QWEN35_MODEL`, `NEMOTRON_MODEL`, `MISTRAL_MODEL`, etc.

## Known Limitations

- **CUTLASS FP4 broken on SM121** — All CUTLASS FP4 GEMM produces zeros/NaN. Workaround: Marlin FP4 backend.
- **Nemotron-H NVFP4 MoE + relu2** — FlashInfer CUTLASS FP4 kernel doesn't support relu2 activation on SM121 (512 experts).
- **EAGLE + NVFP4** — EAGLE draft models trained on BF16 are ineffective with NVFP4 base (~6% accept rate). Disabled by default for Mistral-Small-4.
- **`torch._scaled_mm` slow on SM121** — Dispatches to `sm89_xmma` kernel. Use `fp8_scaled_mm` (sgl-kernel CUTLASS FP8) instead.

## Key Files Modified

<details>
<summary>Quantization framework</summary>

- `python/sglang/srt/layers/quantization/marlin_utils.py` — NVFP4 scale processing + interleaving
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` — Marlin FP4 dense GEMM
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` — Marlin FP4 MoE
- `python/sglang/srt/layers/quantization/fp8_post_quant.py` — CUTLASS FP8 post-quant
- `python/sglang/srt/layers/logits_processor.py` — lm_head FP8 routing

</details>

<details>
<summary>Kernels</summary>

- `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h` — MoE dequant_fp8_scales
- `python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_quant.cuh` — FP4 quantization
- `python/sglang/jit_kernel/nvfp4.py` — FP4 JIT kernel

</details>

<details>
<summary>Models</summary>

- `python/sglang/srt/models/qwen3_5.py` — SM121 GDN post-quant, FP8 post-quant
- `python/sglang/srt/models/mistral_small.py` — Mistral-Small-4 NVFP4 support
- `python/sglang/srt/models/jet_nemotron.py` — Nemotron-H NVFP4 loading

</details>

<details>
<summary>Runtime</summary>

- `python/sglang/srt/server_args.py` — SM121 detection, backend auto-selection
- `sglang.sh` — Build script + model launch presets

</details>

## Upstream

Based on [sgl-project/sglang](https://github.com/sgl-project/sglang) main branch. This fork contains 227 commits of GB10-specific optimizations on top of upstream.

## License

Apache 2.0 (same as upstream SGLang)
