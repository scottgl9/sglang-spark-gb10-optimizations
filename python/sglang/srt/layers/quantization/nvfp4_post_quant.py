"""Post-quantization utilities for converting BF16 linear layers to NVFP4 at load time.

This module provides helpers to post-quantize specific BF16 layers (e.g., GDN
projections like in_proj_qkvz) to NVFP4 format after model weights are loaded.
This is useful for layers that are left in BF16 by the quantization config's
ignore list but are large enough to benefit from FP4 compute.
"""

import logging
from typing import Optional, Sequence

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend
from sglang.srt.layers.quantization.utils import swizzle_blockscale
from sglang.srt.utils.common import is_sm120_supported

logger = logging.getLogger(__name__)


def _is_sm121() -> bool:
    """Check if running on SM121 (GB10)."""
    cc = torch.cuda.get_device_capability()
    return cc[0] == 12 and cc[1] == 1


# FP4 E2M1 lookup table: maps 4-bit values (0-15) to float
# Bits: SEEM (sign, 2-bit exponent, 1-bit mantissa)
_FP4_E2M1_LUT = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,  # positive
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # negative
]
_FP4_MAX = 6.0
_FP8_E4M3_MAX = 448.0


def _quantize_bf16_to_raw_nvfp4(
    weight: torch.Tensor,
    group_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize BF16 weight to raw NVFP4 packed format (checkpoint-compatible).

    Produces the exact same format as compressed-tensors NVFP4 checkpoints:
    - weight_packed: [N, K/2] uint8 (two FP4 nibbles per byte)
    - weight_scale: [N, K/group_size] float8_e4m3fn (per-group scales)
    - weight_global_scale: float32 scalar

    This avoids sgl_fp4_quantize which produces CUTLASS-specific swizzled format.
    """
    device = weight.device
    N, K = weight.shape

    # Compute global scale: global_scale = FP4_MAX * FP8_MAX / amax
    amax = weight.abs().max()
    if amax > 0:
        global_scale = torch.tensor(
            _FP4_MAX * _FP8_E4M3_MAX / amax.item(),
            dtype=torch.float32, device=device,
        )
    else:
        global_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Scale weight by global scale and reshape to groups
    scaled = (weight.float() * global_scale.float()).reshape(N, K // group_size, group_size)

    # Compute per-group scales: scale = group_amax / FP4_MAX
    group_amax = scaled.abs().amax(dim=-1)  # [N, K/gs]
    per_group_scale = (group_amax / _FP4_MAX).clamp(min=1e-12)
    per_group_scale_fp8 = per_group_scale.to(torch.float8_e4m3fn)

    # Normalize each group by its scale
    scale_expanded = per_group_scale_fp8.float().unsqueeze(-1)  # [N, K/gs, 1]
    normalized = scaled / scale_expanded  # [N, K/gs, gs]

    # Build reverse LUT: float -> 4-bit code (nearest neighbor)
    lut_tensor = torch.tensor(_FP4_E2M1_LUT, dtype=torch.float32, device=device)
    # normalized is in [-6, 6], find nearest FP4 value
    flat = normalized.reshape(-1, 1)  # [N*K/gs*gs, 1]
    dists = (flat - lut_tensor.unsqueeze(0)).abs()  # [N*K/gs*gs, 16]
    codes = dists.argmin(dim=-1).to(torch.uint8)  # [N*K/gs*gs]
    codes = codes.reshape(N, K)

    # Pack pairs of FP4 codes into uint8: low_nibble | (high_nibble << 4)
    even = codes[:, 0::2]  # low nibble
    odd = codes[:, 1::2]  # high nibble
    packed = even | (odd << 4)  # [N, K/2] uint8

    return packed, per_group_scale_fp8, global_scale


def _try_import_fp4():
    """Import FP4 quantize and GEMM functions. Returns (fp4_quantize, fp4_gemm) or raises."""
    from sglang.srt.layers.quantization.modelopt_quant import (
        enable_flashinfer_fp4_gemm,
        fp4_gemm,
        fp4_quantize,
    )

    if fp4_quantize is None:
        raise ImportError("fp4_quantize is not available")
    return fp4_quantize, fp4_gemm, enable_flashinfer_fp4_gemm


class Fp4PostQuantLinearMethod:
    """Thin adapter so quant_method.apply() dispatches to FP4 GEMM for post-quantized layers."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.quantization.modelopt_quant import (
            enable_flashinfer_fp4_gemm,
            fp4_gemm,
            fp4_quantize,
        )

        output_dtype = x.dtype
        w_n, _ = layer.weight_packed.shape
        output_shape = [x.shape[0], w_n]

        x_fp4, x_blockscale = fp4_quantize(x, layer.input_global_scale)

        w = layer.weight_packed
        w_blockscale = layer.weight_scale
        if enable_flashinfer_fp4_gemm:
            w = layer.weight_packed.T
            w_blockscale = layer.weight_scale.T

        out = fp4_gemm(
            x_fp4,
            w,
            x_blockscale,
            w_blockscale,
            layer.alpha,
            output_dtype,
            w_n,
        )
        if bias is not None:
            out = out + bias
        return out.view(*output_shape)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Called by the loader after all weights are loaded. No-op for post-quant layers."""
        pass


class MarlinFp4PostQuantLinearMethod:
    """Inference adapter for post-quantized layers using Marlin FP4 dense GEMM (SM121)."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sglang.jit_kernel.gptq_marlin import gptq_marlin_gemm
        from sglang.srt.layers.quantization.marlin_utils import (
            should_use_atomic_add_reduce,
        )
        from sgl_kernel.scalar_type import scalar_types

        reshaped_x = x.reshape(-1, x.shape[-1])
        size_m = reshaped_x.shape[0]
        size_n = layer.marlin_size_n
        size_k = layer.marlin_size_k

        use_atomic_add = should_use_atomic_add_reduce(
            m=size_m, n=size_n, k=size_k, device=x.device, dtype=x.dtype,
        )

        device = x.device
        empty_int = torch.empty(0, dtype=torch.int, device=device)

        out = gptq_marlin_gemm(
            a=reshaped_x,
            c=None,
            b_q_weight=layer.marlin_qweight,
            b_scales=layer.marlin_scales,
            global_scale=layer.marlin_global_scale,
            b_zeros=empty_int,
            g_idx=empty_int,
            perm=empty_int,
            workspace=layer.marlin_workspace,
            b_q_type=scalar_types.float4_e2m1f,
            size_m=size_m,
            size_n=size_n,
            size_k=size_k,
            is_k_full=True,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
        )

        out = out.reshape(x.shape[:-1] + (size_n,))
        if bias is not None:
            out = out + bias
        return out

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass


def _convert_to_marlin_fp4(
    layer: torch.nn.Module,
    layer_name: str,
    weight_packed: torch.Tensor,
    weight_blockscale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    N: int,
    K: int,
    device: torch.device,
) -> bool:
    """Convert post-quantized NVFP4 weights to Marlin FP4 format for SM121."""
    from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack
    from sglang.srt.layers.quantization.marlin_utils import (
        marlin_make_workspace,
        marlin_permute_scales,
        nvfp4_marlin_interleave_scales,
        nvfp4_marlin_process_global_scale,
        nvfp4_marlin_process_scales,
    )

    GROUP_SIZE = 16

    # Alignment check
    if N % 64 != 0 or K % 128 != 0:
        logger.warning(
            f"SM121: Marlin FP4 requires N%64==0, K%128==0 but got "
            f"N={N}, K={K} for {layer_name}. Skipping post-quant."
        )
        return False

    # Repack weights: [N, K/2] uint8 -> Marlin tile layout
    perm = torch.empty(0, dtype=torch.int, device=device)
    qw_int32 = weight_packed.view(torch.int32).T.contiguous()
    marlin_qw = gptq_marlin_repack(qw_int32, perm, K, N, 4)
    del weight_packed, qw_int32  # Free original packed weights

    # Transform scales: [N, K/16] fp8 -> Marlin S0E5M3 + interleaved
    scale_bf16 = weight_blockscale.to(torch.bfloat16)
    del weight_blockscale  # Free original fp8 scales
    scale_t = scale_bf16.T.contiguous()  # [K/16, N]
    del scale_bf16
    scale_permuted = marlin_permute_scales(scale_t, K, N, GROUP_SIZE)
    del scale_t
    marlin_scales = nvfp4_marlin_process_scales(scale_permuted)
    del scale_permuted
    marlin_scales = nvfp4_marlin_interleave_scales(marlin_scales, K, N, GROUP_SIZE)

    # Process global scale: invert + exponent bias correction
    inv_gs = (1.0 / weight_global_scale).to(torch.bfloat16)
    marlin_gs = nvfp4_marlin_process_global_scale(inv_gs)

    # Workspace
    workspace = marlin_make_workspace(device)

    # Remove old weight and register Marlin parameters
    if hasattr(layer, "weight"):
        delattr(layer, "weight")

    layer.register_parameter(
        "marlin_qweight", Parameter(marlin_qw, requires_grad=False)
    )
    layer.register_parameter(
        "marlin_scales", Parameter(marlin_scales, requires_grad=False)
    )
    layer.register_parameter(
        "marlin_global_scale",
        Parameter(marlin_gs.unsqueeze(0), requires_grad=False),
    )
    layer.marlin_workspace = workspace
    layer.marlin_size_n = N
    layer.marlin_size_k = K

    # Swap quant method to Marlin
    layer.quant_method = MarlinFp4PostQuantLinearMethod()

    logger.info(
        f"Post-quantized {layer_name} from BF16 to NVFP4 (Marlin FP4) "
        f"(shape [{N}, {K}], global_scale={weight_global_scale.item():.4f})"
    )
    return True


def _is_already_nvfp4_quantized(layer: torch.nn.Module) -> bool:
    """Detect layers that are already NVFP4 (compressed-tensors layout) so we
    don't try to post-quantize them.

    A layer is considered already-quantized if it carries the compressed-tensors
    NVFP4 tensor set (``weight_packed`` + ``weight_scale``) OR the Marlin FP4
    post-quant tensor set (``marlin_qweight``). Either shape means the checkpoint
    already ships this layer in FP4 and no BF16→FP4 work is needed.
    """
    if hasattr(layer, "weight_packed") and hasattr(layer, "weight_scale"):
        return True
    if hasattr(layer, "marlin_qweight"):
        return True
    return False


def quantize_linear_bf16_to_nvfp4(layer: torch.nn.Module, layer_name: str) -> bool:
    """Quantize a single BF16 linear layer to NVFP4 in-place.

    Replaces the layer's weight with packed FP4 weights and scale tensors,
    and swaps quant_method to Fp4PostQuantLinearMethod.

    Returns True if successful, False if skipped.
    """
    # Compressed-tensors NVFP4 checkpoints ship these layers already quantized —
    # silently skip instead of logging a scary "no weight attribute" warning.
    if _is_already_nvfp4_quantized(layer):
        logger.debug(
            f"Skipping {layer_name}: already NVFP4 (compressed-tensors checkpoint)"
        )
        return False

    if not hasattr(layer, "weight"):
        logger.warning(f"Skipping {layer_name}: no weight attribute")
        return False

    weight = layer.weight.data
    # FP8 per-tensor pre-quantized layers (e.g. lm_head in compressed-tensors
    # float-quantized group): also already quantized — don't try to downgrade.
    if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        logger.debug(
            f"Skipping {layer_name}: weight dtype {weight.dtype} is FP8 "
            f"(already pre-quantized in checkpoint)"
        )
        return False
    if weight.dtype not in (torch.bfloat16, torch.float16):
        logger.debug(f"Skipping {layer_name}: weight dtype {weight.dtype} is not BF16/FP16")
        return False

    device = weight.device
    N, K = weight.shape

    # SM121 (GB10): CUTLASS FP4 is broken. Quantize to raw NVFP4 format
    # (checkpoint-compatible) and use Marlin FP4 dense GEMM instead.
    if _is_sm121():
        weight_packed, weight_blockscale, weight_global_scale = (
            _quantize_bf16_to_raw_nvfp4(weight, group_size=16)
        )
        del weight
        if hasattr(layer, "weight"):
            delattr(layer, "weight")
        return _convert_to_marlin_fp4(
            layer, layer_name, weight_packed, weight_blockscale,
            weight_global_scale, N, K, device,
        )

    # Non-SM121 path: use CUTLASS-compatible quantization
    try:
        fp4_quantize, _, enable_flashinfer_fp4_gemm = _try_import_fp4()
    except ImportError:
        logger.warning(
            f"Cannot post-quantize {layer_name}: fp4_quantize not available"
        )
        return False

    # Quantize the weight to FP4
    # fp4_quantize expects [rows, cols] and returns (packed_uint8, blockscale)
    # Use a dummy global scale of 1.0 for weight quantization
    weight_global_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # We need to compute the actual global scale from the weight max
    amax = weight.abs().max()
    # FP4 E2M1 max representable value is 6.0, group_size=16
    # Global scale = max_fp4 * 448.0 / amax (matching modelopt convention)
    # where 448.0 is max of FP8 E4M3 used for block scales
    FP4_MAX = 6.0
    FP8_MAX = 448.0
    if amax > 0:
        weight_global_scale = torch.tensor(
            FP4_MAX * FP8_MAX / amax.item(), dtype=torch.float32, device=device
        )

    backend = get_fp4_gemm_runner_backend()

    # For non-trtllm backends (e.g. flashinfer_cutlass / sgl_kernel) we need
    # sgl-kernel's scaled_fp4_quant which returns float8_e4m3fn block-scales
    # compatible with swizzle_blockscale().  flashinfer.fp4_quantize returns
    # uint8 scales (already swizzled for flashinfer's own GEMM path) which
    # cause an AssertionError in swizzle_blockscale().
    if not backend.is_flashinfer_trtllm():
        try:
            from sglang.jit_kernel.nvfp4 import scaled_fp4_quant as sgl_fp4_quantize

            weight_packed, weight_blockscale = sgl_fp4_quantize(
                weight, weight_global_scale
            )
        except ImportError:
            weight_packed, weight_blockscale = fp4_quantize(weight, weight_global_scale)
    else:
        weight_packed, weight_blockscale = fp4_quantize(weight, weight_global_scale)

    # Process weight scales (swizzle or shuffle depending on backend)
    if backend.is_flashinfer_trtllm():
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        epilogue_tile_m = 128
        weight_packed = shuffle_matrix_a(weight_packed.view(torch.uint8), epilogue_tile_m)
        weight_blockscale = (
            shuffle_matrix_sf_a(
                weight_blockscale.view(torch.uint8), epilogue_tile_m
            )
            .reshape(weight_blockscale.shape)
            .view(torch.float8_e4m3fn)
        )
    else:
        weight_blockscale = swizzle_blockscale(weight_blockscale)

    # For post-quant, input_global_scale=1.0 (dynamic per-token quantization at runtime)
    input_global_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    alpha = Parameter(
        1.0 / (input_global_scale * weight_global_scale), requires_grad=False
    )

    # Release the BF16 weight tensor before registering FP4 parameters.
    # 'weight' holds a reference that keeps the tensor alive; deleting it now
    # frees ~2× the layer's memory (BF16 + FP4) rather than peak 2× at function exit.
    del weight

    # Remove old weight and register new parameters
    if hasattr(layer, "weight"):
        delattr(layer, "weight")

    layer.register_parameter(
        "weight_packed", Parameter(weight_packed, requires_grad=False)
    )
    layer.register_parameter(
        "weight_scale", Parameter(weight_blockscale, requires_grad=False)
    )
    layer.register_parameter(
        "input_global_scale", Parameter(input_global_scale, requires_grad=False)
    )
    layer.register_parameter(
        "weight_global_scale", Parameter(weight_global_scale, requires_grad=False)
    )
    layer.register_parameter("alpha", alpha)

    # Swap the quant method
    layer.quant_method = Fp4PostQuantLinearMethod()

    logger.info(
        f"Post-quantized {layer_name} from BF16 to NVFP4 "
        f"(shape [{N}, {K}], global_scale={weight_global_scale.item():.4f})"
    )
    return True


def apply_nvfp4_post_quant(
    model: torch.nn.Module,
    layer_patterns: Sequence[str],
) -> int:
    """Walk named_modules of model, converting matching BF16 layers to NVFP4.

    Args:
        model: The model to post-quantize.
        layer_patterns: List of layer name suffixes to match (e.g., ["in_proj_qkvz"]).

    Returns:
        Number of layers successfully converted.
    """
    if not is_sm120_supported():
        return 0

    # SM121 uses pure-Python quantization + Marlin FP4, no fp4_quantize needed
    if not _is_sm121():
        try:
            _try_import_fp4()
        except ImportError:
            logger.warning(
                "NVFP4 post-quantization skipped: fp4_quantize not available. "
                "Install flashinfer or sgl-kernel with FP4 support."
            )
            return 0

    converted = 0
    already_quantized = 0
    for name, module in model.named_modules():
        # Check if module name ends with any of the patterns
        module_short_name = name.rsplit(".", 1)[-1] if "." in name else name
        if module_short_name not in layer_patterns:
            continue

        # Compressed-tensors checkpoints ship matched layers already as NVFP4;
        # count them and move on.
        if _is_already_nvfp4_quantized(module):
            already_quantized += 1
            continue

        if not hasattr(module, "weight"):
            continue

        if quantize_linear_bf16_to_nvfp4(module, name):
            converted += 1

    if already_quantized > 0:
        logger.info(
            f"NVFP4 post-quantization: {already_quantized} layers matching "
            f"patterns {layer_patterns} are already NVFP4 in the checkpoint — skipped"
        )
    if converted > 0:
        logger.info(
            f"NVFP4 post-quantization: converted {converted} BF16 layers "
            f"matching patterns {layer_patterns}"
        )
        # Flush freed BF16 tensors from the CUDA allocator.
        torch.cuda.empty_cache()
    return converted
