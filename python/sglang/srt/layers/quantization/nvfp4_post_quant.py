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


def quantize_linear_bf16_to_nvfp4(layer: torch.nn.Module, layer_name: str) -> bool:
    """Quantize a single BF16 linear layer to NVFP4 in-place.

    Replaces the layer's weight with packed FP4 weights and scale tensors,
    and swaps quant_method to Fp4PostQuantLinearMethod.

    Returns True if successful, False if skipped.
    """
    try:
        fp4_quantize, _, enable_flashinfer_fp4_gemm = _try_import_fp4()
    except ImportError:
        logger.warning(
            f"Cannot post-quantize {layer_name}: fp4_quantize not available"
        )
        return False

    if not hasattr(layer, "weight"):
        logger.warning(f"Skipping {layer_name}: no weight attribute")
        return False

    weight = layer.weight.data
    if weight.dtype not in (torch.bfloat16, torch.float16):
        logger.debug(f"Skipping {layer_name}: weight dtype {weight.dtype} is not BF16/FP16")
        return False

    device = weight.device
    N, K = weight.shape

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

    try:
        _try_import_fp4()
    except ImportError:
        logger.warning(
            "NVFP4 post-quantization skipped: fp4_quantize not available. "
            "Install flashinfer or sgl-kernel with FP4 support."
        )
        return 0

    converted = 0
    for name, module in model.named_modules():
        # Check if module name ends with any of the patterns
        module_short_name = name.rsplit(".", 1)[-1] if "." in name else name
        if module_short_name not in layer_patterns:
            continue

        if not hasattr(module, "weight"):
            continue

        if quantize_linear_bf16_to_nvfp4(module, name):
            converted += 1

    if converted > 0:
        logger.info(
            f"NVFP4 post-quantization: converted {converted} layers "
            f"matching patterns {layer_patterns}"
        )
        # Flush freed BF16 tensors from the CUDA allocator.
        torch.cuda.empty_cache()
    return converted
