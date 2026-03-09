# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
import logging
from collections.abc import Callable
from typing import Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend
from sglang.srt.layers.quantization.modelopt_quant import (
    enable_flashinfer_fp4_gemm,
    fp4_gemm,
    fp4_quantize,
)
from sglang.srt.layers.quantization.utils import swizzle_blockscale

logger = logging.getLogger(__name__)

# FP4 E2M1 lookup table: maps 4-bit values (0-15) to float
# Bits: SEEM (sign, 2-bit exponent, 1-bit mantissa)
_FP4_E2M1_LUT = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,  # positive
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # negative
]


def _is_sm121() -> bool:
    """Check if running on SM121 (GB10)."""
    cc = torch.cuda.get_device_capability()
    return cc[0] == 12 and cc[1] == 1


def _dequant_nvfp4_to_bf16(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """Dequantize NVFP4 packed weights to BF16.

    Args:
        weight_packed: uint8 [N, K/2], 2 FP4 values per byte
        weight_scale: float8_e4m3fn [N, K/group_size], per-group scale
        weight_global_scale: float32 scalar, per-tensor scale
    Returns:
        BF16 tensor [N, K]
    """
    device = weight_packed.device
    N, K_half = weight_packed.shape
    K = K_half * 2

    # Build LUT on device
    lut = torch.tensor(_FP4_E2M1_LUT, dtype=torch.bfloat16, device=device)

    # Unpack uint8 → two FP4 nibbles
    low_nibble = (weight_packed & 0x0F).long()
    high_nibble = ((weight_packed >> 4) & 0x0F).long()

    # Lookup float values and interleave
    low_vals = lut[low_nibble]
    high_vals = lut[high_nibble]
    weight_bf16 = torch.stack([low_vals, high_vals], dim=-1).reshape(N, K)

    # Apply per-group scale (expand from [N, K/group_size] → [N, K])
    scale_bf16 = weight_scale.to(torch.bfloat16)
    scale_expanded = scale_bf16.repeat_interleave(group_size, dim=1)
    weight_bf16 = weight_bf16 * scale_expanded

    # Divide by global scale (global_scale was used to scale UP before quantization)
    weight_bf16 = weight_bf16 / weight_global_scale.to(torch.bfloat16)

    return weight_bf16

__all__ = ["CompressedTensorsW4A4Fp4"]


class CompressedTensorsW4A4Fp4(CompressedTensorsLinearScheme):
    def __init__(self):
        self.group_size = 16

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Weight
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # Global Weight Scale
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_global_scale", input_global_scale)

    def process_weights_after_loading(self, layer) -> None:
        global_input_scale = layer.input_global_scale.max().to(torch.float32)
        layer.input_global_scale = Parameter(global_input_scale, requires_grad=False)

        layer.weight_global_scale = Parameter(
            layer.weight_global_scale.max().to(torch.float32), requires_grad=False
        )

        # SM121 (GB10): CUTLASS FP4 GEMM produces corrupt output.
        # Dequantize weights to BF16 at load time and use standard matmul.
        if _is_sm121():
            weight_bf16 = _dequant_nvfp4_to_bf16(
                layer.weight_packed.data,
                layer.weight_scale.data,
                layer.weight_global_scale.data,
                self.group_size,
            )
            layer.weight_dequantized = Parameter(weight_bf16, requires_grad=False)
            layer._use_bf16_fallback = True
            # Free FP4 tensors
            del layer.weight_packed
            del layer.weight_scale
            logger.warning(
                "SM121 (GB10): dequantized NVFP4 linear layer to BF16 "
                f"(shape {list(weight_bf16.shape)})"
            )
            return

        if get_fp4_gemm_runner_backend().is_flashinfer_trtllm():
            # FlashInfer TRTLLM FP4 GEMM requires a different weight layout.
            # FlashInfer provides nvfp4_quantize to quantize + shuffle the
            # layout but we use our own quantization so we have to call
            # shuffles ourselves.
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

            weight = layer.weight_packed.data
            weight_scale = layer.weight_scale.data

            epilogue_tile_m = 128
            weight = shuffle_matrix_a(weight.view(torch.uint8), epilogue_tile_m)
            weight_scale = (
                shuffle_matrix_sf_a(weight_scale.view(torch.uint8), epilogue_tile_m)
                .reshape(weight_scale.shape)
                .view(torch.float8_e4m3fn)
            )

            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.weight_packed = Parameter(weight, requires_grad=False)
        else:
            swizzled_weight_scale = swizzle_blockscale(layer.weight_scale)
            layer.weight_scale = Parameter(swizzled_weight_scale, requires_grad=False)
            layer.weight_packed = Parameter(
                layer.weight_packed.data, requires_grad=False
            )

        layer.alpha = Parameter(
            1 / (layer.input_global_scale * layer.weight_global_scale),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # SM121 BF16 fallback path
        if getattr(layer, "_use_bf16_fallback", False):
            out = torch.nn.functional.linear(x, layer.weight_dequantized, bias)
            return out

        output_dtype = x.dtype
        w_n, _ = layer.weight_packed.shape
        output_shape = [x.shape[0], w_n]

        # quantize BF16 or FP16 to (FP4 and interleaved block scale)
        x_fp4, x_blockscale = fp4_quantize(x, layer.input_global_scale)

        assert x_fp4.dtype == torch.uint8
        assert layer.weight_packed.dtype == torch.uint8
        assert layer.weight_scale.dtype == torch.float8_e4m3fn
        assert layer.alpha.dtype == torch.float32

        w = layer.weight_packed
        w_blockscale = layer.weight_scale
        if (
            enable_flashinfer_fp4_gemm
            and not get_fp4_gemm_runner_backend().is_cutlass()
        ):
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
