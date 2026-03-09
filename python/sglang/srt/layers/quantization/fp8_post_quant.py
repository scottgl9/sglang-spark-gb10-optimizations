"""Post-quantization utilities for converting BF16 layers to FP8 at load time.

Used primarily for MTP (multi-token prediction) draft model layers where
the checkpoint stores BF16 weights but FP8 would halve memory bandwidth
with acceptable accuracy for draft tokens.
"""

import logging
import os
from typing import Sequence

import torch
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max  # 448.0


def quantize_linear_bf16_to_fp8(module: torch.nn.Linear, name: str) -> bool:
    """Quantize a plain nn.Linear from BF16/FP16 to FP8 in-place.

    Uses per-tensor weight quantization (one scale for entire weight matrix)
    and dynamic per-token activation quantization at runtime via torch._scaled_mm.

    Returns True if successful, False if skipped.
    """
    if not hasattr(module, "weight"):
        return False

    weight = module.weight.data
    if weight.dtype not in (torch.bfloat16, torch.float16):
        return False

    device = weight.device

    # Per-tensor quantization
    amax = weight.abs().max()
    if amax == 0:
        return False

    scale = amax.float() / FP8_MAX
    weight_fp8 = (weight.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)

    # Replace weight in-place
    module.weight = Parameter(weight_fp8, requires_grad=False)

    # Store scale as buffer (not parameter — doesn't need gradient or weight loading)
    module.register_buffer(
        "weight_scale", torch.tensor(scale.item(), dtype=torch.float32, device=device)
    )
    # Mark as fp8-quantized so forward knows to use scaled_mm
    module._fp8_post_quantized = True

    orig_mb = weight.numel() * weight.element_size() / 1024 / 1024
    fp8_mb = weight_fp8.numel() / 1024 / 1024
    logger.info(
        "FP8 post-quantized %s: %.1f MB (bf16) -> %.1f MB (fp8), scale=%.4f",
        name,
        orig_mb,
        fp8_mb,
        scale.item(),
    )
    return True


class FP8PostQuantLinear(torch.nn.Module):
    """Drop-in replacement for nn.Linear that uses FP8 weights with dynamic activation scaling.

    Created by apply_fp8_post_quant() to replace BF16 nn.Linear modules.
    """

    def __init__(self, weight_fp8: torch.Tensor, weight_scale: torch.Tensor, bias=None):
        super().__init__()
        self.weight = Parameter(weight_fp8, requires_grad=False)
        self.register_buffer("weight_scale", weight_scale)
        self.bias = Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamic per-token activation quantization
        x_flat = x.reshape(-1, x.shape[-1])
        x_amax = x_flat.abs().max()
        # Clamp to avoid division by zero (no host-device sync — CUDA graph safe)
        x_scale = (x_amax.float() / FP8_MAX).clamp(min=1e-12)
        x_fp8 = (x_flat.float() / x_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)

        out = torch._scaled_mm(
            x_fp8,
            self.weight.t(),
            out_dtype=x.dtype,
            scale_a=x_scale,
            scale_b=self.weight_scale,
        )

        if self.bias is not None:
            out = out + self.bias

        return out.view(*x.shape[:-1], self.weight.shape[0])


def apply_fp8_post_quant(
    model: torch.nn.Module,
    layer_patterns: Sequence[str],
) -> int:
    """Walk named_modules of model, converting matching BF16 nn.Linear layers to FP8.

    Replaces nn.Linear modules with FP8PostQuantLinear modules that use
    torch._scaled_mm for FP8 matmul.

    Args:
        model: The model to post-quantize.
        layer_patterns: List of layer name suffixes to match (e.g., ["fc"]).

    Returns:
        Number of layers successfully converted.
    """
    converted = 0
    replacements = []

    for name, module in model.named_modules():
        module_short_name = name.rsplit(".", 1)[-1] if "." in name else name
        if module_short_name not in layer_patterns:
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        if module.weight.dtype not in (torch.bfloat16, torch.float16):
            continue

        weight = module.weight.data
        device = weight.device
        amax = weight.abs().max()
        if amax == 0:
            continue

        scale = amax.float() / FP8_MAX
        weight_fp8 = (weight.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        weight_scale = torch.tensor(scale.item(), dtype=torch.float32, device=device)

        replacement = FP8PostQuantLinear(
            weight_fp8, weight_scale, bias=module.bias.data if module.bias is not None else None
        )
        replacements.append((name, replacement))

        orig_mb = weight.numel() * weight.element_size() / 1024 / 1024
        fp8_mb = weight_fp8.numel() / 1024 / 1024
        logger.info(
            "FP8 post-quantized %s: %.1f MB -> %.1f MB, shape=%s",
            name,
            orig_mb,
            fp8_mb,
            list(weight.shape),
        )
        converted += 1

    # Apply replacements (can't modify during iteration)
    for name, replacement in replacements:
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr_name = name
        setattr(parent, attr_name, replacement)

    if converted > 0:
        logger.info(
            "FP8 post-quantization: converted %d layers matching patterns %s",
            converted,
            layer_patterns,
        )
    return converted


class Fp8PostQuantLinearMethod:
    """Quant method for LinearBase modules (ColumnParallelLinear, RowParallelLinear).

    Uses CUTLASS FP8 GEMM (fp8_scaled_mm) with per-channel weight scales and
    per-token activation scales via sglang_per_token_quant_fp8.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """No-op — weights are already FP8 from post-quantization."""
        pass

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: "Optional[torch.Tensor]" = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.quantization.fp8_utils import (
            fp8_scaled_mm,
            sglang_per_token_quant_fp8,
        )

        x_flat = x.reshape(-1, x.shape[-1])
        qx, x_scale = sglang_per_token_quant_fp8(x_flat)

        out = fp8_scaled_mm(
            qx,
            layer.weight,
            x_scale,
            layer.weight_scale,
            out_dtype=x.dtype,
        )

        if bias is not None:
            out = out + bias

        return out.view(*x.shape[:-1], out.shape[-1])


def apply_fp8_post_quant_linear_base(
    model: torch.nn.Module,
    layer_patterns: Sequence[str],
) -> int:
    """Convert matching BF16 LinearBase layers (ColumnParallelLinear etc.) to FP8.

    Uses per-channel weight quantization (one scale per output channel) for
    compatibility with CUTLASS FP8 GEMM (fp8_scaled_mm). Weights are stored
    transposed [K, N] in column-major layout as required by fp8_scaled_mm.

    Args:
        model: The model to post-quantize.
        layer_patterns: List of layer name suffixes to match.

    Returns:
        Number of layers successfully converted.
    """
    from sglang.srt.layers.quantization.fp8_utils import sglang_per_token_quant_fp8

    converted = 0
    fp8_method = Fp8PostQuantLinearMethod()

    for name, module in model.named_modules():
        module_short_name = name.rsplit(".", 1)[-1] if "." in name else name
        if module_short_name not in layer_patterns:
            continue
        if not hasattr(module, "weight") or not hasattr(module, "quant_method"):
            continue
        weight = module.weight.data
        if weight.dtype not in (torch.bfloat16, torch.float16):
            continue

        orig_shape = list(weight.shape)
        orig_mb = weight.numel() * weight.element_size() / 1024 / 1024

        # Per-channel FP8 quantization (one scale per row = per output channel)
        # sglang_per_token_quant_fp8 treats each row as a "token" → per-row scales
        qweight, weight_scale = sglang_per_token_quant_fp8(weight)
        # qweight: [N, K] fp8, weight_scale: [N, 1] float32
        # fp8_scaled_mm expects: weight=[K, N] column-major, scale=[1, N]
        weight_t = qweight.t()  # [K, N], stride=[1, K] → column-major
        scale_t = weight_scale.t().contiguous()  # [1, N]

        fp8_mb = qweight.numel() / 1024 / 1024

        # Delete BF16 weight, replace with FP8
        del weight
        del module.weight
        module.weight = Parameter(weight_t, requires_grad=False)
        module.weight_scale = Parameter(scale_t, requires_grad=False)
        module.quant_method = fp8_method

        logger.info(
            "FP8 post-quantized (LinearBase) %s: %.2f MB -> %.2f MB, shape=%s",
            name,
            orig_mb,
            fp8_mb,
            orig_shape,
        )
        converted += 1

    if converted > 0:
        logger.info(
            "FP8 post-quantization (LinearBase): converted %d layers matching %s",
            converted,
            layer_patterns,
        )
        torch.cuda.empty_cache()
    return converted


def quantize_mtp_moe_fp8(mtp_model: torch.nn.Module) -> int:
    """Post-quantize MTP MoE expert weights from BF16 to FP8.

    Must be called BEFORE CUDA graph capture so the graphs capture the FP8 path.
    Converts w13_weight and w2_weight from bfloat16 to float8_e4m3fn with
    per-expert per-tensor scales. Halves active-expert memory bandwidth.

    Returns the number of MoE layers quantized.
    """
    # SM120+ (Blackwell): Triton doesn't support fp8e4nv in tl.dot, so MoE
    # FP8 quantization is not compatible with the triton MoE kernel path.
    from sglang.srt.utils.common import is_sm120_supported

    if is_sm120_supported():
        logger.info(
            "quantize_mtp_moe_fp8: skipping on SM120+ (Triton fp8e4nv "
            "dot not supported); FC layers still benefit from FP8"
        )
        return 0
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    count = 0
    for name, layer in mtp_model.named_modules():
        qm = getattr(layer, "quant_method", None)
        if not isinstance(qm, UnquantizedFusedMoEMethod):
            continue
        if not (hasattr(layer, "w13_weight") and hasattr(layer, "w2_weight")):
            continue
        if layer.w13_weight.dtype != torch.bfloat16:
            continue

        num_experts = layer.w13_weight.shape[0]
        w13 = layer.w13_weight.data.float()
        w2 = layer.w2_weight.data.float()

        # Per-expert per-tensor scales
        w13_scale = (
            w13.abs().view(num_experts, -1).max(dim=1).values / FP8_MAX
        ).clamp(min=1e-12)
        w2_scale = (
            w2.abs().view(num_experts, -1).max(dim=1).values / FP8_MAX
        ).clamp(min=1e-12)

        w13_fp8 = (w13 / w13_scale.view(-1, 1, 1)).clamp(-FP8_MAX, FP8_MAX).to(
            FP8_DTYPE
        )
        w2_fp8 = (w2 / w2_scale.view(-1, 1, 1)).clamp(-FP8_MAX, FP8_MAX).to(
            FP8_DTYPE
        )

        layer.w13_weight = Parameter(w13_fp8, requires_grad=False)
        layer.w2_weight = Parameter(w2_fp8, requires_grad=False)

        # Store scales for the MoE runner
        layer.w13_weight_scale = Parameter(w13_scale, requires_grad=False)
        layer.w2_weight_scale = Parameter(w2_scale, requires_grad=False)

        orig_mb = (w13.numel() + w2.numel()) * 2 / 1024 / 1024  # bf16 = 2 bytes
        fp8_mb = (w13_fp8.numel() + w2_fp8.numel()) / 1024 / 1024
        logger.info(
            "quantize_mtp_moe_fp8: %s -> FP8, %.0f MB (bf16) -> %.0f MB (fp8), "
            "num_experts=%d",
            name,
            orig_mb,
            fp8_mb,
            num_experts,
        )
        count += 1

    return count


def maybe_quantize_mtp_fp8(draft_model: torch.nn.Module) -> None:
    """Apply FP8 post-quantization to MTP draft model if SGLANG_MTP_FP8=1.

    Quantizes:
    - fc and lm_head layers (nn.Linear, BF16 -> FP8): halves bandwidth
    - MoE expert weights (BF16 -> FP8): halves expert bandwidth per draft step
    """
    if os.environ.get("SGLANG_MTP_FP8", "0") != "1":
        return

    logger.info("SGLANG_MTP_FP8=1: applying FP8 post-quantization to MTP draft model")

    # Quantize fc and lm_head layers (nn.Linear)
    n_fc = apply_fp8_post_quant(draft_model, layer_patterns=["fc", "lm_head"])

    # Quantize MoE expert weights
    n_moe = quantize_mtp_moe_fp8(draft_model)

    if n_fc == 0 and n_moe == 0:
        logger.info("SGLANG_MTP_FP8: no eligible layers found for FP8 post-quantization")
    else:
        logger.info(
            "SGLANG_MTP_FP8: quantized %d fc layers, %d MoE layers to FP8",
            n_fc,
            n_moe,
        )
