"""Sparse FP4 GEMV kernel for bandwidth-limited NVFP4 MoE decode.

Exploits 2:4 structured sparsity in NVFP4 weights to reduce DRAM traffic
by ~25%. Only beneficial for decode (batch_size=1 per expert) on
bandwidth-limited hardware like GB10 (273 GB/s LPDDR5X).

Usage:
    # After weight loading, convert to sparse format:
    convert_nvfp4_to_sparse(layer)

    # During decode forward:
    output = sparse_fp4_moe_forward(hidden_states, topk_weights, topk_ids,
                                     expert_map, layer, inter_size,
                                     apply_router_weight_on_input)

Enabled by SGLANG_NVFP4_SPARSE=1 environment variable.
"""

import logging
import os
import pathlib
from typing import Optional

import torch
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)

_kernel_module = None

FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _load_kernel():
    """JIT-compile the sparse FP4 GEMV CUDA kernel."""
    global _kernel_module
    if _kernel_module is not None:
        return _kernel_module

    from torch.utils.cpp_extension import load

    cuda_src = pathlib.Path(__file__).parent / "sparse_fp4_gemv.cu"
    _kernel_module = load(
        name="sparse_fp4_gemv",
        sources=[str(cuda_src)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _kernel_module


def _unpack_fp4_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Unpack [N, K/2] uint8 -> [N, K] nibbles."""
    N, K_half = packed.shape
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = torch.empty(N, K_half * 2, dtype=torch.uint8, device=packed.device)
    nibbles[:, 0::2] = low
    nibbles[:, 1::2] = high
    return nibbles


def _apply_2of4_sparsity(nibbles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract 2:4 sparsity from FP4 nibbles.

    For each group of 4 values, keeps the 2 with largest magnitude.
    Returns compressed values [N, K/4] and metadata [N, K/8].
    """
    N, K = nibbles.shape
    lut = FP4_LUT.to(nibbles.device)
    mags = lut[nibbles.long()].abs()

    nib_4 = nibbles.view(N, K // 4, 4)
    mag_4 = mags.view(N, K // 4, 4)

    _, top2 = mag_4.topk(2, dim=2)
    top2, _ = top2.sort(dim=2)

    v0 = nib_4.gather(2, top2[:, :, 0:1]).squeeze(2)
    v1 = nib_4.gather(2, top2[:, :, 1:2]).squeeze(2)
    comp = ((v1 << 4) | v0).to(torch.uint8)

    i0 = top2[:, :, 0]
    i1 = top2[:, :, 1]
    i0_p = i0.view(N, K // 8, 2)
    i1_p = i1.view(N, K // 8, 2)

    meta = (
        (i0_p[:, :, 0] & 3)
        | ((i1_p[:, :, 0] & 3) << 2)
        | ((i0_p[:, :, 1] & 3) << 4)
        | ((i1_p[:, :, 1] & 3) << 6)
    ).to(torch.uint8)

    return comp, meta


def _convert_expert_to_sparse(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert one expert from NVFP4 packed to sparse format.

    Args:
        weight: [N, K/2] uint8 (packed FP4)
        scale: [N, n_groups] float8_e4m3fn (block scales)

    Returns:
        comp_T: [K/4, N] uint8
        meta_T: [K/8, N] uint8
        scale_T: [n_groups, N] uint8
    """
    nibbles = _unpack_fp4_nibbles(weight)
    comp, meta = _apply_2of4_sparsity(nibbles)
    comp_T = comp.T.contiguous()
    meta_T = meta.T.contiguous()
    scale_T = scale.view(torch.uint8).T.contiguous()
    return comp_T, meta_T, scale_T


def convert_nvfp4_to_sparse(
    layer: torch.nn.Module,
    device: str = "cuda",
    batch_size: int = 16,
) -> bool:
    """Convert NVFP4 MoE weights on a layer to sparse format.

    Reads w13_weight, w2_weight, w13_weight_scale, w2_weight_scale,
    w13_weight_scale_2, w2_weight_scale_2 from the layer and creates
    sparse equivalents stored as layer attributes.

    Returns True if conversion succeeded.
    """
    if not all(
        hasattr(layer, attr)
        for attr in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
    ):
        return False

    if layer.w13_weight.dtype != torch.uint8:
        return False

    w13_weight = layer.w13_weight.data
    w2_weight = layer.w2_weight.data
    w13_scale = layer.w13_weight_scale.data
    w2_scale = layer.w2_weight_scale.data

    # Global scales (per-expert scalar)
    w13_gs = layer.w13_weight_scale_2.data
    w2_gs = layer.w2_weight_scale_2.data
    if w13_gs.dim() > 1:
        w13_gs = w13_gs[:, 0]

    E = w13_weight.shape[0]

    def _convert_batch(weights, scales, g_scales):
        comp_list, meta_list, scale_list = [], [], []
        for start in range(0, E, batch_size):
            end = min(start + batch_size, E)
            batch_w = weights[start:end].to(device)
            batch_s = scales[start:end].to(device)
            for i in range(end - start):
                ct, mt, st = _convert_expert_to_sparse(batch_w[i], batch_s[i])
                comp_list.append(ct)
                meta_list.append(mt)
                scale_list.append(st)
        return (
            torch.stack(comp_list),
            torch.stack(meta_list),
            torch.stack(scale_list),
            g_scales.float().to(device),
        )

    w13_comp, w13_meta, w13_sc, w13_g = _convert_batch(w13_weight, w13_scale, w13_gs)
    w2_comp, w2_meta, w2_sc, w2_g = _convert_batch(w2_weight, w2_scale, w2_gs)

    # Store sparse weights on layer
    layer._sparse_w13_comp = Parameter(w13_comp, requires_grad=False)
    layer._sparse_w13_meta = Parameter(w13_meta, requires_grad=False)
    layer._sparse_w13_scale = Parameter(w13_sc, requires_grad=False)
    layer._sparse_w13_g_scales = Parameter(w13_g, requires_grad=False)
    layer._sparse_w2_comp = Parameter(w2_comp, requires_grad=False)
    layer._sparse_w2_meta = Parameter(w2_meta, requires_grad=False)
    layer._sparse_w2_scale = Parameter(w2_sc, requires_grad=False)
    layer._sparse_w2_g_scales = Parameter(w2_g, requires_grad=False)

    # Memory savings report
    dense_bytes = (
        w13_weight.numel() * w13_weight.element_size()
        + w2_weight.numel() * w2_weight.element_size()
    )
    sparse_bytes = sum(
        t.numel() * t.element_size()
        for t in [w13_comp, w13_meta, w13_sc, w13_g, w2_comp, w2_meta, w2_sc, w2_g]
    )
    logger.info(
        "Sparse FP4: converted %d experts, "
        "w13 comp=%s, w2 comp=%s, "
        "%.1f MB (dense) -> %.1f MB (sparse+meta+scales)",
        E,
        list(w13_comp.shape),
        list(w2_comp.shape),
        dense_bytes / 1024 / 1024,
        sparse_bytes / 1024 / 1024,
    )

    layer._sparse_fp4_enabled = True
    return True


def has_sparse_weights(layer: torch.nn.Module) -> bool:
    """Check if a layer has sparse FP4 weights."""
    return getattr(layer, "_sparse_fp4_enabled", False)


def sparse_fp4_moe_forward(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: Optional[torch.Tensor],
    layer: torch.nn.Module,
    inter_size: int,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    """Run fused sparse FP4 MoE forward pass.

    Uses the pre-converted sparse weights stored on the layer.
    """
    kernel = _load_kernel()
    empty_map = torch.empty(0, dtype=torch.int32, device=hidden_states.device)

    result = kernel.fused_sparse_moe(
        hidden_states,
        topk_weights.float(),
        topk_ids,
        expert_map if expert_map is not None else empty_map,
        layer._sparse_w13_comp,
        layer._sparse_w13_meta,
        layer._sparse_w13_scale,
        layer._sparse_w13_g_scales,
        layer._sparse_w2_comp,
        layer._sparse_w2_meta,
        layer._sparse_w2_scale,
        layer._sparse_w2_g_scales,
        inter_size,
        apply_router_weight_on_input,
    )
    return result


def is_sparse_fp4_enabled() -> bool:
    """Check if sparse FP4 MoE is enabled via environment variable."""
    return os.environ.get("SGLANG_NVFP4_SPARSE", "0") == "1"
