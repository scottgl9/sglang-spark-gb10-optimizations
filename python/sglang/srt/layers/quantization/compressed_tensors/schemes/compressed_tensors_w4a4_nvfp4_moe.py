from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams, CutlassMoEType
from sglang.srt.layers.moe.utils import RoutingMethodType, get_moe_runner_backend
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.fp8_utils import is_blackwell_supported
from sglang.srt.layers.quantization.utils import (
    prepare_static_weights_for_trtllm_fp4_moe,
    reorder_w1w3_to_w3w1,
    replace_parameter,
    swizzle_blockscale,
)
from sglang.srt.utils import next_power_of_2, set_weight_attrs

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsW4A4Nvfp4MoE"]

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


class CompressedTensorsW4A4Nvfp4MoE(CompressedTensorsMoEScheme):

    def __init__(self):
        if not is_blackwell_supported():
            raise ValueError(
                "Current platform does not support NVFP4"
                " quantization. Please use Blackwell and"
                " above."
            )
        self.group_size = 16

        # SM121 (GB10): CUTLASS FP4 MoE produces all-zero output on SM121.
        # TRT-LLM FP4 MoE fails with SM100f cubin mismatch (compiled for B200).
        # server_args.py routes SM121 + compressed-tensors -> moe_runner_backend=marlin.
        # Marlin dequants FP4->BF16 per tile on-GPU — works on all GPUs.
        self.use_marlin = get_moe_runner_backend().is_marlin()
        self.use_flashinfer_trtllm = get_moe_runner_backend().is_flashinfer_trtllm()

        if self.use_marlin:
            logger.warning(
                "SM121 (GB10) detected: using Marlin FP4 MoE backend "
                "(CUTLASS/TRT-LLM both incompatible with SM121)"
            )

    @classmethod
    def get_min_capability(cls) -> int:
        # Requires sm100(blackwell) architecture
        return 100

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        layer.params_dtype = params_dtype

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight Scales
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Weight Global Scales
        w13_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, 2, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_weight_global_scale", w13_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs)

        w2_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_weight_global_scale", w2_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs)

        # Input Global Scales
        w13_input_scale = torch.nn.Parameter(
            torch.empty(num_experts, 2, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_input_global_scale", w13_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_input_global_scale", w2_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # From packed to weight
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")

        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        if self.use_flashinfer_trtllm:
            w, s = reorder_w1w3_to_w3w1(
                layer.w13_weight.data, layer.w13_weight_scale.data, dim=-2
            )
            layer.w13_weight = torch.nn.Parameter(w, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(s, requires_grad=False)

        if not torch.allclose(
            layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
        ):
            logger.warning_once(
                "w1_weight_global_scale must match w3_weight_global_scale. "
                "Accuracy may be affected."
            )

        # Take inverse of global scale saved to disk
        layer.w13_weight_scale_2 = torch.nn.Parameter(
            1 / layer.w13_weight_global_scale[:, 0], requires_grad=False
        )

        layer.w2_weight_scale_2 = torch.nn.Parameter(
            1 / layer.w2_weight_global_scale.data, requires_grad=False
        )

        # w13
        if self.use_flashinfer_trtllm:
            w13_input_global_scale = (
                layer.w13_input_global_scale.min()
                .to(torch.float32)
                .expand(layer.num_local_experts)
            )
        else:
            w13_input_global_scale = layer.w13_input_global_scale.min(dim=1).values.to(
                torch.float32
            )
        layer.g1_alphas = torch.nn.Parameter(
            ((1 / w13_input_global_scale) * layer.w13_weight_scale_2),
            requires_grad=False,
        )

        layer.w13_input_scale_quant = torch.nn.Parameter(
            (w13_input_global_scale), requires_grad=False
        )

        # w2
        if self.use_flashinfer_trtllm:
            w2_input_global_scale = (
                layer.w2_input_global_scale.min()
                .to(torch.float32)
                .expand(layer.num_local_experts)
            )
        else:
            w2_input_global_scale = layer.w2_input_global_scale

        layer.g2_alphas = torch.nn.Parameter(
            ((1 / w2_input_global_scale) * layer.w2_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )

        layer.w2_input_scale_quant = torch.nn.Parameter(
            (w2_input_global_scale), requires_grad=False
        )

        # Choose backend weight processing
        if self.use_marlin:
            self._process_weights_marlin(layer)
        elif self.use_flashinfer_trtllm:
            # TensorRT-LLM specific processing
            (
                gemm1_weights_fp4_shuffled,
                gemm1_scales_fp4_shuffled,
                gemm2_weights_fp4_shuffled,
                gemm2_scales_fp4_shuffled,
            ) = prepare_static_weights_for_trtllm_fp4_moe(
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                layer.w2_weight.size(-2),  # hidden_size
                layer.w13_weight.size(-2) // 2,  # intermediate_size
                layer.w13_weight.size(0),  # num_experts
            )
            logger.debug("Finished shuffling weights for TRT-LLM MOE")

            replace_parameter(layer, "w13_weight", gemm1_weights_fp4_shuffled)
            replace_parameter(layer, "w2_weight", gemm2_weights_fp4_shuffled)
            replace_parameter(layer, "w13_weight_scale", gemm1_scales_fp4_shuffled)
            replace_parameter(layer, "w2_weight_scale", gemm2_scales_fp4_shuffled)

            # Additional parameter needed for TRT-LLM
            layer.g1_scale_c = torch.nn.Parameter(
                (layer.w2_input_scale_quant * layer.g1_alphas).to(torch.float32),
                requires_grad=False,
            )
        else:
            # CUTLASS path: swizzle weight scales
            layer.w13_weight_scale = torch.nn.Parameter(
                swizzle_blockscale(layer.w13_weight_scale), requires_grad=False
            )

            layer.w2_weight_scale = torch.nn.Parameter(
                swizzle_blockscale(layer.w2_weight_scale), requires_grad=False
            )

            layer.cutlass_moe_params = CutlassMoEParams(
                CutlassMoEType.BlockscaledFP4,
                layer.w13_weight.device,
                num_experts=layer.num_experts,
                intermediate_size_per_partition=layer.w2_weight.shape[2] * 2,
                hidden_size=layer.w13_weight.shape[2] * 2,
            )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def _process_weights_marlin(self, layer: torch.nn.Module) -> None:
        """Repack NVFP4 MoE weights into Marlin layout for SM121 (GB10).

        Weight formats:
          w13_weight: (E, 2*N, K//2) uint8  — gate+up projections, FP4 packed 2/byte
          w2_weight:  (E, K,   N//2) uint8  — down projection, FP4 packed 2/byte
          w13_weight_scale: (E, 2*N, K//16) float8_e4m3fn  — per-group block scales
          w2_weight_scale:  (E, K,   N//16) float8_e4m3fn
          w13_weight_global_scale: (E, 2) float32  — one per expert per projection
          w2_weight_global_scale:  (E,)   float32

        Marlin kernel needs:
          b_q_weight: Marlin-repacked (E, K//8, 2*N) int32  [for w13]
          b_scales:   marlin_permute_scales applied (E, K//16, 2*N) float8_e4m3fn
          global_scale: (E,) bfloat16
        """
        from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack
        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_moe_permute_scales,
            nvfp4_marlin_process_global_scale,
            nvfp4_marlin_process_scales,
        )

        E = layer.w13_weight.shape[0]
        # hidden_size K: each uint8 byte holds 2 FP4 values → K = shape[-1]*2
        K = layer.w13_weight.shape[2] * 2
        # intermediate_size N: w13 is gate+up so shape[-2] = 2*N
        N = layer.w13_weight.shape[1] // 2
        device = layer.w13_weight.device
        perm = torch.empty(0, dtype=torch.int, device=device)

        # Repack w13 (gate+up): (E, 2*N, K//2) uint8 → Marlin int32 layout
        #   view uint8 as int32: (2*N, K//8), transpose: (K//8, 2*N)
        w13_list = []
        for i in range(E):
            qw = layer.w13_weight[i].view(torch.int32).T.contiguous()
            w13_list.append(gptq_marlin_repack(qw, perm, K, 2 * N, 4))
        layer.w13_weight_marlin = torch.nn.Parameter(
            torch.stack(w13_list), requires_grad=False
        )
        del layer.w13_weight

        # Repack w2 (down): (E, K, N//2) uint8 → Marlin int32 layout
        #   For w2: GEMM contracts over N (intermediate), produces K (hidden)
        #   size_k=N, size_n=K
        #   view uint8 as int32: (K, N//8), transpose: (N//8, K)
        w2_list = []
        for i in range(E):
            qw = layer.w2_weight[i].view(torch.int32).T.contiguous()
            w2_list.append(gptq_marlin_repack(qw, perm, N, K, 4))
        layer.w2_weight_marlin = torch.nn.Parameter(
            torch.stack(w2_list), requires_grad=False
        )
        del layer.w2_weight

        # Permute scales to Marlin layout then apply NVFP4 encoding
        # w13_weight_scale: (E, 2*N, K//16) → convert to params_dtype, transpose, permute, encode
        w13_scale_t = layer.w13_weight_scale.to(layer.params_dtype).permute(0, 2, 1).contiguous()
        w13_scale_permuted = marlin_moe_permute_scales(w13_scale_t, K, 2 * N, self.group_size)
        w13_scale_list = [nvfp4_marlin_process_scales(w13_scale_permuted[e]) for e in range(E)]
        layer.w13_scale_marlin = torch.nn.Parameter(
            torch.stack(w13_scale_list), requires_grad=False
        )
        del layer.w13_weight_scale

        # w2_weight_scale: (E, K, N//16) → same pattern
        w2_scale_t = layer.w2_weight_scale.to(layer.params_dtype).permute(0, 2, 1).contiguous()
        w2_scale_permuted = marlin_moe_permute_scales(w2_scale_t, N, K, self.group_size)
        w2_scale_list = [nvfp4_marlin_process_scales(w2_scale_permuted[e]) for e in range(E)]
        layer.w2_scale_marlin = torch.nn.Parameter(
            torch.stack(w2_scale_list), requires_grad=False
        )
        del layer.w2_weight_scale

        # Global scales: Marlin expects the INVERTED scale (1/original_scale)
        # with exponent bias correction applied, matching vLLM's convention.
        # layer.w13_weight_scale_2 = 1/w13_weight_global_scale[:, 0] (set earlier)
        layer.w13_global_scale_marlin = torch.nn.Parameter(
            nvfp4_marlin_process_global_scale(
                layer.w13_weight_scale_2.to(torch.bfloat16)
            ),
            requires_grad=False,
        )
        layer.w2_global_scale_marlin = torch.nn.Parameter(
            nvfp4_marlin_process_global_scale(
                layer.w2_weight_scale_2.to(torch.bfloat16)
            ),
            requires_grad=False,
        )

        # Pre-allocate Marlin workspace (bounded by sms * 4)
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        layer.workspace_marlin = torch.zeros(
            sms * 4, dtype=torch.int, device=device, requires_grad=False
        )

    def _apply_weights_marlin(
        self,
        layer: torch.nn.Module,
        dispatch_output,
    ):
        """NVFP4 MoE forward pass using Marlin FP4 GEMM (SM121 / GB10).

        Follows fused_marlin_moe pattern:
          GEMM1: hidden (M,K) × w13 (E,2N,K) → gate+up (M*topk, 2N)
          SiLU+Mul: gate+up → intermediate (M*topk, N)
          GEMM2: intermediate (M*topk,N) × w2 (E,K,N) → output (M*topk, K)
          Reduce: (M*topk,K) → (M,K)
        """
        from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm
        from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sgl_kernel import moe_sum_reduce, silu_and_mul
        from sgl_kernel.scalar_type import scalar_types

        import os
        _DEBUG = os.environ.get("MARLIN_DEBUG", "0") == "1"

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids

        M, K = x.shape
        E = layer.w13_weight_marlin.shape[0]
        N = layer.w2_scale_marlin.shape[1] * self.group_size  # intermediate_size
        topk = topk_ids.shape[1]

        if _DEBUG:
            x_nan = x.isnan().any().item()
            logger.info(f"[MARLIN_DEBUG] M={M}, K={K}, N={N}, E={E}, topk={topk}, x.dtype={x.dtype}, x_nan={x_nan}")
            if x_nan:
                logger.info(f"[MARLIN_DEBUG] INPUT HAS NaN! x.abs().max()={x[~x.isnan()].abs().max() if (~x.isnan()).any() else 'all-nan'}")
            logger.info(f"[MARLIN_DEBUG] w13_scale_marlin={layer.w13_scale_marlin.shape}, w2_scale_marlin={layer.w2_scale_marlin.shape}")
            logger.info(f"[MARLIN_DEBUG] w13_global={layer.w13_global_scale_marlin[:3]}, w2_global={layer.w2_global_scale_marlin[:3]}")

        # Block size selection (mirrors fused_marlin_moe)
        block_size_m = 8
        for bsm in [8, 16, 32, 48, 64]:
            if M * topk / E / bsm < 0.9:
                break
            block_size_m = bsm

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, E
        )

        use_atomic_add = (
            x.dtype == torch.half
            or torch.cuda.get_device_capability(x.device)[0] >= 9
        )

        if _DEBUG:
            torch.cuda.synchronize()
            logger.info(f"[MARLIN_DEBUG] pre-GEMM1 ok, sorted_ids={sorted_token_ids.shape}, expert_ids={expert_ids.shape}")

        # GEMM 1: x (M, K)  ×  w13 (E, 2*N, K)  →  intermediate1 (M*topk, 2*N)
        intermediate1 = torch.empty(
            (M * topk, 2 * N), dtype=x.dtype, device=x.device
        )
        moe_wna16_marlin_gemm(
            a=x,
            c_or_none=intermediate1,
            b_q_weight=layer.w13_weight_marlin,
            b_bias_or_none=None,
            b_scales=layer.w13_scale_marlin,
            global_scale_or_none=layer.w13_global_scale_marlin,
            b_zeros_or_none=None,
            g_idx_or_none=None,
            perm_or_none=None,
            workspace=layer.workspace_marlin,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            topk_weights=topk_weights,
            moe_block_size=block_size_m,
            top_k=topk,
            mul_topk_weights=False,
            is_ep=False,
            b_q_type=scalar_types.float4_e2m1f,
            size_m=M,
            size_n=2 * N,
            size_k=K,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
        )

        if _DEBUG:
            torch.cuda.synchronize()
            logger.info(f"[MARLIN_DEBUG] GEMM1 ok, intermediate1={intermediate1.shape}, has_nan={intermediate1.isnan().any()}, has_inf={intermediate1.isinf().any()}")

        # SiLU + gate-mul: (M*topk, 2*N) → (M*topk, N)
        intermediate2 = torch.empty(
            (M * topk, N), dtype=x.dtype, device=x.device
        )
        silu_and_mul(intermediate1, intermediate2)

        if _DEBUG:
            torch.cuda.synchronize()
            i2_nan = intermediate2.isnan().any().item()
            logger.info(f"[MARLIN_DEBUG] SiLU: intermediate2={intermediate2.shape}, nan={i2_nan}, abs_max={intermediate2[~intermediate2.isnan()].abs().max().item():.4g}" if not i2_nan else f"[MARLIN_DEBUG] SiLU: intermediate2={intermediate2.shape}, nan=True")

        # GEMM 2: intermediate2 (M*topk, N) × w2 (E, K, N) → output (M*topk, K)
        # Each intermediate row maps to exactly one expert, so recompute routing
        # with size_m=M*topk and top_k=1.
        flat_topk_ids = topk_ids.view(-1).unsqueeze(1)  # (M*topk, 1)
        flat_topk_weights = topk_weights.view(-1).unsqueeze(1)  # (M*topk, 1)

        sorted_token_ids_2, expert_ids_2, num_tokens_post_padded_2 = (
            moe_align_block_size(flat_topk_ids, block_size_m, E)
        )

        if _DEBUG:
            logger.info(f"[MARLIN_DEBUG] pre-GEMM2: flat_topk_ids={flat_topk_ids.shape}, sorted_ids_2={sorted_token_ids_2.shape}, expert_ids_2={expert_ids_2.shape}")
            logger.info(f"[MARLIN_DEBUG] pre-GEMM2: size_m={M*topk}, size_n={K}, size_k={N}")

        output = torch.zeros(
            (M * topk, K), dtype=x.dtype, device=x.device
        )
        moe_wna16_marlin_gemm(
            a=intermediate2,
            c_or_none=output,
            b_q_weight=layer.w2_weight_marlin,
            b_bias_or_none=None,
            b_scales=layer.w2_scale_marlin,
            global_scale_or_none=layer.w2_global_scale_marlin,
            b_zeros_or_none=None,
            g_idx_or_none=None,
            perm_or_none=None,
            workspace=layer.workspace_marlin,
            sorted_token_ids=sorted_token_ids_2,
            expert_ids=expert_ids_2,
            num_tokens_post_padded=num_tokens_post_padded_2,
            topk_weights=flat_topk_weights,
            moe_block_size=block_size_m,
            top_k=1,
            mul_topk_weights=True,
            is_ep=False,
            b_q_type=scalar_types.float4_e2m1f,
            size_m=M * topk,
            size_n=K,
            size_k=N,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
        )

        if _DEBUG:
            torch.cuda.synchronize()
            o_nan = output.isnan().any().item()
            logger.info(f"[MARLIN_DEBUG] GEMM2: output={output.shape}, nan={o_nan}, abs_max={output[~output.isnan()].abs().max().item():.4g}" if not o_nan else f"[MARLIN_DEBUG] GEMM2: output={output.shape}, nan=True, non_nan_count={(~output.isnan()).sum().item()}")

        # Reduce topk expert outputs: (M*topk, K) → (M, K)
        final_output = torch.empty((M, K), dtype=x.dtype, device=x.device)
        moe_sum_reduce(output.view(M, topk, K), final_output, 1.0)

        if _DEBUG:
            torch.cuda.synchronize()
            fo_nan = final_output.isnan().any().item()
            fo_abs = final_output[~final_output.isnan()].abs() if not fo_nan else final_output.abs()
            logger.info(f"[MARLIN_DEBUG] reduce: final_output={final_output.shape}, nan={fo_nan}, abs_max={fo_abs.max().item():.4g}, abs_mean={fo_abs.mean().item():.4g}")

        return StandardCombineInput(hidden_states=final_output)


    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        if self.use_marlin:
            return self._apply_weights_marlin(layer, dispatch_output)
        elif self.use_flashinfer_trtllm:
            from flashinfer import fp4_quantize, trtllm_fp4_block_scale_moe

            router_logits = topk_output.router_logits
            topk_config = topk_output.topk_config

            # Quantize input hidden states using fp4_quantize
            hs_fp4_bytes, hs_sf_bytes = fp4_quantize(
                x,
                layer.w13_input_scale_quant,
                self.group_size,  # sf_vec_size
                False,  # use_ue8m0
                False,  # is_sf_swizzled_layout
            )
            hs_fp4 = hs_fp4_bytes.reshape(x.shape[0], x.shape[1] // 2)
            hs_scale = hs_sf_bytes.view(torch.float8_e4m3fn).reshape(
                *hs_sf_bytes.shape[:-1], -1
            )

            correction_bias = (
                None
                if topk_config.correction_bias is None
                else topk_config.correction_bias.to(x.dtype)
            )

            assert layer.routing_method_type is not None

            # DeepSeekV3 style routing requires float32 router logits
            if layer.routing_method_type == RoutingMethodType.DeepSeekV3:
                router_logits = router_logits.to(torch.float32)

            routed_scaling_factor = self.moe_runner_config.routed_scaling_factor
            routed_scaling_factor = (
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            )

            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                num_tokens = hs_fp4.shape[0]
                hidden_size = (
                    hs_fp4.shape[-1] * 2
                    if hs_fp4.dtype == torch.uint8
                    else hs_fp4.shape[-1]
                )
                symm_output = torch.empty(
                    num_tokens, hidden_size, dtype=torch.bfloat16, device=hs_fp4.device
                )

            output = trtllm_fp4_block_scale_moe(
                routing_logits=router_logits,
                routing_bias=correction_bias,
                hidden_states=hs_fp4,
                hidden_states_scale=hs_scale,
                gemm1_weights=layer.w13_weight,
                gemm1_weights_scale=layer.w13_weight_scale.view(torch.float8_e4m3fn),
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=layer.w2_weight,
                gemm2_weights_scale=layer.w2_weight_scale.view(torch.float8_e4m3fn),
                gemm2_bias=None,
                output1_scale_scalar=layer.g1_scale_c,
                output1_scale_gate_scalar=layer.g1_alphas,
                output2_scale_scalar=layer.g2_alphas,
                num_experts=layer.num_experts,
                top_k=topk_config.top_k,
                n_group=topk_config.num_expert_group,
                topk_group=topk_config.topk_group,
                intermediate_size=layer.intermediate_size_per_partition,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
                local_num_experts=layer.num_local_experts,
                routed_scaling_factor=routed_scaling_factor,
                routing_method_type=layer.routing_method_type,
                do_finalize=True,
                tune_max_num_tokens=next_power_of_2(hs_fp4.shape[0]),
                output=symm_output,
            )[0]
        else:
            from sglang.srt.layers.moe.cutlass_moe import cutlass_moe_fp4

            topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

            output = cutlass_moe_fp4(
                a=x,
                a1_gscale=layer.w13_input_scale_quant,
                w1_fp4=layer.w13_weight,
                w1_blockscale=layer.w13_weight_scale,
                w1_alphas=layer.g1_alphas,
                a2_gscale=layer.w2_input_scale_quant,
                w2_fp4=layer.w2_weight,
                w2_blockscale=layer.w2_weight_scale,
                w2_alphas=layer.g2_alphas,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                params=layer.cutlass_moe_params,
                apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,
            ).to(x.dtype)

        return StandardCombineInput(hidden_states=output)
