"""Regression coverage for compressed-tensors NVFP4 MoE W13 backend order."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestCompressedTensorsNvfp4MoeW13Order(unittest.TestCase):
    @staticmethod
    def _scheme(*, marlin: bool, trtllm: bool):
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4_moe import (
            CompressedTensorsW4A4Nvfp4MoE,
        )

        scheme = object.__new__(CompressedTensorsW4A4Nvfp4MoE)
        scheme.use_marlin = marlin
        scheme.use_flashinfer_trtllm = trtllm
        return scheme

    def test_w13_order_matches_each_backend_contract(self):
        # Marlin's generic silu_and_mul consumes native [gate, up].
        self.assertFalse(
            self._scheme(marlin=True, trtllm=False).load_up_proj_weight_first
        )
        # CUTLASS consumes [up, gate].
        self.assertTrue(
            self._scheme(marlin=False, trtllm=False).load_up_proj_weight_first
        )
        # TRT-LLM receives native order then reorders during post-load.
        self.assertFalse(
            self._scheme(marlin=False, trtllm=True).load_up_proj_weight_first
        )


    def test_marlin_repacker_import_targets_available_kernel_module(self):
        """Explicit Marlin must import the repacker that this checkout provides."""
        import importlib
        import inspect

        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4_moe import (
            CompressedTensorsW4A4Nvfp4MoE,
        )

        source = inspect.getsource(CompressedTensorsW4A4Nvfp4MoE._process_weights_marlin)
        self.assertIn(
            "from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack",
            source,
        )
        module = importlib.import_module("sglang.kernels.ops.quantization.gptq_marlin_repack")
        self.assertTrue(callable(module.gptq_marlin_repack))


    def test_marlin_gemm_import_targets_available_kernel_module(self):
        """The explicit Marlin runtime must import the GEMM wrapper in this checkout."""
        import importlib
        import inspect

        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4_moe import (
            CompressedTensorsW4A4Nvfp4MoE,
        )

        source = inspect.getsource(CompressedTensorsW4A4Nvfp4MoE._apply_weights_marlin)
        self.assertIn(
            "from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm",
            source,
        )
        module = importlib.import_module("sglang.kernels.ops.moe.moe_wna16_marlin")
        self.assertTrue(callable(module.moe_wna16_marlin_gemm))


    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA Marlin repacking")
    def test_marlin_moe_scale_layout_matches_canonical_nvfp4_contract(self):
        """Custom compressed-tensors Marlin scales must match the canonical NVFP4 layout."""
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4_moe import (
            CompressedTensorsW4A4Nvfp4MoE,
        )
        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_permute_scales,
            nvfp4_marlin_process_scales,
        )

        torch.manual_seed(0)
        experts, hidden_size, intermediate_size, group_size = 1, 128, 128, 16
        device = "cuda"
        layer = SimpleNamespace(
            w13_weight=torch.randint(
                0,
                256,
                (experts, 2 * intermediate_size, hidden_size // 2),
                device=device,
                dtype=torch.uint8,
            ),
            w2_weight=torch.randint(
                0,
                256,
                (experts, hidden_size, intermediate_size // 2),
                device=device,
                dtype=torch.uint8,
            ),
            w13_weight_scale=(
                torch.rand(
                    (experts, 2 * intermediate_size, hidden_size // group_size),
                    device=device,
                    dtype=torch.float16,
                )
                + 0.1
            ).to(torch.float8_e4m3fn),
            w2_weight_scale=(
                torch.rand(
                    (experts, hidden_size, intermediate_size // group_size),
                    device=device,
                    dtype=torch.float16,
                )
                + 0.1
            ).to(torch.float8_e4m3fn),
            w13_weight_scale_2=torch.ones(experts, device=device, dtype=torch.bfloat16),
            w2_weight_scale_2=torch.ones(experts, device=device, dtype=torch.bfloat16),
            params_dtype=torch.bfloat16,
        )
        raw_w13_scale = layer.w13_weight_scale.clone()
        raw_w2_scale = layer.w2_weight_scale.clone()
        scheme = object.__new__(CompressedTensorsW4A4Nvfp4MoE)
        scheme.group_size = group_size
        scheme._process_weights_marlin(layer)

        expected_w13 = nvfp4_marlin_process_scales(
            marlin_permute_scales(
                raw_w13_scale[0].to(torch.bfloat16).T.contiguous(),
                hidden_size,
                2 * intermediate_size,
                group_size,
            )
        )
        expected_w2 = nvfp4_marlin_process_scales(
            marlin_permute_scales(
                raw_w2_scale[0].to(torch.bfloat16).T.contiguous(),
                intermediate_size,
                hidden_size,
                group_size,
            )
        )
        self.assertTrue(torch.equal(layer.w13_scale_marlin[0], expected_w13))
        self.assertTrue(torch.equal(layer.w2_scale_marlin[0], expected_w2))


if __name__ == "__main__":
    unittest.main()
