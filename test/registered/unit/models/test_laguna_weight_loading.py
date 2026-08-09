"""Regression coverage for Laguna checkpoint weight-name loading."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.models.laguna import LagunaForCausalLM


class _FakeParam:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, param, loaded_weight, *args, **kwargs):
        self.loaded = (param, loaded_weight, args, kwargs)


class TestLagunaWeightLoading(unittest.TestCase):
    def _make_minimal_model(self, named_parameters=()):
        model = object.__new__(LagunaForCausalLM)
        model.config = SimpleNamespace(
            num_experts=0,
            mlp_layer_types=(),
            tie_word_embeddings=False,
        )
        model.model = SimpleNamespace(start_layer=0, end_layer=1)
        model.named_parameters = lambda: iter(named_parameters)
        return model

    def test_static_kv_scales_load_into_nested_radix_attention(self):
        params = {
            "model.layers.0.self_attn.attn.k_scale": _FakeParam(),
            "model.layers.0.self_attn.attn.v_scale": _FakeParam(),
        }
        model = self._make_minimal_model(list(params.items()))
        weights = [
            ("model.layers.0.self_attn.k_scale", torch.tensor(0.0125)),
            ("model.layers.0.self_attn.v_scale", torch.tensor(0.025)),
        ]

        model.load_weights(weights)

        expected_weights = dict(weights)
        for name, param in params.items():
            self.assertIsNotNone(param.loaded, f"{name} was not loaded")
            checkpoint_name = name.replace(".attn.", ".")
            self.assertIs(param.loaded[1], expected_weights[checkpoint_name])


    def test_fp8_kv_cache_scheme_registers_nested_radix_scales(self):
        quant_config = CompressedTensorsConfig.from_config(
            {
                "format": "nvfp4-pack-quantized",
                "config_groups": {},
                "ignore": [],
                "kv_cache_scheme": {"type": "float", "num_bits": 8},
            }
        )
        attn = RadixAttention(
            num_heads=4,
            head_dim=8,
            scaling=0.25,
            num_kv_heads=1,
            layer_id=0,
            quant_config=quant_config,
            prefix="model.layers.0.self_attn.attn",
        )

        params = dict(attn.named_parameters())
        self.assertIsNotNone(attn.quant_method)
        self.assertIn("k_scale", params)
        self.assertIn("v_scale", params)
        self.assertEqual(attn.k_scale.item(), -1.0)
        self.assertEqual(attn.v_scale.item(), -1.0)


if __name__ == "__main__":
    unittest.main()
