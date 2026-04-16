"""Fix 9d: Register FP4 JIT kernels as torch.library custom ops with fake impls.

Required for --enable-piecewise-cuda-graph on GB10 SM 12.1.
Patches flashinfer/fp4_quantization.py in site-packages.
"""
import sys

path = sys.argv[1]
with open(path) as f:
    src = f.read()

marker = "\ndef _compute_swizzled_layout_sf_size("
if marker not in src:
    print(f"WARNING: Fix 9d marker not found in {path} — skipping", file=sys.stderr)
    sys.exit(0)

if "_ensure_fp4_fns_cached" in src:
    print("Fix 9d: already applied, skipping")
    sys.exit(0)

INJECTION = '''
# Module-level caches for torch.library custom-op wrappers (Fix 9d — GB10 SM 12.1)
_fp4_quantize_fn = None
_block_scale_interleave_fn = None


def _ensure_fp4_fns_cached(arch_key: str) -> None:
    """Register FP4 JIT kernels as torch.library custom ops with fake impls."""
    global _fp4_quantize_fn, _block_scale_interleave_fn
    if _fp4_quantize_fn is not None:
        return

    jit_mod = get_fp4_quantization_module(arch_key)

    _fp4_q_name = f"flashinfer_fp4_rt::fp4_quantize_{arch_key}"

    @torch.library.custom_op(_fp4_q_name, mutates_args=())
    def _fp4_quantize_op(
        input: torch.Tensor,
        global_scale,
        sf_vec_size: int,
        sf_use_ue8m0: bool,
        is_sf_swizzled_layout: bool,
        is_sf_8x4_layout: bool,
        enable_pdl: bool,
    ):
        return jit_mod.fp4_quantize_sm100(
            input, global_scale, sf_vec_size, sf_use_ue8m0,
            is_sf_swizzled_layout, is_sf_8x4_layout, enable_pdl,
        )

    @torch.library.register_fake(_fp4_q_name)
    def _fp4_quantize_fake(
        input: torch.Tensor,
        global_scale,
        sf_vec_size: int,
        sf_use_ue8m0: bool,
        is_sf_swizzled_layout: bool,
        is_sf_8x4_layout: bool,
        enable_pdl: bool,
    ):
        out_val = input.new_empty(
            (*input.shape[:-1], input.shape[-1] // 2), dtype=torch.uint8
        )
        m = input.numel() // input.shape[-1]
        k = input.shape[-1]
        if is_sf_swizzled_layout:
            row_size = 8 if is_sf_8x4_layout else 128
            padded_row = (m + row_size - 1) // row_size * row_size
            padded_col = (k // sf_vec_size + 3) // 4 * 4
            out_sf_size = padded_row * padded_col
        else:
            out_sf_size = m * k // sf_vec_size
        out_sf = input.new_empty((out_sf_size,), dtype=torch.uint8)
        return out_val, out_sf

    _bsi_name = f"flashinfer_fp4_rt::block_scale_interleave_{arch_key}"

    @torch.library.custom_op(_bsi_name, mutates_args=())
    def _block_scale_interleave_op(unswizzled_sf: torch.Tensor) -> torch.Tensor:
        return jit_mod.block_scale_interleave_sm100(unswizzled_sf)

    @torch.library.register_fake(_bsi_name)
    def _block_scale_interleave_fake(unswizzled_sf: torch.Tensor) -> torch.Tensor:
        num_experts = unswizzled_sf.shape[0] if unswizzled_sf.dim() == 3 else 1
        padded_row = (unswizzled_sf.shape[-2] + 127) // 128 * 128
        padded_col = (unswizzled_sf.shape[-1] + 3) // 4 * 4
        return unswizzled_sf.new_empty(
            (num_experts * padded_row * padded_col,), dtype=unswizzled_sf.dtype
        )

    _fp4_quantize_fn = _fp4_quantize_op
    _block_scale_interleave_fn = _block_scale_interleave_op

'''

src = src.replace(marker, INJECTION + marker)
with open(path, 'w') as f:
    f.write(src)
print(f"Fix 9d: patched {path}")
