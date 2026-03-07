"""GB10 (SM121) FlashInfer compatibility patches.

SM121 (NVIDIA GB10) supports FP4 tensor core MMA but lacks the
``cvt.rn.satfinite.e2m1x2.f32`` PTX instruction for hardware float-to-E2M1
conversion.  FlashInfer's JIT-compiled kernels reference this instruction via
CUTLASS and TRT-LLM headers, so they must be patched before the first JIT
compilation on SM121.

All patches are idempotent; sentinel comments detect prior application.
"""

import logging
import os
import shutil
import site
from functools import cache

logger = logging.getLogger(__name__)


@cache
def _find_flashinfer_data_dir() -> str | None:
    """Return the FlashInfer data directory, or None if not found."""
    candidates: list[str] = []
    try:
        candidates += [
            os.path.join(sp, "flashinfer", "data")
            for sp in site.getsitepackages()
        ]
        candidates.append(
            os.path.join(site.getusersitepackages(), "flashinfer", "data")
        )
    except Exception:
        pass

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        for pyver in ("python3.12", "python3.11", "python3.13", "python3.10"):
            candidates.append(
                os.path.join(
                    venv, "lib", pyver, "site-packages", "flashinfer", "data"
                )
            )

    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def _patch_float_subbyte(data_dir: str) -> None:
    """Remove SM121A/F from CUDA_PTX_FP4FP6_CVT_ENABLED in float_subbyte.h."""
    target = os.path.join(
        data_dir, "cutlass", "include", "cutlass", "float_subbyte.h"
    )
    if not os.path.exists(target):
        return

    with open(target) as f:
        content = f.read()

    if "SM121 removed" in content:
        return

    patched = False
    for suffix in ("A", "F"):
        old = f"defined(CUTLASS_ARCH_MMA_SM121{suffix}_ENABLED)"
        if old in content:
            content = content.replace(f" || \\\n     {old}", "")
            content = content.replace(f"{old} || \\\n     ", "")
            content = content.replace(f" || {old}", "")
            patched = True

    if patched:
        content = content.replace(
            "#  define CUDA_PTX_FP4FP6_CVT_ENABLED 1",
            "/* SM121 removed: no cvt.rn.satfinite.e2m1x2.f32 on GB10 */\n"
            "#  define CUDA_PTX_FP4FP6_CVT_ENABLED 1",
            2,
        )
        with open(target, "w") as f:
            f.write(content)
        logger.debug("GB10 compat: patched %s", target)


def _patch_quantization_utils(data_dir: str) -> None:
    """Exclude SM121 from PTX E2M1 path in TRT-LLM quantization_utils.cuh."""
    target = os.path.join(
        data_dir,
        "csrc",
        "nv_internal",
        "tensorrt_llm",
        "kernels",
        "quantization_utils.cuh",
    )
    if not os.path.exists(target):
        return

    with open(target) as f:
        content = f.read()

    if "__CUDA_ARCH__ != 1210" in content:
        return

    old_guard = "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)"
    new_guard = (
        "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)"
        " && (__CUDA_ARCH__ != 1210)"
    )
    if old_guard in content:
        content = content.replace(old_guard, new_guard)
        with open(target, "w") as f:
            f.write(content)
        logger.debug("GB10 compat: patched %s", target)


def _patch_arch_condition(data_dir: str) -> None:
    """Allow SM121 in FlashInfer's arch_condition.h."""
    target = os.path.join(
        data_dir, "include", "flashinfer", "arch_condition.h"
    )
    if not os.path.exists(target):
        return

    with open(target) as f:
        content = f.read()

    if "GB10" in content or "__CUDA_ARCH__ == 1210" in content:
        return

    error_line = '#error "Compiling for SM90 or newer'
    if error_line not in content:
        return

    bypass = (
        "// GB10 (SM121): sm_121a is valid, bypass arch-family check\n"
        "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210\n"
        "  // SM121 detected - allow\n"
        "#else\n"
    )
    idx = content.find(error_line)
    block_start = content.rfind("#if", 0, idx)
    if block_start < 0:
        return
    endif_idx = content.find("#endif", idx)
    if endif_idx < 0:
        return
    endif_end = content.find("\n", endif_idx) + 1
    original_block = content[block_start:endif_end]
    patched_block = bypass + original_block + "#endif  // GB10\n"
    content = content[:block_start] + patched_block + content[endif_end:]
    with open(target, "w") as f:
        f.write(content)
    logger.debug("GB10 compat: patched %s", target)


def _copy_fp4_header(data_dir: str) -> None:
    """Copy nv_fp4_dummy.h to FlashInfer include dir for JIT compilation."""
    # Look for the header in sgl-kernel
    import sglang

    sglang_root = os.path.dirname(os.path.dirname(os.path.abspath(sglang.__file__)))
    src = os.path.join(sglang_root, "sgl-kernel", "csrc", "nv_fp4_dummy.h")
    if not os.path.exists(src):
        # Try relative to this file
        src = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..",
            "sgl-kernel", "csrc", "nv_fp4_dummy.h"
        )
        src = os.path.normpath(src)

    dst_dir = os.path.join(data_dir, "include", "flashinfer")
    if not os.path.isdir(dst_dir) or not os.path.exists(src):
        return

    dst = os.path.join(dst_dir, "nv_fp4_dummy.h")
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
        logger.debug("GB10 compat: copied %s -> %s", src, dst)


def _patch_trtllm_fused_moe_runtime_checks(data_dir: str) -> None:
    """Relax SM10x-only runtime checks in TRTLLM fused MoE to allow SM12x."""
    target = os.path.join(
        data_dir, "csrc", "trtllm_fused_moe_kernel_launcher.cu"
    )
    if not os.path.exists(target):
        return

    with open(target) as f:
        content = f.read()

    if "SM121 patched" in content:
        return

    patched = False

    old1 = (
        'TVM_FFI_ICHECK_EQ(major, 10) << "MoE kernel requires 10.x '
        'architecture. Current device has SM "\n'
        '                               << major << minor;'
    )
    new1 = (
        'TVM_FFI_ICHECK(major == 10 || major == 12) /* SM121 patched */ '
        '<< "MoE kernel requires 10.x or 12.x architecture. '
        'Current device has SM "\n'
        '                               << major << minor;'
    )
    if old1 in content:
        content = content.replace(old1, new1)
        patched = True

    old2 = (
        'TVM_FFI_ICHECK_EQ(std::get<0>(device_props), 10)\n'
        '        << "This kernel requires 10.x architecture. '
        'Current device has SM "\n'
        '        << std::get<0>(device_props) << std::get<1>(device_props);'
    )
    new2 = (
        'TVM_FFI_ICHECK(std::get<0>(device_props) == 10 || '
        'std::get<0>(device_props) == 12) /* SM121 patched */\n'
        '        << "This kernel requires 10.x or 12.x architecture. '
        'Current device has SM "\n'
        '        << std::get<0>(device_props) << std::get<1>(device_props);'
    )
    if old2 in content:
        content = content.replace(old2, new2)
        patched = True

    if patched:
        with open(target, "w") as f:
            f.write(content)
        logger.debug("GB10 compat: patched TRTLLM runtime checks in %s", target)


def _patch_trtllm_fused_moe_jit() -> None:
    """Add SM12x to TRTLLM fused MoE JIT supported_major_versions."""
    candidates: list[str] = []
    try:
        candidates += [
            os.path.join(sp, "flashinfer", "jit", "fused_moe.py")
            for sp in site.getsitepackages()
        ]
        candidates.append(
            os.path.join(
                site.getusersitepackages(), "flashinfer", "jit", "fused_moe.py"
            )
        )
    except Exception:
        pass

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        for pyver in ("python3.12", "python3.11", "python3.13", "python3.10"):
            candidates.append(
                os.path.join(
                    venv, "lib", pyver, "site-packages",
                    "flashinfer", "jit", "fused_moe.py"
                )
            )

    target = None
    for c in candidates:
        if os.path.isfile(c):
            target = c
            break
    if target is None:
        return

    with open(target) as f:
        content = f.read()

    if "SM121 patched" in content:
        return

    old = (
        "    # currently only support Blackwell\n"
        "    nvcc_flags = current_compilation_context.get_nvcc_flags_list(\n"
        "        supported_major_versions=[10]\n"
        "    )"
    )
    new = (
        "    # currently only support Blackwell\n"
        "    # SM121 patched: add major=12 for DGX Spark GB10\n"
        "    nvcc_flags = current_compilation_context.get_nvcc_flags_list(\n"
        "        supported_major_versions=[10, 12]\n"
        "    )"
    )
    if old in content:
        content = content.replace(old, new)
        with open(target, "w") as f:
            f.write(content)
        logger.debug("GB10 compat: patched TRTLLM JIT in %s", target)


def _clear_moe_jit_cache() -> None:
    """Clear FlashInfer MoE JIT cache to force recompilation."""
    cache_dir = os.path.expanduser("~/.cache/flashinfer")
    if not os.path.exists(cache_dir):
        return
    for root, dirs, _ in os.walk(cache_dir):
        for d in dirs:
            if "fused_moe" in d or "moe" in d:
                path = os.path.join(root, d)
                shutil.rmtree(path, ignore_errors=True)
                logger.debug("GB10 compat: cleared JIT cache %s", path)


@cache
def ensure_flashinfer_sm121_compat() -> None:
    """Apply all SM121 FlashInfer header patches (idempotent, runs once).

    Should be called at server startup when an SM121 device is detected
    and FlashInfer is available.
    """
    data_dir = _find_flashinfer_data_dir()
    if data_dir is None:
        logger.debug(
            "GB10 compat: FlashInfer data dir not found, skipping patches"
        )
        return

    logger.info(
        "GB10 (SM121) detected - applying FlashInfer compatibility patches"
    )
    _patch_float_subbyte(data_dir)
    _patch_quantization_utils(data_dir)
    _patch_arch_condition(data_dir)
    _copy_fp4_header(data_dir)
    _patch_trtllm_fused_moe_runtime_checks(data_dir)
    _patch_trtllm_fused_moe_jit()
    _clear_moe_jit_cache()
    logger.info("GB10 compat: FlashInfer patches complete")
