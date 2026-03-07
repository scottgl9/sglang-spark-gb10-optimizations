// ============================================================================
// NVIDIA FP4 Type Stubs for CUDA 13.0 CCCL Compatibility
// ============================================================================
// CUDA 13.0's CCCL headers reference __nv_fp4_e2m1 types for SM120/SM121
// but NVIDIA hasn't released the official type yet. This header provides
// minimal stubs to allow compilation.
//
// Format: 1 sign + 2 exponent + 1 mantissa bits (E2M1)
// ============================================================================

#ifndef NV_FP4_DUMMY_H
#define NV_FP4_DUMMY_H

#include <cuda_runtime.h>
#include <stdint.h>

// Only define if not already provided by CUDA toolkit
#ifndef __nv_fp4_e2m1

struct __align__(1) __nv_fp4_e2m1 {
    unsigned char __x;

    __host__ __device__ constexpr
    __nv_fp4_e2m1() : __x(0) {}

    __host__ __device__ constexpr
    __nv_fp4_e2m1(unsigned char val) : __x(val & 0x0F) {}

    __host__ __device__ __forceinline__
    operator float() const {
        unsigned char sign = (__x >> 3) & 0x1;
        unsigned char exp = (__x >> 1) & 0x3;
        unsigned char mantissa = __x & 0x1;

        float value;
        if (exp == 0) {
            value = (mantissa == 0) ? 0.0f : 0.25f;
        } else {
            float base = 1.0f + mantissa * 0.5f;
            float exponent_scale;
            switch (exp) {
                case 1: exponent_scale = 1.0f; break;
                case 2: exponent_scale = 2.0f; break;
                case 3: exponent_scale = 4.0f; break;
                default: exponent_scale = 1.0f; break;
            }
            value = base * exponent_scale;
        }
        return sign ? -value : value;
    }

    __host__ __device__ constexpr __forceinline__
    bool operator==(const __nv_fp4_e2m1& other) const {
        return __x == other.__x;
    }

    __host__ __device__ constexpr __forceinline__
    bool operator!=(const __nv_fp4_e2m1& other) const {
        return __x != other.__x;
    }
};

#ifndef __nv_fp4x2_storage_t
struct __align__(1) __nv_fp4x2_storage_t {
    unsigned char __x;

    __host__ __device__ constexpr
    __nv_fp4x2_storage_t() : __x(0) {}

    __host__ __device__ constexpr
    __nv_fp4x2_storage_t(unsigned char val) : __x(val) {}

    __host__ __device__ constexpr __forceinline__
    __nv_fp4x2_storage_t operator<<(int shift) const {
        return __nv_fp4x2_storage_t(__x << shift);
    }

    __host__ __device__ constexpr __forceinline__
    __nv_fp4x2_storage_t operator>>(int shift) const {
        return __nv_fp4x2_storage_t(__x >> shift);
    }

    __host__ __device__ constexpr __forceinline__
    __nv_fp4x2_storage_t operator|(const __nv_fp4x2_storage_t& other) const {
        return __nv_fp4x2_storage_t(__x | other.__x);
    }

    __host__ __device__ constexpr __forceinline__
    __nv_fp4x2_storage_t operator&(const __nv_fp4x2_storage_t& other) const {
        return __nv_fp4x2_storage_t(__x & other.__x);
    }

    __host__ __device__ constexpr __forceinline__
    operator unsigned short() const {
        return static_cast<unsigned short>(__x);
    }
};
#endif // __nv_fp4x2_storage_t

#endif // __nv_fp4_e2m1

#ifndef __NV_E2M1
#define __NV_E2M1 0
#endif

#endif // NV_FP4_DUMMY_H
