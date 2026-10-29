// SPDX-License-Identifier: MIT
// Adapted from llama.cpp ggml/src/ggml-cuda (MIT, The ggml authors); see
// VENDOR.md.
#pragma once

#define HIP_DISABLE_WARP_SYNC_BUILTINS 1
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#if defined(GGML_HIP_ROCWMMA_FATTN)
#include <rocwmma/rocwmma-version.hpp>
#endif


#define __shfl_sync(mask, var, laneMask, width) __shfl(var, laneMask, width)
#define __shfl_down_sync(mask, var, delta, ...) __shfl_down(var, delta, ##__VA_ARGS__)
#define __shfl_up_sync(mask, var, laneMask, width) __shfl_up(var, laneMask, width)
#define __shfl_xor_sync(mask, var, laneMask, width) __shfl_xor(var, laneMask, width)
#define __all_sync(mask, var) __all(var)
#define __any_sync(mask, var) __any(var)
#define __trap() do { abort(); __builtin_unreachable(); } while (0)


#if !defined(__HIP_PLATFORM_AMD__)
#error "The HIP backend supports only AMD targets"
#endif


#if defined(__gfx900__) || defined(__gfx906__)
#define GCN5
#endif

#if defined(__gfx803__)
#define GCN4
#endif

#if defined(GCN5) || defined(GCN4)
#define GCN
#endif

#if defined(__gfx950__)
#define CDNA4
#endif

#if defined(__gfx942__)
#define CDNA3
#endif

#if defined(__gfx90a__)
#define CDNA2
#endif

#if defined(__gfx908__)
#define CDNA1
#endif

#if defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)
#define CDNA
#endif

#if defined(__GFX12__)
#define RDNA4
#endif

#if defined(__GFX11__)
#define RDNA3
#endif

#if defined(__gfx1150__) || defined(__gfx1151__)
#define RDNA3_5
#endif

#if defined(RDNA3) && !defined(RDNA3_5)
#define RDNA3_0
#endif

#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#define RDNA2
#endif

#if defined(__gfx1010__) || defined(__gfx1012__)
#define RDNA1
#endif

#if defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)
#define RDNA
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

typedef __hip_bfloat16 nv_bfloat16;
typedef __hip_bfloat162 nv_bfloat162;

#if HIP_VERSION >= 60200000
#include <hip/hip_fp8.h>
typedef __hip_fp8_e4m3 __nv_fp8_e4m3;
#define FP8_AVAILABLE
#endif

typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));
typedef uint8_t uint8x4_t __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ int __vsubss4(const int a, const int b) {
    const int8x4_t va = reinterpret_cast<const int8x4_t &>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t &>(b);
#if __has_builtin(__builtin_elementwise_sub_sat)
    const int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    return reinterpret_cast<const int &>(c);
#else
    int8x4_t c;
    int16_t tmp;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        if (tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        if (tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    return reinterpret_cast<int &>(c);
#endif
}

static __device__ __forceinline__ int __vsub4(const int a, const int b) {
    return __vsubss4(a, b);
}

static __device__ __forceinline__ unsigned int __vcmpeq4(unsigned int a, unsigned int b) {
    const uint8x4_t &va = reinterpret_cast<const uint8x4_t &>(a);
    const uint8x4_t &vb = reinterpret_cast<const uint8x4_t &>(b);
    unsigned int c;
    uint8x4_t &vc = reinterpret_cast<uint8x4_t &>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    return c;
}

static __device__ __forceinline__ unsigned int __vcmpne4(unsigned int a, unsigned int b) {
    const uint8x4_t &va = reinterpret_cast<const uint8x4_t &>(a);
    const uint8x4_t &vb = reinterpret_cast<const uint8x4_t &>(b);
    unsigned int c;
    uint8x4_t &vc = reinterpret_cast<uint8x4_t &>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0x00 : 0xff;
    }
    return c;
}
