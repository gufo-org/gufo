#pragma once

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-hip.h"

#include <cstdint>
#include <memory>

#define GGML_COMMON_DECL_HIP
#define GGML_COMMON_IMPL_HIP
#include "ggml-common.h"

#include <array>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "vendors/hip.h"

#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#define WARP_SIZE 32

#define GGML_HIP_CC_OFFSET_AMD      0x1000000

// AMD
// GCN/CDNA, wave size is 64
#define GGML_HIP_CC_GCN4       (GGML_HIP_CC_OFFSET_AMD + 0x803)  // Tonga, Fiji, Polaris, minimum for fast fp16
#define GGML_HIP_CC_VEGA       (GGML_HIP_CC_OFFSET_AMD + 0x900)  // Vega56/64, minimum for fp16 dual issue
#define GGML_HIP_CC_VEGA20     (GGML_HIP_CC_OFFSET_AMD + 0x906)  // MI50/Radeon VII, minimum for dp4a
#define GGML_HIP_CC_CDNA1      (GGML_HIP_CC_OFFSET_AMD + 0x908)  // MI100, minimum for MFMA, acc registers
#define GGML_HIP_CC_CDNA2      (GGML_HIP_CC_OFFSET_AMD + 0x90a)  // MI210 (gfx90a), minimum acc register renaming
#define GGML_HIP_CC_CDNA3      (GGML_HIP_CC_OFFSET_AMD + 0x942)  // MI300
#define GGML_HIP_CC_CDNA4      (GGML_HIP_CC_OFFSET_AMD + 0x950)  // MI350X/MI355X

// RDNA removes MFMA, dp4a, xnack, acc registers, wave size is 32
#define GGML_HIP_CC_RDNA1      (GGML_HIP_CC_OFFSET_AMD + 0x1010) // RX 5000
#define GGML_HIP_CC_RDNA2      (GGML_HIP_CC_OFFSET_AMD + 0x1030) // RX 6000, minimum for dp4a
#define GGML_HIP_CC_RDNA3      (GGML_HIP_CC_OFFSET_AMD + 0x1100) // RX 7000, minimum for WMMA
#define GGML_HIP_CC_RDNA3_5    (GGML_HIP_CC_OFFSET_AMD + 0x1150) // AI 370, AI Max 395 laptops.
#define GGML_HIP_CC_RDNA4      (GGML_HIP_CC_OFFSET_AMD + 0x1200) // RX 9000

#define GGML_HIP_CC_IS_AMD(cc)     (cc >= GGML_HIP_CC_OFFSET_AMD)
#define GGML_HIP_CC_IS_RDNA(cc)    (cc >= GGML_HIP_CC_RDNA1)
#define GGML_HIP_CC_IS_RDNA1(cc)   (cc >= GGML_HIP_CC_RDNA1 && cc < GGML_HIP_CC_RDNA2)
#define GGML_HIP_CC_IS_RDNA2(cc)   (cc >= GGML_HIP_CC_RDNA2 && cc < GGML_HIP_CC_RDNA3)
#define GGML_HIP_CC_IS_RDNA3_0(cc) (cc >= GGML_HIP_CC_RDNA3 && cc < GGML_HIP_CC_RDNA3_5)
#define GGML_HIP_CC_IS_RDNA3_5(cc) (cc >= GGML_HIP_CC_RDNA3_5 && cc < GGML_HIP_CC_RDNA4)
#define GGML_HIP_CC_IS_RDNA3(cc)   (GGML_HIP_CC_IS_RDNA3_0(cc) || GGML_HIP_CC_IS_RDNA3_5(cc))
#define GGML_HIP_CC_IS_RDNA4(cc)   (cc >= GGML_HIP_CC_RDNA4)
#define GGML_HIP_CC_IS_GCN(cc)     (cc > GGML_HIP_CC_OFFSET_AMD && cc < GGML_HIP_CC_CDNA1)
#define GGML_HIP_CC_IS_CDNA(cc)    (cc >= GGML_HIP_CC_CDNA1 && cc < GGML_HIP_CC_RDNA1)
#define GGML_HIP_CC_IS_CDNA1(cc)   (cc >= GGML_HIP_CC_CDNA1 && cc < GGML_HIP_CC_CDNA2)
#define GGML_HIP_CC_IS_CDNA2(cc)   (cc >= GGML_HIP_CC_CDNA2 && cc < GGML_HIP_CC_CDNA3)
#define GGML_HIP_CC_IS_CDNA3(cc)   (cc >= GGML_HIP_CC_CDNA3 && cc < GGML_HIP_CC_CDNA4)
#define GGML_HIP_CC_IS_CDNA4(cc)   (cc >= GGML_HIP_CC_CDNA4 && cc < GGML_HIP_CC_RDNA1)

static int ggml_hip_highest_compiled_arch(const int arch) {
    return arch;
}

// ---------------------------------------------------------------------------------------------------------

#define MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#define GGML_HIP_MAX_STREAMS 8

[[noreturn]]
void ggml_hip_error(const char * stmt, const char * func, const char * file, int line, const char * msg);

#define HIP_CHECK_GEN(err, success, error_fn)                                      \
     do {                                                                           \
        auto err_ = (err);                                                          \
        if (err_ != (success)) {                                                    \
            ggml_hip_error(#err, __func__, __FILE__, __LINE__, error_fn(err_));    \
        }                                                                           \
    } while (0)

#define HIP_CHECK(err) HIP_CHECK_GEN(err, hipSuccess, hipGetErrorString)

    static const char * hipblas_get_error_str(const hipblasStatus_t err) {
        switch (err) {
            case HIPBLAS_STATUS_SUCCESS: return "HIPBLAS_STATUS_SUCCESS";
            case HIPBLAS_STATUS_NOT_INITIALIZED: return "HIPBLAS_STATUS_NOT_INITIALIZED";
            case HIPBLAS_STATUS_ALLOC_FAILED: return "HIPBLAS_STATUS_ALLOC_FAILED";
            case HIPBLAS_STATUS_INVALID_VALUE: return "HIPBLAS_STATUS_INVALID_VALUE";
            case HIPBLAS_STATUS_ARCH_MISMATCH: return "HIPBLAS_STATUS_ARCH_MISMATCH";
            case HIPBLAS_STATUS_MAPPING_ERROR: return "HIPBLAS_STATUS_MAPPING_ERROR";
            case HIPBLAS_STATUS_EXECUTION_FAILED: return "HIPBLAS_STATUS_EXECUTION_FAILED";
            case HIPBLAS_STATUS_INTERNAL_ERROR: return "HIPBLAS_STATUS_INTERNAL_ERROR";
            case HIPBLAS_STATUS_NOT_SUPPORTED: return "HIPBLAS_STATUS_NOT_SUPPORTED";
            default: return "unknown error";
        }
    }

#define HIPBLAS_CHECK(err) HIP_CHECK_GEN(err, HIPBLAS_STATUS_SUCCESS, hipblas_get_error_str)

#    define HIP_SET_SHARED_MEMORY_LIMIT(kernel, nbytes) \
        do {                                             \
            GGML_UNUSED(nbytes);                         \
        } while (0)

#define GGML_HIP_ASSUME(x)

#define FP16_AVAILABLE

#define FAST_FP16_AVAILABLE

#if defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)
#define AMD_MFMA_AVAILABLE
#endif // defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)

#if (defined(RDNA4) || defined(RDNA3))
#define AMD_WMMA_AVAILABLE
#endif // defined(RDNA4)

static bool amd_mfma_available(const int cc) {
#if !defined(GGML_HIP_NO_MMQ_MFMA)
    return GGML_HIP_CC_IS_CDNA(cc);
#else
    return false;
#endif //!defined(GGML_HIP_NO_MMQ_MFMA)
}

static bool amd_wmma_available(const int cc) {
    return (GGML_HIP_CC_IS_RDNA4(cc) || GGML_HIP_CC_IS_RDNA3(cc));
}

static constexpr __device__ int ggml_hip_get_physical_warp_size() {
#if (defined(__GFX9__) || defined(__GFX8__))
    return 64;
#else
    return 32;
#endif // (defined(__GFX9__) || defined(__GFX8__))
}

// Maximum number of bytes that can be copied in a single instruction.
static constexpr __device__ int ggml_hip_get_max_cpy_bytes() {
    return 16;
}


[[noreturn]]
static __device__ void no_device_code(
    const char * file_name, const int line, const char * function_name, const char * arch) {

    printf("%s:%d: ERROR: HIP kernel %s has no device code compatible with HIP arch %s.\n",
           file_name, line, function_name, arch);
    __trap();

    GGML_UNUSED(no_device_code); // suppress unused function warning

}

#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, "gfx1151")

// The compiler is always able to unroll loops if they contain continue expressions.
// In such cases loop unrolling can still be achieved via recursion:
template <int n>
struct ggml_hip_unroll {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        ggml_hip_unroll<n - 1>{}(f, args...);
    }
};

template <>
struct ggml_hip_unroll<1> {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};

// Lane i receives lane (i ^ offset)'s value, as __shfl_xor does for a full
// wave. On gfx1151 this is a DPP row_xmask (offsets below 16) or
// v_permlanex16 (16): a register move, not a round trip through the LDS
// crossbar (ds_bpermute). Only the transport differs, so sums stay bitwise.
template<int offset>
static __device__ __forceinline__ int xor_lane(int x) {
#if defined(__gfx1151__)
    if constexpr (offset < 16) {
        return __builtin_amdgcn_update_dpp(0, x, 0x160 | offset, 0xF, 0xF, false);
    } else if constexpr (offset == 16) {
        return __builtin_amdgcn_permlanex16(x, x, 0x76543210, 0xfedcba98, false, false);
    }
#endif
    return __shfl_xor(x, offset, WARP_SIZE);
}

template<int offset>
static __device__ __forceinline__ float xor_lane(float x) {
    return __builtin_bit_cast(float, xor_lane<offset>(__builtin_bit_cast(int, x)));
}

// The butterfly in the shuffle loop's order (width/2 down to 1).
template<int width, typename T>
static __device__ __forceinline__ T xor_reduce_sum(T x) {
    static_assert(width <= 32, "xor_lane covers offsets up to 16");
    if constexpr (width >= 32) x += xor_lane<16>(x);
    if constexpr (width >= 16) x += xor_lane<8>(x);
    if constexpr (width >= 8) x += xor_lane<4>(x);
    if constexpr (width >= 4) x += xor_lane<2>(x);
    if constexpr (width >= 2) x += xor_lane<1>(x);
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_sum(int x) {
    return xor_reduce_sum<width>(x);
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
    return xor_reduce_sum<width>(x);
}

// Reduce-scatter levels for n sums: the first levels of the wave butterfly
// each halve the sums a lane carries.
constexpr int ScatterLevels(int n) { return n <= 1 ? 0 : n <= 2 ? 1 : n <= 4 ? 2 : 3; }

// One butterfly level that keeps half of `count` sums: the lane on the
// `offset` side keeps the upper half and sends the lower, its partner the
// reverse, so each kept sum adds the partner's copy of the same sum.
template<int offset, int count>
static __device__ __forceinline__ void scatter_level(float* s, int lane) {
    const bool upper = (lane & offset) != 0;
#pragma unroll
    for (int i = 0; i < count / 2; ++i) {
        const float keep = upper ? s[count / 2 + i] : s[i];
        const float send = upper ? s[i] : s[count / 2 + i];
        s[i] = keep + xor_lane<offset>(send);
    }
}

// warp_reduce_sum<32> over n sums at once: the same pairs (16, 8, 4, 2, 1)
// and operands as the full butterfly, which leaves every lane with identical
// bits, but lane l only finishes sum l >> (5 - ScatterLevels(n)).
template<int n>
static __device__ __forceinline__ float scatter_reduce_sum(const float (&v)[n], int lane) {
    constexpr int levels = ScatterLevels(n);
    float s[1 << levels];
#pragma unroll
    for (int j = 0; j < (1 << levels); ++j)
        s[j] = j < n ? v[j] : 0.0f;
    if constexpr (levels == 3) {
        scatter_level<16, 8>(s, lane);
        scatter_level<8, 4>(s, lane);
        scatter_level<4, 2>(s, lane);
    } else if constexpr (levels == 2) {
        scatter_level<16, 4>(s, lane);
        scatter_level<8, 2>(s, lane);
        s[0] += xor_lane<4>(s[0]);
    } else if constexpr (levels == 1) {
        scatter_level<16, 2>(s, lane);
        s[0] += xor_lane<8>(s[0]);
        s[0] += xor_lane<4>(s[0]);
    } else {
        s[0] += xor_lane<16>(s[0]);
        s[0] += xor_lane<8>(s[0]);
        s[0] += xor_lane<4>(s[0]);
    }
    s[0] += xor_lane<2>(s[0]);
    s[0] += xor_lane<1>(s[0]);
    return s[0];
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        a.x += __shfl_xor(a.x, offset, width);
        a.y += __shfl_xor(a.y, offset, width);
    }
    return a;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_reduce_sum(half2 a) {
#ifdef FP16_AVAILABLE
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        a = __hadd2(a, __shfl_xor(a, offset, width));
    }
    return a;

#else
    NO_DEVICE_CODE;
    return a;
#endif // FP16_AVAILABLE
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_all(int x) {
    if (width == ggml_hip_get_physical_warp_size()) {
        return __all(x);
    } else {
#pragma unroll
        for (int offset = width/2; offset > 0; offset >>= 1) {
            x = __shfl_xor(x, offset, width) && x;
        }
        return x;
    }
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_any(int x) {
    if (width == ggml_hip_get_physical_warp_size()) {
        return __any(x);
    } else {
#pragma unroll
        for (int offset = width/2; offset > 0; offset >>= 1) {
            x = __shfl_xor(x, offset, width) || x;
        }
        return x;
    }
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor(x, offset, width));
    }
    return x;
}

template<typename T, int width = WARP_SIZE>
static __device__ __forceinline__ T warp_prefix_inclusive_sum(T x) {
    const int lane_id = threadIdx.x % width;
#pragma unroll
    for (int offset = 1; offset < width; offset <<= 1) {
        const T t = __shfl_up(x, offset, width);
        if (lane_id >= offset) {
            x += t;
        }
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float2 warp_prefix_inclusive_sum(float2 a) {
    const int lane_id = threadIdx.x % width;
#pragma unroll
    for (int offset = 1; offset < width; offset <<= 1) {
        const float t_x = __shfl_up(a.x, offset, width);
        const float t_y = __shfl_up(a.y, offset, width);
        if (lane_id >= offset) {
            a.x += t_x;
            a.y += t_y;
        }
    }
    return a;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_prefix_inclusive_sum(half2 a) {
#ifdef FP16_AVAILABLE
    const int lane_id = threadIdx.x % width;
#pragma unroll
    for (int offset = 1; offset < width; offset <<= 1) {
        const half2 t = __shfl_up(a, offset, width);
        if (lane_id >= offset) {
            a = __hadd2(a, t);
        }
    }
    return a;

#else
    NO_DEVICE_CODE;
    return a;
#endif // FP16_AVAILABLE
}

enum class block_reduce_method {
    MAX,
    SUM,
};

template<block_reduce_method method_t, typename T>
struct block_reduce_policy;

template <typename T, typename... Ts>
inline constexpr bool is_any = (std::is_same_v<T, Ts> || ...);

template<typename...>
inline constexpr bool ggml_hip_dependent_false_v = false;

template <typename T> struct block_reduce_policy<block_reduce_method::SUM, T> {
    static __device__ T reduce(T val) {
        if constexpr(is_any<T, float, float2, half2, int>) {
            return warp_reduce_sum(val);
        } else {
            static_assert(ggml_hip_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    static __device__ T sentinel() {
        if constexpr (std::is_same_v<T, float>) {
            return 0.0f;
        } else if constexpr (std::is_same_v<T, float2>) {
            return make_float2(0.0f, 0.0f);
        } else if constexpr (std::is_same_v<T, half2>) {
            return make_half2(0.0f, 0.0f);
        } else if constexpr (std::is_same_v<T, int>) {
            return 0;
        } else {
            static_assert(ggml_hip_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

template <typename T> struct block_reduce_policy<block_reduce_method::MAX, T> {
    static __device__ T reduce(T val) {
        if constexpr (is_any<T, float, half2>) {
            return warp_reduce_max(val);
        } else {
            static_assert(ggml_hip_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    static __device__ T sentinel() {
        if constexpr (std::is_same_v<T, float>) {
            return -INFINITY;
        } else if constexpr (std::is_same_v<T, half2>) {
            return make_half2(-INFINITY, -INFINITY);
        } else {
            static_assert(ggml_hip_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};

template <block_reduce_method reduce_method_t, const unsigned int block_size_template = 0, typename T>
static __device__ T block_reduce(T val, T * shared_vals) {
    val                           = block_reduce_policy<reduce_method_t, T>::reduce(val);
    const unsigned int block_size = block_size_template == 0 ? blockDim.x : block_size_template;
    if (block_size > WARP_SIZE) {
        assert((block_size <= 1024) && (block_size % WARP_SIZE) == 0);
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        __syncthreads();
        val = block_reduce_policy<reduce_method_t, T>::sentinel();
        if (lane_id < (static_cast<int>(block_size) / WARP_SIZE)) {
            val = shared_vals[lane_id];
        }
        return block_reduce_policy<reduce_method_t, T>::reduce(val);
    }

    return val;
}

static __device__ __forceinline__ half ggml_hip_hmax(const half a, const half b) {
#ifdef FP16_AVAILABLE

    return __hmax(a, b);

#else
   NO_DEVICE_CODE;
   GGML_UNUSED(b);
   return a;
#endif // FP16_AVAILABLE
}

static __device__ __forceinline__ half2 ggml_hip_hmax2(const half2 a, const half2 b) {
    return half2(__hmax(a.x, b.x), __hmax(a.y, b.y));
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_reduce_max(half2 x) {
#pragma unroll
   for (int offset = width/2; offset > 0; offset >>= 1) {
       x = ggml_hip_hmax2(x, __shfl_xor(x, offset, width));
   }
   return x;
}

static __device__ __forceinline__ uint32_t __hgt2_mask(const half2 a, const half2 b) {
    const uint32_t mask_low  = 0x0000FFFF * (float( __low2half(a)) > float( __low2half(b)));
    const uint32_t mask_high = 0xFFFF0000 * (float(__high2half(a)) > float(__high2half(b)));
    return mask_low | mask_high;
}

static __device__ __forceinline__ int ggml_hip_dp4a(const int a, const int b, int c) {
#if defined(CDNA) || defined(RDNA2) || defined(__gfx906__)
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(RDNA3) || defined(RDNA4)
    c = __builtin_amdgcn_sudot4( true, a, true, b, c, false);
#elif defined(RDNA1) || defined(__gfx900__)
    int tmp1;
    int tmp2;
    asm("\n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        "
        : "+v"(c), "=&v"(tmp1), "=&v"(tmp2)
        : "v"(a), "v"(b)
    );
#else
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
    c += va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
#endif
    return c;

}

static __device__ __forceinline__ void ggml_hip_mad(float & acc, const float v, const float u) {
    acc += v*u;
}

static __device__ __forceinline__ void ggml_hip_mad(float & acc, const float2 v, const float2 u) {
    acc += v.x*u.x;
    acc += v.y*u.y;
}

#if (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(__gfx906__) || defined(CDNA))
#define V_DOT2_F32_F16_AVAILABLE
#endif // (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(__gfx906__) || defined(CDNA))

static __device__ __forceinline__ void ggml_hip_mad(float & acc, const half2 v, const half2 u) {
#ifdef V_DOT2_F32_F16_AVAILABLE
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(v), "v"(u));
#else
#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(v*u);
    acc += tmp.x + tmp.y;
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    acc += tmpv.x * tmpu.x;
    acc += tmpv.y * tmpu.y;
#endif // FAST_FP16_AVAILABLE
#endif // V_DOT2_F32_F16_AVAILABLE
}

static __device__ __forceinline__ void ggml_hip_mad(half2 & acc, const half2 v, const half2 u) {
#ifdef FAST_FP16_AVAILABLE
    acc += v*u;
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    float2 tmpacc = __half22float2(acc);
    tmpacc.x += tmpv.x * tmpu.x;
    tmpacc.y += tmpv.y * tmpu.y;
    acc = make_half2(tmpacc.x, tmpacc.y);
#endif // FAST_FP16_AVAILABLE
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes, especially on AMD.
// Important: do not use this function if dst and src both point at registers.
//     Due to the strict aliasing rule the compiler can do incorrect optimizations if src and dst have different types.
//     The function is intended for copies between registers and SRAM/VRAM to make the compiler emit the right instructions.
//     If dst and src point at different address spaces then they are guaranteed to not be aliased.
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_hip_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    static_assert(
        nbytes <= ggml_hip_get_max_cpy_bytes() || alignment == 0,
        "You are misusing the alignment parameter for ggml_hip_memcpy_1. "
        "The intent is for the parameter is only as a workaround if either one of the pointers is not properly aligned. "
        "If you use it to do more bytes per copy than ggml_hip_max_cpy_bytes() the reads and writes may not be coalesced. "
        "Call ggml_hip_memcpy_1 in a loop instead.");
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

static __device__ __forceinline__ float ggml_hip_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

static __device__ __forceinline__ float ggml_hip_ue4m3_to_fp32(uint8_t x) {
#if defined(CDNA3) && defined(FP8_AVAILABLE) && HIP_VERSION >= 60200000
    // ROCm does not support fp8 in software on devices with fp8 hardware,
    // but CDNA3 supports only e4m3_fnuz (no inf).
    const uint32_t bits = x * (x != 0x7F && x != 0xFF); // Convert NaN to 0.0f to match CPU implementation.
    const __hip_fp8_e4m3_fnuz xf = *reinterpret_cast<const __hip_fp8_e4m3_fnuz *>(&bits);
    return static_cast<float>(xf) / 2;
#else
    if (x == 0 || (x == 0x7F && x != 0xFF)) { // Convert NaN to 0.0f
        return 0.0f;
    }
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = ldexpf((float) man, -9);
    } else {
        raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    }
    return static_cast<float>(raw / 2);
#endif // defined(CDNA3) && defined(FP8_AVAILABLE) && HIP_VERSION >= 60200000
}

static __device__ __forceinline__ uint8_t ggml_hip_fp32_to_ue4m3(float x) {
     NO_DEVICE_CODE; // Used only for NVFP4 Scales for Activations, only for Blackwell
}

__device__ __forceinline__ uint8_t ggml_hip_float_to_fp4_e2m1(float x, float e) {
    const uint8_t sign_bit = (x < 0.0f) << 3;
    float         ax       = fabsf(x) * e;

    // Positive LUT
    static constexpr float pos_lut[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

    int   best_i   = 0;
    float best_err = fabsf(ax - pos_lut[0]);

#pragma unroll
    for (int i = 1; i < 8; ++i) {
        const float err = fabsf(ax - pos_lut[i]);
        if (err < best_err) {
            best_err = err;
            best_i   = i;
        }
    }

    return static_cast<uint8_t>(best_i | sign_bit);
}

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
static const uint3 init_fastdiv_values(uint64_t d_64) {
    GGML_ASSERT(d_64 != 0);
    GGML_ASSERT(d_64 <= std::numeric_limits<uint32_t>::max());

    uint32_t d = (uint32_t)d_64;

    // compute L = ceil(log2(d));
    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    // pack divisor as well to reduce error surface
    return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    // expects fastdiv_values to contain <mp, L, divisor> in <x, y, z>
    // fastdiv_values.z is unused and optimized away by the compiler.
    // Compute high 32 bits of n * mp
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    // add n, apply bit shift
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
}

// Calculate both division and modulo at once, returns <n/divisor, n%divisor>
static __device__ __forceinline__ uint2 fast_div_modulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z;
    return make_uint2(div_val, mod_val);
}

typedef void (*dequantize_kernel_t)(const void * vx, const int64_t ib, const int iqs, float2 & v);

static __device__ __forceinline__ float get_alibi_slope(
    const float max_bias, const uint32_t h, const uint32_t n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return powf(base, exph);
}

template <ggml_type type>
struct ggml_hip_type_traits;

template<>
struct ggml_hip_type_traits<GGML_TYPE_F16> {
    static constexpr int qk = 1;
    static constexpr int qr = 1;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_Q1_0> {
    static constexpr int qk = QK1_0;
    static constexpr int qr = QR1_0;
    static constexpr int qi = QI1_0;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_Q4_0> {
    static constexpr int qk = QK4_0;
    static constexpr int qr = QR4_0;
    static constexpr int qi = QI4_0;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_Q4_1> {
    static constexpr int qk = QK4_1;
    static constexpr int qr = QR4_1;
    static constexpr int qi = QI4_1;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_Q5_0> {
    static constexpr int qk = QK5_0;
    static constexpr int qr = QR5_0;
    static constexpr int qi = QI5_0;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_Q5_1> {
    static constexpr int qk = QK5_1;
    static constexpr int qr = QR5_1;
    static constexpr int qi = QI5_1;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_Q8_0> {
    static constexpr int qk = QK8_0;
    static constexpr int qr = QR8_0;
    static constexpr int qi = QI8_0;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_MXFP4> {
    static constexpr int qk = QK_MXFP4;
    static constexpr int qr = QR_MXFP4;
    static constexpr int qi = QI_MXFP4;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_NVFP4> {
    static constexpr int qk = QK_NVFP4;
    static constexpr int qr = QR_NVFP4;
    static constexpr int qi = QI_NVFP4;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_Q2_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_K;
    static constexpr int qi = QI2_K;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_Q3_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_K;
    static constexpr int qi = QI3_K;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_Q4_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_K;
    static constexpr int qi = QI4_K;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_Q5_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR5_K;
    static constexpr int qi = QI5_K;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_Q6_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR6_K;
    static constexpr int qi = QI6_K;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_IQ2_XXS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_XXS;
    static constexpr int qi = QI2_XXS;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_IQ2_XS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_XS;
    static constexpr int qi = QI2_XS;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_IQ2_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_S;
    static constexpr int qi = QI2_S;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_IQ3_XXS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_XXS;
    static constexpr int qi = QI3_XXS;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_IQ1_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR1_S;
    static constexpr int qi = QI1_S;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_IQ1_M> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR1_M;
    static constexpr int qi = QI1_M;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_IQ4_NL> {
    static constexpr int qk = QK4_NL;
    static constexpr int qr = QR4_NL;
    static constexpr int qi = QI4_NL;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_IQ4_XS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_hip_type_traits<GGML_TYPE_IQ3_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_S;
    static constexpr int qi = QI3_S;
};

//////////////////////

struct ggml_hip_device_info {
    int device_count;

    struct hip_device_info {
        int     cc;                             // compute capability
        int     nsm;                            // number of streaming multiprocessors
        size_t  smpb;                           // max. shared memory per block
        size_t  smpbo;                          // max. shared memory per block (with opt-in)
        bool    integrated;                     // Device is integrated as opposed to discrete
        bool    vmm;                            // virtual memory support
        size_t  vmm_granularity;                // granularity of virtual memory
        size_t  total_vram;
        int     warp_size;                      // Number of threads in a dispatch
        bool    supports_cooperative_launch;    // whether cooperative launch is supported
    };

    hip_device_info devices[GGML_HIP_MAX_DEVICES] = {};

    std::array<float, GGML_HIP_MAX_DEVICES> default_tensor_split = {};
};

const ggml_hip_device_info & ggml_hip_info();

void ggml_hip_set_device(int device);
int ggml_hip_get_device();

struct ggml_hip_pool {
    virtual ~ggml_hip_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

template<typename T>
struct ggml_hip_pool_alloc {
    ggml_hip_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_hip_pool_alloc() = default;

    explicit ggml_hip_pool_alloc(ggml_hip_pool & pool) : pool(&pool) {
    }

    ggml_hip_pool_alloc(ggml_hip_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_hip_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T * alloc(ggml_hip_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T * get() {
        return ptr;
    }

    ggml_hip_pool_alloc(const ggml_hip_pool_alloc &) = delete;
    ggml_hip_pool_alloc(ggml_hip_pool_alloc &&) = delete;
    ggml_hip_pool_alloc& operator=(const ggml_hip_pool_alloc &) = delete;
    ggml_hip_pool_alloc& operator=(ggml_hip_pool_alloc &&) = delete;
};


// backend interface

struct ggml_tensor_extra_gpu {
    void * data_device[GGML_HIP_MAX_DEVICES]; // 1 pointer for each device for split tensors
    hipEvent_t events[GGML_HIP_MAX_DEVICES][GGML_HIP_MAX_STREAMS]; // events for synchronizing multiple GPUs
};


struct ggml_hip_graph {
};

struct ggml_hip_concurrent_event {
    std::vector<hipEvent_t> join_events;
    hipEvent_t              fork_event = nullptr;

    int                                          n_streams = 0;
    std::unordered_map<const ggml_tensor *, int> stream_mapping;

    // Original order of nodes in this concurrent region (before interleaving)
    // Used to restore grouping for fusion within streams
    std::vector<const ggml_tensor *> original_order;

    const ggml_tensor * join_node;

    ggml_hip_concurrent_event() = default;

    ggml_hip_concurrent_event(const ggml_hip_concurrent_event &) = delete;
    ggml_hip_concurrent_event & operator=(const ggml_hip_concurrent_event &) = delete;

    explicit ggml_hip_concurrent_event(int n_streams) : n_streams(n_streams) {
        join_events.resize(n_streams);

        for (size_t i = 0; i < join_events.size(); ++i) {
            HIP_CHECK(hipEventCreateWithFlags(&join_events[i], hipEventDisableTiming));
        }

        HIP_CHECK(hipEventCreateWithFlags(&fork_event, hipEventDisableTiming));
    }

    ggml_hip_concurrent_event(ggml_hip_concurrent_event && other) noexcept
    : join_events(std::move(other.join_events))
    , fork_event(other.fork_event)
    , n_streams(other.n_streams)
    , stream_mapping(std::move(other.stream_mapping))
    , original_order(std::move(other.original_order))
    , join_node(other.join_node) {
        other.fork_event = nullptr;
    }

    // 1. check if any branches write to overlapping memory ranges (except the join node)
    // 2. check whether all srcs are either within the branch or outside the nodes covered by ggml_hip_concurrent_event
    // we assume all nodes have the same buffer
    bool is_valid() const {
        std::vector<std::vector<std::pair<int64_t, int64_t>>> write_ranges;
        write_ranges.resize(n_streams);

        // get join_node's memory range to exclude from overlap checking.
        // multiple nodes can use join_node's buffer; we synchronize on the join node.
        const ggml_tensor * join_t     = join_node->view_src ? join_node->view_src : join_node;
        const int64_t       join_start = (int64_t) join_t->data;
        const int64_t       join_end   = join_start + ggml_nbytes(join_t);

        for (const auto & [tensor, stream] : stream_mapping) {
            const ggml_tensor * t = tensor->view_src ? tensor->view_src : tensor;
            const int64_t       t_start = (int64_t) t->data;
            const int64_t       t_end   = t_start + ggml_nbytes(t);

            // skip tensors that overlap with join_node's buffer.
            if ((t_start <= join_start && join_start < t_end) || (join_start <= t_start && t_start < join_end)) {
                continue;
            }

            // concurrent streams begin from 1
            write_ranges[stream - 1].emplace_back(t_start, t_end);
        }

        for (int i = 0; i < n_streams; ++i) {
            // sorts first by start then by end of write range
            std::sort(write_ranges[i].begin(), write_ranges[i].end());
        }

        bool writes_overlap = false;
        bool dependent_srcs = false;
        for (const auto & [tensor, stream] : stream_mapping) {
            const ggml_tensor * t = tensor->view_src ? tensor->view_src : tensor;
            const int64_t       t_start = (int64_t) t->data;
            const int64_t       t_end   = t_start + ggml_nbytes(t);

            // skip tensors that overlap with join_node's buffer
            if ((t_start <= join_start && join_start < t_end) || (join_start <= t_start && t_start < join_end)) {
                continue;
            }

            // check if this buffer's write data overlaps with another stream's
            std::pair<int64_t, int64_t> data_range = std::make_pair(t_start, t_end);
            for (int i = 0; i < n_streams; ++i) {
                if (i == stream - 1) {
                    continue;
                }
                auto it = std::lower_bound(write_ranges[i].begin(), write_ranges[i].end(), data_range);

                if (it != write_ranges[i].end()) {
                    const std::pair<int64_t, int64_t> & other = *it;

                    // std::lower_bound returns the first element where other >= data_range (lexicographically).
                    // This guarantees other.first >= data_range.first.
                    // Therefore, overlap occurs iff other.first < data_range.second
                    // (i.e., the other range starts before this range ends).
                    if (other.first < data_range.second) {
                        GGML_LOG_DEBUG("Writes overlap for %s", tensor->name);
                        writes_overlap = true;
                        break;
                    }
                }
            }

            //check if all srcs are either in branch or don't have a branch
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                if (!tensor->src[i]) {
                    continue;
                }

                auto it = stream_mapping.find(tensor->src[i]);

                if (it == stream_mapping.end()) {
                    continue;
                }

                if (it->second != stream) {
                    dependent_srcs = true;
                    break;
                }
            }

            if (dependent_srcs || writes_overlap) {
                break;
            }
        }

        return !writes_overlap && !dependent_srcs;
    }

    ~ggml_hip_concurrent_event() {
        if (fork_event != nullptr) {
            HIP_CHECK(hipEventDestroy(fork_event));
        }
        for (hipEvent_t e : join_events) {
            if (e != nullptr) {
                HIP_CHECK(hipEventDestroy(e));
            }
        }
    }
};

struct ggml_hip_stream_context {
    std::unordered_map<const ggml_tensor *, ggml_hip_concurrent_event> concurrent_events;

    void reset() {
        concurrent_events.clear();
    }
};

struct ggml_backend_hip_context {
    int device;
    std::string name;
    hipEvent_t copy_event = nullptr;

    hipStream_t streams[GGML_HIP_MAX_DEVICES][GGML_HIP_MAX_STREAMS] = { { nullptr } };
    hipblasHandle_t hipblas_handles[GGML_HIP_MAX_DEVICES] = {nullptr};

    int curr_stream_no = 0;

    explicit ggml_backend_hip_context(int device) :
        device(device),
        name(GGML_HIP_NAME + std::to_string(device)) {
    }

    ggml_hip_stream_context concurrent_stream_context;

    ~ggml_backend_hip_context();

    hipStream_t stream(int device, int stream) {
        if (streams[device][stream] == nullptr) {
            ggml_hip_set_device(device);
            HIP_CHECK(hipStreamCreateWithFlags(&streams[device][stream], hipStreamNonBlocking));
        }
        return streams[device][stream];
    }

    hipStream_t stream() { return stream(device, curr_stream_no); }

    ggml_hip_stream_context & stream_context() { return concurrent_stream_context; }

    hipblasHandle_t hipblas_handle(int device) {
        if (hipblas_handles[device] == nullptr) {
            ggml_hip_set_device(device);
            HIPBLAS_CHECK(hipblasCreate(&hipblas_handles[device]));
            HIPBLAS_CHECK(hipblasSetMathMode(hipblas_handles[device], HIPBLAS_TF32_TENSOR_OP_MATH));
        }
        return hipblas_handles[device];
    }

    hipblasHandle_t hipblas_handle() {
        return hipblas_handle(device);
    }

    // pool
    std::unique_ptr<ggml_hip_pool> pools[GGML_HIP_MAX_DEVICES][GGML_HIP_MAX_STREAMS];

    static std::unique_ptr<ggml_hip_pool> new_pool_for_device(int device, int stream_no);

    ggml_hip_pool & pool(int device) {
        if (pools[device][curr_stream_no] == nullptr) {
            pools[device][curr_stream_no] = new_pool_for_device(device, curr_stream_no);
        }
        return *pools[device][curr_stream_no];
    }

    ggml_hip_pool & pool() {
        return pool(device);
    }
};

struct ggml_hip_mm_fusion_args_host {
    const ggml_tensor * x_bias = nullptr;
    const ggml_tensor * gate = nullptr;
    const ggml_tensor * gate_bias = nullptr;
    ggml_glu_op glu_op;
};
struct ggml_hip_mm_fusion_args_device {
    const void * x_bias = nullptr;
    const void * gate = nullptr;
    const void * gate_bias = nullptr;
    ggml_glu_op glu_op;
};
