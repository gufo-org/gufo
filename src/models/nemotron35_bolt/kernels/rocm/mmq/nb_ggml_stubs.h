// SPDX-License-Identifier: MIT
// nb_ggml_stubs.h - minimal ggml-API stubs for Nemotron 3.5 Bolt's vendored
// matrix kernels.
//
// The common/mma/mmq/vecdotq/quantize/mmid headers in this directory derive
// from llama.cpp's ggml-cuda backend (MIT, copyright 2023-2026 The ggml
// authors). They include "ggml.h", "ggml-impl.h" and "ggml-hip.h"; here those
// names are thin redirect headers that include this file.
//
// This header declares the minimum ggml surface the templates reference,
// excluding what common.hip.hpp provides itself (compute-capability
// constants, MMA flags, nb_ggml_hip_device_info, the pool and context
// types, NB_HIP_CHECK / NB_HIPBLAS_CHECK, nb_ggml_hip_get_device,
// nb_ggml_hip_set_device and nb_ggml_hip_info).
//
// It provides:
//   * GGML_ASSERT / GGML_ABORT / GGML_UNUSED / GGML_UNUSED_VARS / GGML_PAD
//   * GGML_MAX_DIMS / GGML_MAX_SRC / NB_GGML_HIP_NAME /
//     NB_GGML_HIP_MAX_DEVICES / NB_GGML_HIP_MAX_STREAMS / GGML_LOG_DEBUG
//   * enum ggml_type with upstream's values (only Q8_0, IQ2_XXS and NVFP4
//     are instantiated, but the shared switches name every MMQ type)
//   * enum ggml_glu_op and struct ggml_tensor, complete enough for unused
//     upstream declarations to compile
//   * ggml_type_size() / ggml_blck_size() lookups
//
// ggml-common.h owns ggml_half / ggml_half2, GGML_EXTENSION and the block
// structs. Everything here lives in namespace nb_mmq (see nb_mmq_prelude.h).

#pragma once

#include <cassert>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// ----------------------------------------------------------------------------
// Macros
// ----------------------------------------------------------------------------

#ifndef GGML_ASSERT
#define GGML_ASSERT(cond) \
    do { if (!(cond)) { \
        fprintf(stderr, "GGML_ASSERT(%s) failed at %s:%d\n", #cond, __FILE__, __LINE__); \
        abort(); \
    } } while (0)
#endif

#ifndef GGML_ABORT
#define GGML_ABORT(fmt, ...) \
    do { \
        fprintf(stderr, "GGML_ABORT: " fmt " at %s:%d\n", ##__VA_ARGS__, __FILE__, __LINE__); \
        abort(); \
    } while (0)
#endif

#ifndef GGML_UNUSED
#define GGML_UNUSED(x) ((void)(x))
#endif

// Variadic GGML_UNUSED_VARS: drop up to 12 unused names without warnings.
#ifndef GGML_UNUSED_VARS
#define GGML_UNUSED_VARS_1(_1)                                             GGML_UNUSED(_1)
#define GGML_UNUSED_VARS_2(_1,_2)                                          GGML_UNUSED(_1); GGML_UNUSED(_2)
#define GGML_UNUSED_VARS_3(_1,_2,_3)                                       GGML_UNUSED_VARS_2(_1,_2); GGML_UNUSED(_3)
#define GGML_UNUSED_VARS_4(_1,_2,_3,_4)                                    GGML_UNUSED_VARS_3(_1,_2,_3); GGML_UNUSED(_4)
#define GGML_UNUSED_VARS_5(_1,_2,_3,_4,_5)                                 GGML_UNUSED_VARS_4(_1,_2,_3,_4); GGML_UNUSED(_5)
#define GGML_UNUSED_VARS_6(_1,_2,_3,_4,_5,_6)                              GGML_UNUSED_VARS_5(_1,_2,_3,_4,_5); GGML_UNUSED(_6)
#define GGML_UNUSED_VARS_7(_1,_2,_3,_4,_5,_6,_7)                           GGML_UNUSED_VARS_6(_1,_2,_3,_4,_5,_6); GGML_UNUSED(_7)
#define GGML_UNUSED_VARS_8(_1,_2,_3,_4,_5,_6,_7,_8)                        GGML_UNUSED_VARS_7(_1,_2,_3,_4,_5,_6,_7); GGML_UNUSED(_8)
#define GGML_UNUSED_VARS_9(_1,_2,_3,_4,_5,_6,_7,_8,_9)                     GGML_UNUSED_VARS_8(_1,_2,_3,_4,_5,_6,_7,_8); GGML_UNUSED(_9)
#define GGML_UNUSED_VARS_10(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10)                GGML_UNUSED_VARS_9(_1,_2,_3,_4,_5,_6,_7,_8,_9); GGML_UNUSED(_10)
#define GGML_UNUSED_VARS_11(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11)            GGML_UNUSED_VARS_10(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10); GGML_UNUSED(_11)
#define GGML_UNUSED_VARS_12(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12)        GGML_UNUSED_VARS_11(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11); GGML_UNUSED(_12)
#define GGML_UNUSED_VARS_PICK(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,NAME,...) NAME
#define GGML_UNUSED_VARS(...) \
    GGML_UNUSED_VARS_PICK(__VA_ARGS__, \
        GGML_UNUSED_VARS_12, GGML_UNUSED_VARS_11, GGML_UNUSED_VARS_10, \
        GGML_UNUSED_VARS_9, GGML_UNUSED_VARS_8, GGML_UNUSED_VARS_7, \
        GGML_UNUSED_VARS_6, GGML_UNUSED_VARS_5, GGML_UNUSED_VARS_4, \
        GGML_UNUSED_VARS_3, GGML_UNUSED_VARS_2, GGML_UNUSED_VARS_1)(__VA_ARGS__)
#endif

#ifndef GGML_PAD
#define GGML_PAD(x, n) (((x) + (n) - 1) / (n) * (n))
#endif

#ifndef GGML_MAX_DIMS
#define GGML_MAX_DIMS 4
#endif

#ifndef GGML_MAX_SRC
#define GGML_MAX_SRC  10
#endif

#ifndef NB_GGML_HIP_NAME
#define NB_GGML_HIP_NAME "NB_HIP"
#endif

#ifndef NB_GGML_HIP_MAX_DEVICES
#define NB_GGML_HIP_MAX_DEVICES 16
#endif

#ifndef NB_GGML_HIP_MAX_STREAMS
#define NB_GGML_HIP_MAX_STREAMS 8
#endif

#ifndef GGML_LOG_DEBUG
#define GGML_LOG_DEBUG(...) ((void)0)
#endif

// HIP graphs are disabled here; the caller owns streams and graphs.
#undef NB_GGML_HIP_USE_GRAPHS
#undef GGML_HIP_GRAPHS
#undef GGML_MUSA_GRAPHS

// GGML_EXTENSION: ggml-common.h provides the canonical definition. We leave
// it undefined here so the vendored header's `#define GGML_EXTENSION
// __extension__` wins.

// ----------------------------------------------------------------------------
// Quantization type enum.
//
// Order matches llama.cpp's enum ggml_type. Values are pinned because the
// mmq switch uses them as case labels.
// ----------------------------------------------------------------------------

enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 / Q4_3 deprecated
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_MXFP4   = 39,
    GGML_TYPE_NVFP4   = 40,
    GGML_TYPE_Q1_0    = 41,
    GGML_TYPE_COUNT   = 42,
};

enum ggml_glu_op {
    GGML_GLU_OP_REGLU,
    GGML_GLU_OP_GEGLU,
    GGML_GLU_OP_SWIGLU,
    GGML_GLU_OP_SWIGLU_OAI, // referenced by mmvq.cu's fused-GLU epilogue
    GGML_GLU_OP_COUNT,
};

// ----------------------------------------------------------------------------
// ggml_tensor: complete enough for common.hip.hpp's
// nb_ggml_hip_concurrent_event::is_valid() to compile cleanly. We NEVER
// instantiate or dereference one of these - the concurrent path is
// disabled.
//
// Field set matches the upstream order/types. Nothing here copies into
// these.
// ----------------------------------------------------------------------------

struct ggml_tensor {
    enum ggml_type type;
    int32_t op;                              // enum ggml_op (opaque to us)
    int32_t flags;
    int64_t ne[GGML_MAX_DIMS];               // shape
    size_t  nb[GGML_MAX_DIMS];               // stride bytes
    int32_t op_params[16];                   // GGML_MAX_OP_PARAMS / sizeof(int32_t)
    struct ggml_tensor * src[GGML_MAX_SRC];
    struct ggml_tensor * view_src;
    size_t  view_offs;
    void *  data;
    char    name[64];                        // GGML_MAX_NAME
    void *  extra;
    char    padding[8];
};

// ggml_nbytes: byte size of tensor data. We never call this; provide a
// stub so common.hip.hpp's is_valid() compiles. If anything does call it the
// returned 0 will surface as an immediate logic error.
static inline int64_t ggml_nbytes(const struct ggml_tensor * /*t*/) { return 0; }

// Microsecond timer (used only inside USE_CUDA_GRAPH paths we disable).
int64_t ggml_time_us();

// Selects the per-stream scratch arena for the next kernels
// (nb_ggml_stubs.hip.cpp).
void nb_pool_set_stream(hipStream_t stream);
struct nb_ggml_hip_pool;
// Allocate or free the scratch arena of one stream outside any capture.
hipError_t nb_pool_reserve_stream(nb_ggml_hip_pool & pool, hipStream_t stream);
void nb_pool_release_stream(nb_ggml_hip_pool & pool, hipStream_t stream);

// ----------------------------------------------------------------------------
// Inline size traits.
//
// Lookup tables hand-aligned with llama.cpp's ggml_type_traits in
// ggml/src/ggml.c. Only types referenced by the vendored switch are needed.
// ----------------------------------------------------------------------------

inline size_t ggml_type_size(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:     return 4;
        case GGML_TYPE_F16:     return 2;
        case GGML_TYPE_BF16:    return 2;
        case GGML_TYPE_I8:      return 1;
        case GGML_TYPE_I16:     return 2;
        case GGML_TYPE_I32:     return 4;
        case GGML_TYPE_I64:     return 8;
        case GGML_TYPE_F64:     return 8;
        case GGML_TYPE_Q4_0:    return 18;
        case GGML_TYPE_Q4_1:    return 20;
        case GGML_TYPE_Q5_0:    return 22;
        case GGML_TYPE_Q5_1:    return 24;
        case GGML_TYPE_Q8_0:    return 34;
        case GGML_TYPE_Q8_1:    return 36;
        case GGML_TYPE_Q2_K:    return 84;
        case GGML_TYPE_Q3_K:    return 110;
        case GGML_TYPE_Q4_K:    return 144;
        case GGML_TYPE_Q5_K:    return 176;
        case GGML_TYPE_Q6_K:    return 210;
        case GGML_TYPE_Q8_K:    return 292;
        case GGML_TYPE_IQ2_XXS: return 66;
        case GGML_TYPE_IQ2_XS:  return 74;
        case GGML_TYPE_IQ2_S:   return 82;
        case GGML_TYPE_IQ3_XXS: return 98;
        case GGML_TYPE_IQ3_S:   return 110;
        case GGML_TYPE_IQ1_S:   return 50;
        case GGML_TYPE_IQ1_M:   return 56;
        case GGML_TYPE_IQ4_NL:  return 18;
        case GGML_TYPE_IQ4_XS:  return 136;
        case GGML_TYPE_MXFP4:   return 17;
        case GGML_TYPE_NVFP4:   return 36;
        case GGML_TYPE_Q1_0:    return 36;
        default:                return 0;
    }
}

inline int64_t ggml_blck_size(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_I8:
        case GGML_TYPE_I16:
        case GGML_TYPE_I32:
        case GGML_TYPE_I64:
        case GGML_TYPE_F64:
            return 1;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
            return 32;
        case GGML_TYPE_NVFP4:
            return 64;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q1_0:
            return 256;
        default:
            return 1;
    }
}
