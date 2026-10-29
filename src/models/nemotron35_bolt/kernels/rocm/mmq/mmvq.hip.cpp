// SPDX-License-Identifier: MIT
// Adapted from llama.cpp ggml/src/ggml-cuda (MIT, The ggml authors); see
// VENDOR.md.
#include "nb_mmq_prelude.h"

namespace nb_mmq {

#include "mmvq.hip.hpp"
#include "quantize.hip.hpp"
#include "unary.hip.hpp"
#include "vecdotq.hip.hpp"

#include <cstdint>

typedef float (*nb_vec_dot_q_hip_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ nb_vec_dot_q_hip_t nb_get_vec_dot_q_hip(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (NB_GGML_HIP_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (NB_GGML_HIP_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (NB_GGML_HIP_CC_IS_RDNA2(cc) || NB_GGML_HIP_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (NB_GGML_HIP_CC_IS_GCN(cc) || NB_GGML_HIP_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (NB_GGML_HIP_CC_IS_NVIDIA(cc)) {
        if (cc == NB_GGML_HIP_CC_VOLTA || cc >= NB_GGML_HIP_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= NB_GGML_HIP_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (NB_GGML_HIP_CC_IS_AMD(cc)) {
        if (NB_GGML_HIP_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (NB_GGML_HIP_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (NB_GGML_HIP_CC_IS_RDNA1(cc) || NB_GGML_HIP_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (NB_GGML_HIP_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (NB_GGML_HIP_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#else
    // Host compilation has no device architecture macros.
    return MMVQ_MAX_BATCH_SIZE;
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id())*nb_ggml_hip_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids, const nb_ggml_hip_mm_fusion_args_device fusion, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {

    constexpr int qk  = nb_ggml_hip_type_traits<type>::qk;
    constexpr int qi  = nb_ggml_hip_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id);
    constexpr int rows_per_hip_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = nb_ggml_hip_get_physical_warp_size();

    constexpr nb_vec_dot_q_hip_t nb_vec_dot_q_hip = nb_get_vec_dot_q_hip(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_hip_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    // A negative expert id cast to uint32 is a wild weight-channel offset.
    // Redirect to channel 0 and skip the dot loop so the output is a clean 0.
    const bool invalid_id = ncols_dst == 1 && ids && (int32_t)channel_x < 0;
    if (invalid_id) channel_x = 0;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    ggml_glu_op active_glu;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
    }


    float x_biases[ncols_dst]    = { 0.0f };
    float gate_biases[ncols_dst] = { 0.0f };
    if constexpr (has_fusion) {
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (use_bias) {
            x_bias = x_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            // 1. Hide latency by prefetching bias and gate here
            // 2. load only on threads that won't die after partial sum calculation
            if (threadIdx.x < rows_per_hip_block && threadIdx.y == 0 &&
                (rows_per_hip_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
        if (use_gate_bias) {
            gate_bias = gate_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            if (threadIdx.x < rows_per_hip_block && threadIdx.y == 0 &&
                (rows_per_hip_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_hip_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_hip_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_hip_block; ++i) {
                tmp[j][i] += nb_vec_dot_q_hip(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += nb_vec_dot_q_hip(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_hip_block][warp_size];
    __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_hip_block][warp_size];
    if constexpr (!has_fusion) {
        (void) tmp_shared_gate;
    } else if (!use_gate) {
        (void) tmp_shared_gate;
    }

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_hip_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_hip_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }
        }

        if (threadIdx.x < rows_per_hip_block && (rows_per_hip_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            float result = tmp[j][threadIdx.x];
            if constexpr (has_fusion) {
                result *= fusion.dst_scale;
                if (use_bias) {
                    result += x_biases[j];
                }
                if (use_gate) {
                    float gate_value = tmp_gate[j][threadIdx.x];
                    if (use_gate_bias) {
                        gate_value += gate_biases[j];
                    }
                    switch (active_glu) {
                        case GGML_GLU_OP_SWIGLU:
                            result *= nb_ggml_hip_op_silu_single(gate_value);
                            break;
                        case GGML_GLU_OP_GEGLU:
                            result *= nb_ggml_hip_op_gelu_single(gate_value);
                            break;
                        case GGML_GLU_OP_SWIGLU_OAI: {
                            result = nb_ggml_hip_op_swiglu_oai_single(gate_value, result);
                            break;
                        }
                        default:
                            result = result * gate_value;
                            break;
                    }
                }
            }
            dst[j*stride_col_dst + threadIdx.x] = result;
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, active_glu, gate_bias, x_bias, tmp_gate);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*nb_ggml_hip_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {

    constexpr int qk  = nb_ggml_hip_type_traits<type>::qk;
    constexpr int qi  = nb_ggml_hip_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = nb_ggml_hip_get_physical_warp_size();

    constexpr nb_vec_dot_q_hip_t nb_vec_dot_q_hip = nb_get_vec_dot_q_hip(type);

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    // A negative expert id would be a wild weight-channel offset: redirect the
    // pointer math to expert 0 and skip the dot loop so the row writes a clean
    // 0. Warp-uniform: token_idx is constant across a warp's lanes.
    const int32_t  id_raw     = ids[channel_dst + token_idx * ids_stride];
    const bool     invalid_id = id_raw < 0;
    const uint32_t channel_x  = invalid_id ? 0u : (uint32_t)id_raw;
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += nb_vec_dot_q_hip(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
        }
    }

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id);
    const int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const nb_ggml_hip_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, hipStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.dst_scale != 1.0f;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            mul_mat_vec_q<type, c_ncols_dst, true, small_k><<<block_nums, block_dims, nbytes_shared, stream>>>
                (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    mul_mat_vec_q<type, c_ncols_dst, false, small_k><<<block_nums, block_dims, nbytes_shared, stream>>>
        (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, hipStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst);
    const dim3 block_dims(warp_size, ncols_dst);

    mul_mat_vec_q_moe<type, rows_per_block><<<block_nums, block_dims, 0, stream>>>(
        vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        ncols_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const nb_ggml_hip_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, hipStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = nb_ggml_hip_get_device();
    const int                     cc        = nb_ggml_hip_info().devices[device].cc;
    const int warp_size = nb_ggml_hip_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    const bool has_ids = ids != nullptr;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        constexpr int qk                    = nb_ggml_hip_type_traits<type>::qk;
        constexpr int qi                    = nb_ggml_hip_type_traits<type>::qi;
        constexpr int vdr                   = get_vdr_mmvq(type);
        const int     blocks_per_row_x      = ncols_x / qk;
        const int     blocks_per_iter_1warp = vdr * warp_size / qi;
        const int     nwarps                = calc_nwarps(type, c_ncols_dst, table_id);
        bool          use                   = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = NB_GGML_HIP_CC_IS_NVIDIA(cc) && cc >= NB_GGML_HIP_CC_TURING;
        const bool is_nvidia_pascal_older = NB_GGML_HIP_CC_IS_NVIDIA(cc) && cc < NB_GGML_HIP_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                NB_GGML_HIP_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            constexpr int c_ncols_dst = 1;

            bool use_small_k = should_use_small_k(c_ncols_dst);

            if (use_small_k) {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id, true);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, true>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            } else {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }

    GGML_UNUSED(has_fusion);
}
// Was `static` upstream. Exposed so the raw-pointer adapter can call MMVQ
// without ggml_tensor structs.
void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const nb_ggml_hip_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, hipStream_t stream) {
    // Nemotron 3.5 Bolt instantiates only its three weight formats.
    switch (type_x) {
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// The ggml-graph entry points are elided; nb_mmq.hip.cpp calls
// mul_mat_vec_q_switch_type directly.

}  // namespace nb_mmq
