// DS4 ROCm Q8_0 matmul / grouped-output / HC-expand kernels.
//
// Included from ds4_rocm.hip.cpp in the same translation unit so kernel helpers stay
// private/static while we gradually split the custom ROCm backend into modules.

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <rocwmma/rocwmma.hpp>
#endif

__device__ __forceinline__ static int32_t load_i8x4_i32_aligned(const int8_t *p) {
    return *(const int32_t *)p;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_unaligned(const int8_t *p) {
    const uint8_t *u = (const uint8_t *)p;
    return (int32_t)((uint32_t)u[0] |
                     ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) |
                     ((uint32_t)u[3] << 24));
}

__device__ __forceinline__ static int32_t dot_i8x32_dp4a(const int8_t *a, const int8_t *b) {
    int32_t dot = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        dot = __dp4a(load_i8x4_i32_unaligned(a + i), load_i8x4_i32_aligned(b + i), dot);
    }
    return dot;
}

__device__ __forceinline__ static int32_t dot_i8x32_dp4a_loaded(
        int32_t a0,
        int32_t a1,
        int32_t a2,
        int32_t a3,
        int32_t a4,
        int32_t a5,
        int32_t a6,
        int32_t a7,
        const int8_t *b) {
    int32_t dot = 0;
    dot = __dp4a(a0, load_i8x4_i32_aligned(b + 0u), dot);
    dot = __dp4a(a1, load_i8x4_i32_aligned(b + 4u), dot);
    dot = __dp4a(a2, load_i8x4_i32_aligned(b + 8u), dot);
    dot = __dp4a(a3, load_i8x4_i32_aligned(b + 12u), dot);
    dot = __dp4a(a4, load_i8x4_i32_aligned(b + 16u), dot);
    dot = __dp4a(a5, load_i8x4_i32_aligned(b + 20u), dot);
    dot = __dp4a(a6, load_i8x4_i32_aligned(b + 24u), dot);
    dot = __dp4a(a7, load_i8x4_i32_aligned(b + 28u), dot);
    return dot;
}

__device__ __forceinline__ static int32_t dot_i8_block(const int8_t *a, const int8_t *b, uint64_t n, int use_dp4a) {
    if (use_dp4a && n == 32u) return dot_i8x32_dp4a(a, b);
    int32_t dot = 0;
    for (uint64_t i = 0; i < n; i++) dot += (int32_t)a[i] * (int32_t)b[i];
    return dot;
}

__global__ static void quantize_q8_0_f32_kernel(
        int8_t *xq,
        float *xscale,
        const float *x,
        uint64_t in_dim,
        uint64_t blocks) {
    uint64_t b = blockIdx.x;
    uint64_t tok = blockIdx.y;
    if (b >= blocks) return;
    uint64_t i0 = b * 32;
    uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
    const float *xr = x + tok * in_dim + i0;

    float a = 0.0f;
    if (threadIdx.x < bn) a = fabsf(xr[threadIdx.x]);
    a = warp_max_f32(a);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    const float d = __shfl(a, 0, 32) / 127.0f;
#else
    const float d = __shfl_sync(FULL_WARP_MASK, a, 0, 32) / 127.0f;
#endif
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0) xscale[tok * blocks + b] = d;
    int8_t *dst = xq + (tok * blocks + b) * 32;
    if (threadIdx.x < bn) {
        int v = (int)lrintf(xr[threadIdx.x] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[threadIdx.x] = (int8_t)v;
    } else {
        dst[threadIdx.x] = 0;
    }
}

__global__ static void matmul_q8_0_preq_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void matmul_q8_0_preq_rows_w32_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        uint32_t rows_per_block,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34u;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const __half *scale_h = (const __half *)(wr + b * 34u);
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int8_t *xqb = xq + b * 32u;
        const int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[row] = acc;
}


/*
 * Narrow-batch Q8_0 projection: read each weight block once, apply it to every
 * row in the batch.
 *
 * The prompt-chunk batch kernel stages float activations and reaches only about a
 * quarter of DRAM bandwidth at these widths, while the prequantized batch kernel
 * re-reads the whole weight matrix per row. This one keeps the single-row decode
 * kernel's shape - one warp per output row, lane-strided 34-byte blocks, dp4a
 * dots - and adds an inner loop over the batch, so the weights move once and the
 * activations are already quantized.
 *
 * The accumulation order per (output row, row index) is deliberately identical to
 * matmul_q8_0_preq_rows_w32_kernel: same block sequence per lane, same
 * multiply order, same warp reduction. That is what keeps a verified speculative
 * row bitwise equal to what ordinary decode would have produced.
 */
template <uint32_t MAXT, bool EXACT>
__global__ static void matmul_q8_0_preq_batch_reuse_w32_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        uint32_t n_tok,
        uint32_t rows_per_block) {
    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    /* row depends only on threadIdx.x >> 5, so a warp exits as a whole and the
     * reductions below always run with a full warp. */
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34u;

    float acc[MAXT];
#pragma unroll
    for (uint32_t t = 0; t < MAXT; t++) acc[t] = 0.0f;

    for (uint64_t b = lane; b < blocks; b += 32u) {
        const float wscale = __half2float(*(const __half *)(wr + b * 34u));
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int32_t q0 = load_i8x4_i32_unaligned(qs + 0u);
        const int32_t q1 = load_i8x4_i32_unaligned(qs + 4u);
        const int32_t q2 = load_i8x4_i32_unaligned(qs + 8u);
        const int32_t q3 = load_i8x4_i32_unaligned(qs + 12u);
        const int32_t q4 = load_i8x4_i32_unaligned(qs + 16u);
        const int32_t q5 = load_i8x4_i32_unaligned(qs + 20u);
        const int32_t q6 = load_i8x4_i32_unaligned(qs + 24u);
        const int32_t q7 = load_i8x4_i32_unaligned(qs + 28u);
#pragma unroll
        for (uint32_t t = 0; t < MAXT; t++) {
            if constexpr (!EXACT) {
                if (t >= n_tok) break;
            }
            const int8_t *xqb = xq + (uint64_t)t * blocks * 32u + b * 32u;
            const int dot = dot_i8x32_dp4a_loaded(
                q0, q1, q2, q3, q4, q5, q6, q7, xqb);
            acc[t] += wscale * xscale[(uint64_t)t * blocks + b] * (float)dot;
        }
    }

#pragma unroll
    for (uint32_t t = 0; t < MAXT; t++) {
        if constexpr (!EXACT) {
            if (t >= n_tok) break;
        }
        const float sum = warp_sum_f32(acc[t]);
        if (lane == 0u) out[(uint64_t)t * out_dim + row] = sum;
    }
}

/*
 * Two independent verification blocks share one Q8 weight stream while each
 * block keeps the exact accumulation order of the original narrow-batch
 * kernel. DSpark pairs C2/C4/C6/C8 requests, so this halves the dominant
 * projection traffic without flattening request boundaries.
 */
template <uint32_t NT>
__global__ static void matmul_q8_0_preq_group_pair_w32_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        uint32_t row0,
        uint32_t row1,
        uint32_t rows_per_block) {
    const uint64_t row =
        (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34u;

    float acc0[NT];
    float acc1[NT];
#pragma unroll
    for (uint32_t t = 0; t < NT; ++t) {
        acc0[t] = 0.0f;
        acc1[t] = 0.0f;
    }

    for (uint64_t b = lane; b < blocks; b += 32u) {
        const float wscale =
            __half2float(*(const __half *)(wr + b * 34u));
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int32_t q0 = load_i8x4_i32_unaligned(qs + 0u);
        const int32_t q1 = load_i8x4_i32_unaligned(qs + 4u);
        const int32_t q2 = load_i8x4_i32_unaligned(qs + 8u);
        const int32_t q3 = load_i8x4_i32_unaligned(qs + 12u);
        const int32_t q4 = load_i8x4_i32_unaligned(qs + 16u);
        const int32_t q5 = load_i8x4_i32_unaligned(qs + 20u);
        const int32_t q6 = load_i8x4_i32_unaligned(qs + 24u);
        const int32_t q7 = load_i8x4_i32_unaligned(qs + 28u);
#pragma unroll
        for (uint32_t t = 0; t < NT; ++t) {
            const uint64_t token0 = (uint64_t)row0 + t;
            const uint64_t token1 = (uint64_t)row1 + t;
            const int dot0 = dot_i8x32_dp4a_loaded(
                q0, q1, q2, q3, q4, q5, q6, q7,
                xq + token0 * blocks * 32u + b * 32u);
            const int dot1 = dot_i8x32_dp4a_loaded(
                q0, q1, q2, q3, q4, q5, q6, q7,
                xq + token1 * blocks * 32u + b * 32u);
            const float scaled0 =
                wscale * xscale[token0 * blocks + b];
            const float scaled1 =
                wscale * xscale[token1 * blocks + b];
            acc0[t] = fmaf(scaled0, (float)dot0, acc0[t]);
            acc1[t] = fmaf(scaled1, (float)dot1, acc1[t]);
        }
    }

#pragma unroll
    for (uint32_t t = 0; t < NT; ++t) {
        const float sum0 = warp_sum_f32(acc0[t]);
        const float sum1 = warp_sum_f32(acc1[t]);
        if (lane == 0u) {
            out[((uint64_t)row0 + t) * out_dim + row] = sum0;
            out[((uint64_t)row1 + t) * out_dim + row] = sum1;
        }
    }
}

/*
 * Ragged request pairs retain the qualified per-row accumulation order while
 * sharing each Q8 weight block across two independently sized draft prefixes.
 */
template <uint32_t NT0, uint32_t NT1>
__global__ static void matmul_q8_0_preq_ragged_group_pair_w32_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        uint32_t row0,
        uint32_t row1,
        uint32_t rows_per_block) {
    static_assert(NT0 >= 1u && NT0 <= 6u);
    static_assert(NT1 >= 1u && NT1 <= 6u);
    const uint64_t row =
        (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34u;

    float acc0[NT0] = {};
    float acc1[NT1] = {};
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const float wscale =
            __half2float(*(const __half *)(wr + b * 34u));
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int32_t q0 = load_i8x4_i32_unaligned(qs + 0u);
        const int32_t q1 = load_i8x4_i32_unaligned(qs + 4u);
        const int32_t q2 = load_i8x4_i32_unaligned(qs + 8u);
        const int32_t q3 = load_i8x4_i32_unaligned(qs + 12u);
        const int32_t q4 = load_i8x4_i32_unaligned(qs + 16u);
        const int32_t q5 = load_i8x4_i32_unaligned(qs + 20u);
        const int32_t q6 = load_i8x4_i32_unaligned(qs + 24u);
        const int32_t q7 = load_i8x4_i32_unaligned(qs + 28u);
#pragma unroll
        for (uint32_t t = 0u; t < NT0; ++t) {
            const uint64_t token = (uint64_t)row0 + t;
            const int dot = dot_i8x32_dp4a_loaded(
                q0, q1, q2, q3, q4, q5, q6, q7,
                xq + token * blocks * 32u + b * 32u);
            acc0[t] = fmaf(
                wscale * xscale[token * blocks + b],
                (float)dot, acc0[t]);
        }
#pragma unroll
        for (uint32_t t = 0u; t < NT1; ++t) {
            const uint64_t token = (uint64_t)row1 + t;
            const int dot = dot_i8x32_dp4a_loaded(
                q0, q1, q2, q3, q4, q5, q6, q7,
                xq + token * blocks * 32u + b * 32u);
            acc1[t] = fmaf(
                wscale * xscale[token * blocks + b],
                (float)dot, acc1[t]);
        }
    }

#pragma unroll
    for (uint32_t t = 0u; t < NT0; ++t) {
        const float sum = warp_sum_f32(acc0[t]);
        if (lane == 0u) {
            out[((uint64_t)row0 + t) * out_dim + row] = sum;
        }
    }
#pragma unroll
    for (uint32_t t = 0u; t < NT1; ++t) {
        const float sum = warp_sum_f32(acc1[t]);
        if (lane == 0u) {
            out[((uint64_t)row1 + t) * out_dim + row] = sum;
        }
    }
}

/*
 * Equal-width request pairs use the same arithmetic as the single-pair kernel
 * above, but place the pair index in grid.y. This keeps every request's
 * accumulator independent while replacing C/2 launches per projection with
 * one launch for C2/C4/C6/C8.
 */
template <uint32_t NT>
__global__ static void matmul_q8_0_preq_equal_group_pairs_w32_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        uint32_t rows_per_block) {
    const uint64_t row =
        (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;

    const uint32_t row0 = (uint32_t)blockIdx.y * (2u * NT);
    const uint32_t row1 = row0 + NT;
    const unsigned char *wr = w + row * blocks * 34u;

    float acc0[NT];
    float acc1[NT];
#pragma unroll
    for (uint32_t t = 0; t < NT; ++t) {
        acc0[t] = 0.0f;
        acc1[t] = 0.0f;
    }

    for (uint64_t b = lane; b < blocks; b += 32u) {
        const float wscale =
            __half2float(*(const __half *)(wr + b * 34u));
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int32_t q0 = load_i8x4_i32_unaligned(qs + 0u);
        const int32_t q1 = load_i8x4_i32_unaligned(qs + 4u);
        const int32_t q2 = load_i8x4_i32_unaligned(qs + 8u);
        const int32_t q3 = load_i8x4_i32_unaligned(qs + 12u);
        const int32_t q4 = load_i8x4_i32_unaligned(qs + 16u);
        const int32_t q5 = load_i8x4_i32_unaligned(qs + 20u);
        const int32_t q6 = load_i8x4_i32_unaligned(qs + 24u);
        const int32_t q7 = load_i8x4_i32_unaligned(qs + 28u);
#pragma unroll
        for (uint32_t t = 0; t < NT; ++t) {
            const uint64_t token0 = (uint64_t)row0 + t;
            const uint64_t token1 = (uint64_t)row1 + t;
            const int dot0 = dot_i8x32_dp4a_loaded(
                q0, q1, q2, q3, q4, q5, q6, q7,
                xq + token0 * blocks * 32u + b * 32u);
            const int dot1 = dot_i8x32_dp4a_loaded(
                q0, q1, q2, q3, q4, q5, q6, q7,
                xq + token1 * blocks * 32u + b * 32u);
            const float scaled0 =
                wscale * xscale[token0 * blocks + b];
            const float scaled1 =
                wscale * xscale[token1 * blocks + b];
            acc0[t] = fmaf(scaled0, (float)dot0, acc0[t]);
            acc1[t] = fmaf(scaled1, (float)dot1, acc1[t]);
        }
    }

#pragma unroll
    for (uint32_t t = 0; t < NT; ++t) {
        const float sum0 = warp_sum_f32(acc0[t]);
        const float sum1 = warp_sum_f32(acc1[t]);
        if (lane == 0u) {
            out[((uint64_t)row0 + t) * out_dim + row] = sum0;
            out[((uint64_t)row1 + t) * out_dim + row] = sum1;
        }
    }
}

/*
 * A DSpark multi-request decode is one logical row batch. Request boundaries
 * only select KV-cache slices; dense projections do not depend on them. Keep
 * one accumulator per flattened row so every Q8 weight block is loaded once
 * for the whole C2/C4/C6/C8 batch while preserving the single-row fmaf order.
 */
template <uint32_t NT>
__global__ static void matmul_q8_0_preq_all_rows_exact_w32_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        uint32_t rows_per_block) {
    const uint64_t row =
        (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;

    const unsigned char *wr = w + row * blocks * 34u;
    float acc[NT];
#pragma unroll
    for (uint32_t token = 0; token < NT; ++token) {
        acc[token] = 0.0f;
    }

    for (uint64_t b = lane; b < blocks; b += 32u) {
        const float wscale =
            __half2float(*(const __half *)(wr + b * 34u));
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int32_t q0 = load_i8x4_i32_unaligned(qs + 0u);
        const int32_t q1 = load_i8x4_i32_unaligned(qs + 4u);
        const int32_t q2 = load_i8x4_i32_unaligned(qs + 8u);
        const int32_t q3 = load_i8x4_i32_unaligned(qs + 12u);
        const int32_t q4 = load_i8x4_i32_unaligned(qs + 16u);
        const int32_t q5 = load_i8x4_i32_unaligned(qs + 20u);
        const int32_t q6 = load_i8x4_i32_unaligned(qs + 24u);
        const int32_t q7 = load_i8x4_i32_unaligned(qs + 28u);
#pragma unroll
        for (uint32_t token = 0; token < NT; ++token) {
            const int dot = dot_i8x32_dp4a_loaded(
                q0, q1, q2, q3, q4, q5, q6, q7,
                xq + ((uint64_t)token * blocks + b) * 32u);
            const float scaled =
                wscale * xscale[(uint64_t)token * blocks + b];
            acc[token] = fmaf(scaled, (float)dot, acc[token]);
        }
    }

#pragma unroll
    for (uint32_t token = 0; token < NT; ++token) {
        const float sum = warp_sum_f32(acc[token]);
        if (lane == 0u) {
            out[(uint64_t)token * out_dim + row] = sum;
        }
    }
}

__global__ static void matmul_q8_0_pair_preq_warp8_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out0_dim && row >= out1_dim) return;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const unsigned char *wr0 = row < out0_dim ? w0 + row * blocks * 34 : NULL;
    const unsigned char *wr1 = row < out1_dim ? w1 + row * blocks * 34 : NULL;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const int8_t *xqb = xq + b * 32;
        const float xs = xscale[b];
        if (wr0) {
            const __half *scale_h = (const __half *)(wr0 + b * 34);
            const int8_t *qs = (const int8_t *)(wr0 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc0 += __half2float(*scale_h) * xs * (float)dot;
        }
        if (wr1) {
            const __half *scale_h = (const __half *)(wr1 + b * 34);
            const int8_t *qs = (const int8_t *)(wr1 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc1 += __half2float(*scale_h) * xs * (float)dot;
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        if (row < out0_dim) out0[row] = acc0;
        if (row < out1_dim) out1[row] = acc1;
    }
}

__global__ static void matmul_q8_0_preq_batch_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok >= n_tok) return;

    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[tok * out_dim + row] = acc;
}

__device__ static float q8_0_scale_scalar(const unsigned char *blk) {
    const uint16_t bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    return __half2float(__ushort_as_half((unsigned short)bits));
}

__device__ static float q8_0_scale_broadcast_w32(const unsigned char *blk) {
    float d = 0.0f;
    if ((threadIdx.x & 31u) == 0u) d = q8_0_scale_scalar(blk);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    return __shfl(d, 0, 32);
#else
    return __shfl_sync(FULL_WARP_MASK, d, 0, 32);
#endif
}

__global__ static void matmul_q8_0_f32_batch_warp8_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok >= n_tok) return;
    const unsigned char *wr = w + row * blocks * 34u;
    const float *xr = x + tok * in_dim;
    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i = b * 32u + lane;
        if (i < in_dim) {
            const unsigned char *blk = wr + b * 34u;
            const float d = q8_0_scale_broadcast_w32(blk);
            const int8_t q = ((const int8_t *)(blk + 2u))[lane];
            acc += d * (float)q * xr[i];
        }
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[tok * out_dim + row] = acc;
}

template <uint32_t TOK_TILE, uint32_t BLOCKS_TILE>
__global__ static void shared_gate_up_swiglu_q8_0_batch_sharedx_w32_kernel(
        float *gate,
        float *up,
        float *mid,
        const unsigned char *wg,
        const unsigned char *wu,
        const float *x,
        uint32_t n_blocks,
        uint32_t out_dim,
        uint32_t n_tok,
        uint64_t row_bytes,
        int store_gate_up) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t t0 = blockIdx.y * TOK_TILE;
    if (t0 >= n_tok) return;
    const bool row_valid = row < out_dim;
    const unsigned char *wgr = wg + (uint64_t)(row_valid ? row : 0u) * row_bytes;
    const unsigned char *wur = wu + (uint64_t)(row_valid ? row : 0u) * row_bytes;
    const uint32_t in_dim = n_blocks << 5u;
    float accg[TOK_TILE];
    float accu[TOK_TILE];
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) {
        accg[u] = 0.0f;
        accu[u] = 0.0f;
    }

    for (uint32_t b0 = 0; b0 < n_blocks; b0 += BLOCKS_TILE) {
        const uint32_t b_count = ((b0 + BLOCKS_TILE) <= n_blocks) ? BLOCKS_TILE : (n_blocks - b0);
        for (uint32_t j = tid; j < TOK_TILE * BLOCKS_TILE * 32u; j += blockDim.x) {
            const uint32_t u = j / (BLOCKS_TILE * 32u);
            const uint32_t r = j - u * (BLOCKS_TILE * 32u);
            const uint32_t bb = r >> 5u;
            const uint32_t k = r & 31u;
            const uint32_t t = t0 + u;
            shx[j] = (t < n_tok && bb < b_count)
                ? x[(uint64_t)t * in_dim + ((uint64_t)(b0 + bb) << 5u) + k]
                : 0.0f;
        }
        __syncthreads();
        if (row_valid) {
            for (uint32_t bb = 0; bb < b_count; bb++) {
                const unsigned char *bg = wgr + (uint64_t)(b0 + bb) * 34u;
                const unsigned char *bu = wur + (uint64_t)(b0 + bb) * 34u;
                const float dg = q8_0_scale_broadcast_w32(bg);
                const float du = q8_0_scale_broadcast_w32(bu);
                const float wvg = dg * (float)((const int8_t *)(bg + 2u))[lane];
                const float wvu = du * (float)((const int8_t *)(bu + 2u))[lane];
#pragma unroll
                for (uint32_t u = 0; u < TOK_TILE; u++) {
                    const float xv = shx[(u * BLOCKS_TILE + bb) * 32u + lane];
                    accg[u] += wvg * xv;
                    accu[u] += wvu * xv;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) {
        accg[u] = warp_sum_f32(accg[u]);
        accu[u] = warp_sum_f32(accu[u]);
    }
    if (lane == 0u && row_valid) {
#pragma unroll
        for (uint32_t u = 0; u < TOK_TILE; u++) {
            const uint32_t t = t0 + u;
            if (t < n_tok) {
                const uint64_t off = (uint64_t)t * out_dim + row;
                const float g = accg[u];
                const float uv = accu[u];
                if (store_gate_up) {
                    gate[off] = g;
                    up[off] = uv;
                }
                mid[off] = (g / (1.0f + expf(-g))) * uv;
            }
        }
    }
}

template <uint32_t TOK_TILE, uint32_t BLOCKS_TILE>
__global__ static void matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_blocks,
        uint32_t out_dim,
        uint32_t n_tok,
        uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t t0 = blockIdx.y * TOK_TILE;
    if (t0 >= n_tok) return;
    const bool row_valid = row < out_dim;
    const unsigned char *wr = w + (uint64_t)(row_valid ? row : 0u) * row_bytes;
    const uint32_t in_dim = n_blocks << 5u;
    float acc[TOK_TILE];
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) acc[u] = 0.0f;

    for (uint32_t b0 = 0; b0 < n_blocks; b0 += BLOCKS_TILE) {
        const uint32_t b_count = ((b0 + BLOCKS_TILE) <= n_blocks) ? BLOCKS_TILE : (n_blocks - b0);
        for (uint32_t j = tid; j < TOK_TILE * BLOCKS_TILE * 32u; j += blockDim.x) {
            const uint32_t u = j / (BLOCKS_TILE * 32u);
            const uint32_t r = j - u * (BLOCKS_TILE * 32u);
            const uint32_t bb = r >> 5u;
            const uint32_t k = r & 31u;
            const uint32_t t = t0 + u;
            shx[j] = (t < n_tok && bb < b_count)
                ? x[(uint64_t)t * in_dim + ((uint64_t)(b0 + bb) << 5u) + k]
                : 0.0f;
        }
        __syncthreads();
        if (row_valid) {
            for (uint32_t bb = 0; bb < b_count; bb++) {
                const unsigned char *blk = wr + (uint64_t)(b0 + bb) * 34u;
                const float d = q8_0_scale_broadcast_w32(blk);
                const int8_t q = ((const int8_t *)(blk + 2u))[lane];
                const float wv = d * (float)q;
#pragma unroll
                for (uint32_t u = 0; u < TOK_TILE; u++) acc[u] += wv * shx[(u * BLOCKS_TILE + bb) * 32u + lane];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) acc[u] = warp_sum_f32(acc[u]);
    if (lane == 0u && row_valid) {
#pragma unroll
        for (uint32_t u = 0; u < TOK_TILE; u++) {
            const uint32_t t = t0 + u;
            if (t < n_tok) out[(uint64_t)t * out_dim + row] = acc[u];
        }
    }
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
typedef _Float16 __attribute__((ext_vector_type(16))) ds4_q8_half16_t;
typedef float    __attribute__((ext_vector_type(8)))  ds4_q8_float8_t;

/* Four-wave, 64x64 output-tile Q8_0 batched GEMM for large prefill chunks.
 * This is the hipfire/llama.cpp-style MMQ shape adapted to DS4's existing
 * F32 activation buffers: each block stages a 64-token x 32-K activation tile
 * into LDS as f16, while each wave owns 16 output rows and computes four
 * 16-token WMMA columns.  It is opt-in from host code because it only wins once
 * the token batch is large enough to amortize the bigger tile. */
__launch_bounds__(128, 2)
__global__ static void matmul_q8_0_f32_batch_wmma_4w_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_tokens,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes) {
    constexpr uint32_t M_TILE = 64u;
    constexpr uint32_t N_TILE = 64u;
    constexpr uint32_t K_TILE = 32u;
    constexpr uint32_t WARPS = 4u;
    constexpr uint32_t M_PER_WARP = M_TILE / WARPS;
    constexpr uint32_t N_TILES_PER_WARP = N_TILE / 16u;

    const uint32_t block_m = (uint32_t)blockIdx.x * M_TILE;
    const uint32_t block_n = (uint32_t)blockIdx.y * N_TILE;
    if (block_m >= out_dim || block_n >= n_tokens) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t warp_m = block_m + warp_id * M_PER_WARP;
    const uint32_t my_row = warp_m + lane16;
    const uint32_t safe_row = my_row < out_dim ? my_row : (out_dim - 1u);
    const unsigned char *row_base = w + (uint64_t)safe_row * row_bytes;
    const uint32_t n_blocks = in_dim >> 5u;

    ds4_q8_float8_t acc0 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ds4_q8_float8_t acc1 = acc0;
    ds4_q8_float8_t acc2 = acc0;
    ds4_q8_float8_t acc3 = acc0;

    __shared__ _Float16 lds_x[N_TILE * K_TILE];

    for (uint32_t bi = 0; bi < n_blocks; bi++) {
        for (uint32_t j = tid; j < N_TILE * K_TILE; j += blockDim.x) {
            const uint32_t nt = j >> 5u;
            const uint32_t kk = j & 31u;
            const uint32_t tok = block_n + nt;
            float xv = 0.0f;
            if (tok < n_tokens) xv = x[(uint64_t)tok * in_dim + bi * 32u + kk];
            lds_x[j] = (_Float16)xv;
        }
        __syncthreads();

        const unsigned char *bp = row_base + (uint64_t)bi * 34u;
        _Float16 sc;
        {
            uint16_t s_bits;
            __builtin_memcpy(&s_bits, bp, 2);
            __builtin_memcpy(&sc, &s_bits, 2);
        }

        const int8_t *w0 = (const int8_t *)(bp + 2u);
        const int8_t *w1 = (const int8_t *)(bp + 18u);
        ds4_q8_half16_t a0;
        ds4_q8_half16_t a1;
#pragma unroll
        for (uint32_t i = 0; i < 16u; i++) {
            a0[i] = sc * (_Float16)(float)(int)w0[i];
            a1[i] = sc * (_Float16)(float)(int)w1[i];
        }

#pragma unroll
        for (uint32_t ntile = 0; ntile < N_TILES_PER_WARP; ntile++) {
            const uint32_t nt = ntile * 16u + lane16;
            const _Float16 *xb = lds_x + nt * K_TILE;
            const ds4_q8_half16_t b0 = *(const ds4_q8_half16_t *)(xb);
            const ds4_q8_half16_t b1 = *(const ds4_q8_half16_t *)(xb + 16u);
            if (ntile == 0u) {
                acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc0);
                acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc0);
            } else if (ntile == 1u) {
                acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc1);
                acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc1);
            } else if (ntile == 2u) {
                acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc2);
                acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc2);
            } else {
                acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc3);
                acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc3);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t ntile = 0; ntile < N_TILES_PER_WARP; ntile++) {
        const uint32_t tok = block_n + ntile * 16u + lane16;
        if (tok >= n_tokens) continue;
        ds4_q8_float8_t acc = ntile == 0u ? acc0 : (ntile == 1u ? acc1 : (ntile == 2u ? acc2 : acc3));
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t row = warp_m + 2u * j + (lane >> 4u);
            if (row < out_dim) out[(uint64_t)tok * out_dim + row] = acc[j];
        }
    }
}

#endif

__global__ static void matmul_q8_0_hc_expand_preq_rows_w32_kernel(
    float* out_hc, float* block_out, const float* block_add,
    const float* residual_hc, const float* split, const unsigned char* w,
    const int8_t* xq, const float* xscale, uint64_t in_dim, uint64_t out_dim,
    uint32_t n_embd, uint32_t n_hc, uint64_t blocks, uint32_t rows_per_block,
    int has_add, int use_dp4a) {
  const uint64_t row =
      (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
  const uint32_t lane = threadIdx.x & 31u;
  if (row >= out_dim)
    return;
  const unsigned char* wr = w + row * blocks * 34u;
  float acc = 0.0f;
  for (uint64_t b = lane; b < blocks; b += 32u) {
    const uint64_t i0 = b * 32u;
    const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
    const __half* scale_h = (const __half*)(wr + b * 34u);
    const int8_t* qs = (const int8_t*)(wr + b * 34u + 2u);
    const int8_t* xqb = xq + b * 32u;
    const int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
    acc += __half2float(*scale_h) * xscale[b] * (float)dot;
  }
  acc = warp_sum_f32(acc);
  if (lane == 0u) {
    const uint32_t d = (uint32_t)row;
    block_out[d] = acc;
    float block_v = acc;
    if (has_add)
      block_v += block_add[d];
    const float* post = split + n_hc;
    const float* comb = split + 2u * n_hc;
    for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
      float hc_acc = block_v * post[dst_hc];
      for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        hc_acc += residual_hc[(uint64_t)src_hc * n_embd + d] *
                  comb[(uint64_t)src_hc * n_hc + dst_hc];
      }
      out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
    }
  }
}

__global__ static void grouped_q8_0_a_f32_batch_warp8_kernel(
    float* low, const unsigned char* w, const float* heads, uint64_t group_dim,
    uint64_t rank, uint32_t n_groups, uint32_t n_tokens, uint64_t blocks) {
  const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
  const uint64_t tok = (uint64_t)blockIdx.y;
  const uint32_t lane = threadIdx.x & 31u;
  const uint64_t low_dim = (uint64_t)n_groups * rank;
  if (row >= low_dim || tok >= n_tokens)
    return;
  const uint64_t group = row / rank;
  const uint64_t row_in_group = row - group * rank;
  const unsigned char* wr = w + (group * rank + row_in_group) * blocks * 34u;
  const float* x = heads + (tok * (uint64_t)n_groups + group) * group_dim;
  float acc = 0.0f;
  for (uint64_t b = 0; b < blocks; b++) {
    const uint64_t i = b * 32u + lane;
    if (i < group_dim) {
      const unsigned char* blk = wr + b * 34u;
      const float d = q8_0_scale_broadcast_w32(blk);
      const int8_t q = ((const int8_t*)(blk + 2u))[lane];
      acc += d * (float)q * x[i];
    }
  }
  acc = warp_sum_f32(acc);
  if (lane == 0)
    low[tok * low_dim + row] = acc;
}

template<uint32_t TOK_TILE, uint32_t BLOCKS_TILE>
__global__ static void grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel(
    float* low, const unsigned char* w, const float* heads, uint32_t n_tokens,
    uint32_t n_groups, uint32_t n_blocks, uint32_t rank, uint64_t row_bytes) {
  extern __shared__ float shx[];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t wave = tid >> 5u;
  const uint32_t rows_per_block = blockDim.x >> 5u;
  const uint32_t row_blocks = (rank + rows_per_block - 1u) / rows_per_block;
  const uint32_t g = blockIdx.x / row_blocks;
  const uint32_t row0 = (blockIdx.x - g * row_blocks) * rows_per_block + wave;
  const uint32_t t0 = blockIdx.y * TOK_TILE;
  if (g >= n_groups || t0 >= n_tokens)
    return;
  const uint32_t group_dim = n_blocks << 5u;
  const bool row_valid = row0 < rank;
  const unsigned char* wr =
      w + ((uint64_t)g * rank + (row_valid ? row0 : 0u)) * row_bytes;
  float acc[TOK_TILE];
#pragma unroll
  for (uint32_t u = 0; u < TOK_TILE; u++)
    acc[u] = 0.0f;

  for (uint32_t b0 = 0; b0 < n_blocks; b0 += BLOCKS_TILE) {
    const uint32_t b_count =
        ((b0 + BLOCKS_TILE) <= n_blocks) ? BLOCKS_TILE : (n_blocks - b0);
    for (uint32_t j = tid; j < TOK_TILE * BLOCKS_TILE * 32u; j += blockDim.x) {
      const uint32_t u = j / (BLOCKS_TILE * 32u);
      const uint32_t r = j - u * (BLOCKS_TILE * 32u);
      const uint32_t bb = r >> 5u;
      const uint32_t k = r & 31u;
      const uint32_t t = t0 + u;
      const uint64_t xoff = ((uint64_t)t * n_groups + g) * group_dim +
                            ((uint64_t)(b0 + bb) << 5u) + k;
      shx[j] = (t < n_tokens && bb < b_count) ? heads[xoff] : 0.0f;
    }
    __syncthreads();
    if (row_valid) {
      for (uint32_t bb = 0; bb < b_count; bb++) {
        const unsigned char* blk = wr + (uint64_t)(b0 + bb) * 34u;
        const float d = q8_0_scale_broadcast_w32(blk);
        const int8_t q = ((const int8_t*)(blk + 2u))[lane];
        const float wv = d * (float)q;
#pragma unroll
        for (uint32_t u = 0; u < TOK_TILE; u++)
          acc[u] += wv * shx[(u * BLOCKS_TILE + bb) * 32u + lane];
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t u = 0; u < TOK_TILE; u++)
    acc[u] = warp_sum_f32(acc[u]);
  if (lane == 0u && row_valid) {
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) {
      const uint32_t t = t0 + u;
      if (t < n_tokens)
        low[((uint64_t)t * n_groups + g) * rank + row0] = acc[u];
    }
  }
}

/*
 * Variant of the grouped shared-X kernel for inputs whose logical groups are
 * slices of a wider physical row. Consecutive groups can remain a fixed
 * physical stride apart.
 */
template <uint32_t TOK_TILE, uint32_t BLOCKS_TILE>
__global__ static void grouped_q8_0_a_f32_batch_sharedx_chunked_strided_w32_kernel(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t n_blocks,
        uint32_t rank,
        uint32_t x_token_stride,
        uint32_t x_group_stride,
        uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row_blocks = (rank + rows_per_block - 1u) / rows_per_block;
    const uint32_t g = blockIdx.x / row_blocks;
    const uint32_t row0 = (blockIdx.x - g * row_blocks) * rows_per_block + wave;
    const uint32_t t0 = blockIdx.y * TOK_TILE;
    if (g >= n_groups || t0 >= n_tokens) return;
    const bool row_valid = row0 < rank;
    const unsigned char *wr =
        w + ((uint64_t)g * rank + (row_valid ? row0 : 0u)) * row_bytes;
    float acc[TOK_TILE];
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) acc[u] = 0.0f;

    for (uint32_t b0 = 0; b0 < n_blocks; b0 += BLOCKS_TILE) {
        const uint32_t b_count =
            ((b0 + BLOCKS_TILE) <= n_blocks) ? BLOCKS_TILE : (n_blocks - b0);
        for (uint32_t j = tid;
             j < TOK_TILE * BLOCKS_TILE * 32u;
             j += blockDim.x) {
            const uint32_t u = j / (BLOCKS_TILE * 32u);
            const uint32_t r = j - u * (BLOCKS_TILE * 32u);
            const uint32_t bb = r >> 5u;
            const uint32_t k = r & 31u;
            const uint32_t t = t0 + u;
            const uint64_t xoff =
                (uint64_t)t * x_token_stride +
                (uint64_t)g * x_group_stride +
                ((uint64_t)(b0 + bb) << 5u) + k;
            shx[j] =
                (t < n_tokens && bb < b_count) ? heads[xoff] : 0.0f;
        }
        __syncthreads();
        if (row_valid) {
            for (uint32_t bb = 0; bb < b_count; bb++) {
                const unsigned char *blk =
                    wr + (uint64_t)(b0 + bb) * 34u;
                const float d = q8_0_scale_broadcast_w32(blk);
                const int8_t q = ((const int8_t *)(blk + 2u))[lane];
                const float wv = d * (float)q;
#pragma unroll
                for (uint32_t u = 0; u < TOK_TILE; u++) {
                    acc[u] +=
                        wv * shx[(u * BLOCKS_TILE + bb) * 32u + lane];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) {
        acc[u] = warp_sum_f32(acc[u]);
    }
    if (lane == 0u && row_valid) {
#pragma unroll
        for (uint32_t u = 0; u < TOK_TILE; u++) {
            const uint32_t t = t0 + u;
            if (t < n_tokens) {
                low[((uint64_t)t * n_groups + g) * rank + row0] = acc[u];
            }
        }
    }
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#endif

__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    uint64_t row = gid / in_dim;
    uint64_t i = gid - row * in_dim;
    uint64_t b = i / 32;
    uint64_t j = i - b * 32;
    const unsigned char *blk = w + (row * blocks + b) * 34;
    const __half scale = *(const __half *)blk;
    const int8_t q = *(const int8_t *)(blk + 2 + j);
    out[gid] = __hmul(scale, __float2half((float)q));
}

__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    uint64_t row = gid / in_dim;
    uint64_t i = gid - row * in_dim;
    uint64_t b = i / 32;
    uint64_t j = i - b * 32;
    const unsigned char *blk = w + (row * blocks + b) * 34;
    const float scale = q8_0_scale_scalar(blk);
    const int8_t q = *(const int8_t *)(blk + 2 + j);
    out[gid] = scale * (float)q;
}

/* LDS-tiled transpose for the same dequantization.
 *
 * A scalar kernel gives each lane a private output address `i * out_dim + row`,
 * so a wave's 32 stores land 8,192 B apart: 32 separate cache lines for 64 B of
 * payload, and (8192 / 256) % 16 == 0 puts every one of them on a single memory
 * channel. Measured 1.9 GB/s, about 25x off roofline.
 *
 * Here a workgroup owns a 32-`i` by 64-`row` tile. Reads stay wave-contiguous
 * (32 lanes sweep one Q8_0 block's 32 codes), LDS holds the tile transposed,
 * and each store writes 64 consecutive halves. `row` is on `blockIdx.x` so the
 * concurrently dispatched blocks cover the full `out_dim` span of a row and
 * spread across all 16 channels. The arithmetic per element is unchanged, so
 * output is bit-identical. */
__global__ static void dequant_q8_0_to_f16_transpose_tiled_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    __shared__ __half tile[DS4_Q8_T_TILE_I * DS4_Q8_T_LDS_PITCH];
    const uint64_t r0 = (uint64_t)blockIdx.x * DS4_Q8_T_TILE_ROW;
    const uint64_t b = (uint64_t)blockIdx.y;
    const uint64_t i0 = b * DS4_Q8_T_TILE_I;
    const uint32_t tid = threadIdx.x;
    const uint32_t j = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t waves = blockDim.x >> 5u;

    for (uint32_t rl = wave; rl < DS4_Q8_T_TILE_ROW; rl += waves) {
        const uint64_t row = r0 + rl;
        __half v = __float2half(0.0f);
        if (row < out_dim && i0 + j < in_dim) {
            const unsigned char *blk = w + (row * blocks + b) * 34u;
            const __half scale = *(const __half *)blk;
            const int8_t q = *(const int8_t *)(blk + 2u + j);
            v = __hmul(scale, __float2half((float)q));
        }
        tile[j * DS4_Q8_T_LDS_PITCH + rl] = v;
    }
    __syncthreads();

    const uint32_t rows_per_pass = blockDim.x / DS4_Q8_T_TILE_ROW;
    const uint32_t store_row = tid & (DS4_Q8_T_TILE_ROW - 1u);
    for (uint32_t jj = tid / DS4_Q8_T_TILE_ROW; jj < DS4_Q8_T_TILE_I;
         jj += rows_per_pass) {
        const uint64_t i = i0 + jj;
        const uint64_t row = r0 + store_row;
        if (i < in_dim && row < out_dim) {
            out[i * out_dim + row] = tile[jj * DS4_Q8_T_LDS_PITCH + store_row];
        }
    }
}

__global__ static void grouped_q8_0_a_preq_warp8_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks,
        int use_dp4a) {
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim || tok >= n_tokens) return;

    const uint64_t group = row / rank;
    const uint64_t row_in_group = row - group * rank;
    const unsigned char *wr = w + (group * rank + row_in_group) * blocks * 34;
    const uint64_t xrow = tok * (uint64_t)n_groups + group;
    const int8_t *xqr = xq + xrow * blocks * 32;
    const float *xsr = xscale + xrow * blocks;
    float acc = 0.0f;

    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = group_dim - i0 < 32 ? group_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) low[tok * low_dim + row] = acc;
}

/*
 * Session-batch attention output-A projection.
 *
 * Each warp owns one output row and walks the Q8 blocks in the same lane order
 * as grouped_q8_0_a_preq_warp8_kernel. Keeping one accumulator per concurrent
 * session therefore preserves C1's quantization, multiply order, and warp
 * reduction while reusing every weight block across W2-W8.
 */
template <uint32_t MAXT, bool EXACT>
__global__ static void grouped_q8_0_a_preq_batch_reuse_w32_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks,
        uint32_t rows_per_block) {
    const uint64_t row =
        (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim) return;

    const uint64_t group = row / rank;
    const uint64_t row_in_group = row - group * rank;
    const unsigned char *wr =
        w + (group * rank + row_in_group) * blocks * 34u;
    float acc[MAXT];
#pragma unroll
    for (uint32_t t = 0; t < MAXT; ++t) acc[t] = 0.0f;

    for (uint64_t b = lane; b < blocks; b += 32u) {
        const float weight_scale =
            __half2float(*(const __half *)(wr + b * 34u));
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int32_t q0 = load_i8x4_i32_unaligned(qs + 0u);
        const int32_t q1 = load_i8x4_i32_unaligned(qs + 4u);
        const int32_t q2 = load_i8x4_i32_unaligned(qs + 8u);
        const int32_t q3 = load_i8x4_i32_unaligned(qs + 12u);
        const int32_t q4 = load_i8x4_i32_unaligned(qs + 16u);
        const int32_t q5 = load_i8x4_i32_unaligned(qs + 20u);
        const int32_t q6 = load_i8x4_i32_unaligned(qs + 24u);
        const int32_t q7 = load_i8x4_i32_unaligned(qs + 28u);
#pragma unroll
        for (uint32_t t = 0; t < MAXT; ++t) {
            if constexpr (!EXACT) {
                if (t >= n_tokens) break;
            }
            const uint64_t xrow = (uint64_t)t * n_groups + group;
            const int8_t *xqb =
                xq + (xrow * blocks + b) * 32u;
            const float activation_scale =
                xscale[xrow * blocks + b];
            const int dot = dot_i8x32_dp4a_loaded(
                q0, q1, q2, q3, q4, q5, q6, q7, xqb);
            acc[t] +=
                weight_scale * activation_scale * (float)dot;
        }
    }

#pragma unroll
    for (uint32_t t = 0; t < MAXT; ++t) {
        if constexpr (!EXACT) {
            if (t >= n_tokens) break;
        }
        const float sum = warp_sum_f32(acc[t]);
        if (lane == 0u) low[(uint64_t)t * low_dim + row] = sum;
    }
}
