// DS4 ROCm Q8_0 matmul / grouped-output / HC-expand kernels.
//
// Included from ds4_rocm.hip.cpp in the same translation unit so kernel helpers stay
// private/static while we gradually split the custom ROCm backend into modules.

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <rocwmma/rocwmma.hpp>
#endif

#include "ds4_rocm_q8_decode.hip.hpp"

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
 * 16-token WMMA columns. The host selects it when the token batch is large
 * enough to amortize the bigger tile. */
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
