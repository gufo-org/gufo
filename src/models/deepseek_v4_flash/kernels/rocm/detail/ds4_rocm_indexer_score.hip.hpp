#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <rocwmma/rocwmma.hpp>

// DS4 Flash indexer: 64 heads of width 128, one 128-thread block per score.
__global__ static void indexer_score_one_direct_kernel(
    float* scores, const float* q, const float* weights,
    const float* index_comp, uint32_t n_comp, uint32_t pos0, uint32_t ratio,
    float scale, int causal) {
  const uint32_t token = blockIdx.y;
  scores += (uint64_t)token * n_comp;
  q += (uint64_t)token * 64u * 128u;
  weights += (uint64_t)token * 64u;
  pos0 += token;
  const uint32_t c = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t warp = tid >> 5u;
  if (c >= n_comp || tid >= 128u)
    return;
  if (causal) {
    const uint32_t visible = ratio ? (pos0 + 1u) / ratio : n_comp;
    if (c >= visible) {
      if (tid == 0)
        scores[c] = -INFINITY;
      return;
    }
  }

  __shared__ float krow[128];
  // Each head owns its slot: no inter-warp synchronization inside the loop.
  __shared__ float partial[64];
  if (tid < 128u)
    krow[tid] = index_comp[(uint64_t)c * 128u + tid];
  __syncthreads();

  for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
    const uint32_t h = h0 + warp;
    const float4 qv = ((const float4*)(q + (uint64_t)h * 128u))[lane];
    const float4 kv = ((const float4*)krow)[lane];
    float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
    for (int offset = 16; offset > 0; offset >>= 1)
      dot += __shfl_down(dot, offset, 32);
    if (lane == 0)
      partial[h] = fmaxf(dot, 0.0f) * weights[h] * scale;
  }
  __syncthreads();
  if (tid == 0) {
#pragma clang fp reassociate(off)
    float total = 0.0f;
    // Preserve the qualified compiler's sequential head accumulation.
    // Fast-math must not regroup four heads before adding the running sum.
    for (uint32_t h = 0; h < 64u; h += 4u) {
      total += partial[h];
      total += partial[h + 1];
      total += partial[h + 2];
      total += partial[h + 3];
    }
    scores[c] = total;
  }
}

// Pad the last 128-key tile with zeroes. The F16 conversion is identical
// to the original per-block conversion; persistent indexer keys stay F32.
__global__ static void indexer_pack_keys_kernel(const float* keys,
                                                __half* packed, uint32_t count,
                                                uint64_t padded_elements) {
  const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < padded_elements)
    packed[i] = __float2half_rn(i < uint64_t(count) * 128u ? keys[i] : 0.0F);
}

__global__ static void indexer_pack_queries_kernel(const float* q,
                                                   __half* packed,
                                                   uint32_t tokens) {
  __shared__ __half tile[128][18];
  // Adjacent blocks visit adjacent heads in the source token rows.
  const uint32_t tid = threadIdx.x, token0 = blockIdx.y * 16, h = blockIdx.x;
  for (uint32_t i = tid; i < 512; i += 256) {
    const uint32_t row = i / 32, d = (i % 32) * 4;
    const float4 x = token0 + row < tokens
                         ? *reinterpret_cast<const float4*>(
                               q + (uint64_t(token0 + row) * 64 + h) * 128 + d)
                         : make_float4(0, 0, 0, 0);
    tile[d][row] = __float2half_rn(x.x);
    tile[d + 1][row] = __float2half_rn(x.y);
    tile[d + 2][row] = __float2half_rn(x.z);
    tile[d + 3][row] = __float2half_rn(x.w);
  }
  __syncthreads();
  const uint32_t d = tid / 2, row = (tid % 2) * 8;
  union {
    uint4 packed;
    __half halves[8];
  } value;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i)
    value.halves[i] = tile[d][row + i];
  *reinterpret_cast<uint4*>(packed + (uint64_t(blockIdx.y) * 64 + h) * 2048 +
                            d * 16 + row) = value.packed;
}

// Each wave keeps its eight key fragments across all 64 query heads. Packed
// queries feed WMMA directly, avoiding per-head LDS staging and barriers.
// Callers without enough scratch retain the shared-memory route. Both use
// the same F16 operands, K16 products and head reduction order.
template<bool kCached>
__global__ static void indexer_scores_wmma128_kernel(
    float* scores, const void* query_input, const float* weights,
    const void* index_input, uint32_t n_comp, uint32_t n_tokens, uint32_t pos0,
    uint32_t n_head, uint32_t head_dim, uint32_t ratio, float scale,
    int causal) {
#if defined(__HIP_DEVICE_COMPILE__)
  namespace wmma = rocwmma;
  const auto* q = static_cast<const float*>(query_input);
  const auto* index_comp = static_cast<const float*>(index_input);
  const uint32_t tile_c = blockIdx.x * 128u;
  const uint32_t tile_t = blockIdx.y * 16u;
  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid >> 5u;
  if (tid >= 256u || head_dim != 128u)
    return;

  if (causal) {
    const uint32_t last_token = min(tile_t + 16u, n_tokens);
    const uint32_t max_visible =
        last_token > tile_t ? min((pos0 + last_token) / ratio, n_comp) : 0u;
    if (tile_c >= max_visible) {
      for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
        const uint32_t r = i >> 7u;
        const uint32_t c = i & 127u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + c;
        if (token < n_tokens && comp < n_comp) {
          scores[(uint64_t)token * n_comp + comp] = -INFINITY;
        }
      }
      return;
    }
  }

  constexpr uint32_t QT_PITCH = 18u;
  __shared__ __half a_sh[kCached ? 1 : 2][kCached ? 1 : 128 * QT_PITCH];
  __shared__ __half b_sh[kCached ? 1 : 128 * 128];

  const uint32_t lane = tid & 31u;

  const uint32_t acc_col = lane & 15u;
  const uint32_t acc_row_base = lane >> 4u;
  const uint32_t token_own = tile_t + acc_col;
  const uint32_t comp_base = tile_c + warp * 16u + acc_row_base;

  float acc[8];
#pragma unroll
  for (uint32_t i = 0; i < 8u; i++)
    acc[i] = 0.0f;

  if constexpr (!kCached) {
    for (uint32_t i4 = tid; i4 < 128u * 32u; i4 += 256u) {
      const uint32_t c = i4 >> 5u;
      const uint32_t d = (i4 & 31u) * 4u;
      const uint32_t comp = tile_c + c;
      uint2 packed = make_uint2(0u, 0u);
      if (comp < n_comp) {
        const float4 v4 = *reinterpret_cast<const float4*>(
            index_comp + (uint64_t)comp * head_dim + d);
        const __half2 lo = __floats2half2_rn(v4.x, v4.y);
        const __half2 hi = __floats2half2_rn(v4.z, v4.w);
        packed.x = *reinterpret_cast<const uint32_t*>(&lo);
        packed.y = *reinterpret_cast<const uint32_t*>(&hi);
      }
      *reinterpret_cast<uint2*>(&b_sh[d + c * 128u]) = packed;
    }
  }
  if constexpr (!kCached)
    __syncthreads();

  const auto stage_q_head = [&](uint32_t h, __half* dst) {
    for (uint32_t i4 = tid; i4 < 16u * 32u; i4 += 256u) {
      const uint32_t r = i4 >> 5u;
      const uint32_t d = (i4 & 31u) * 4u;
      const uint32_t token = tile_t + r;
      float4 v4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (token < n_tokens) {
        v4 = *reinterpret_cast<const float4*>(
            q + ((uint64_t)token * n_head + h) * head_dim + d);
      }
      dst[(d + 0u) * QT_PITCH + r] = __float2half_rn(v4.x);
      dst[(d + 1u) * QT_PITCH + r] = __float2half_rn(v4.y);
      dst[(d + 2u) * QT_PITCH + r] = __float2half_rn(v4.z);
      dst[(d + 3u) * QT_PITCH + r] = __float2half_rn(v4.w);
    }
  };

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>
      keys[kCached ? 8 : 1];
  if constexpr (kCached) {
#pragma unroll
    for (uint32_t step = 0; step < 8; ++step)
      wmma::load_matrix_sync(keys[step],
                             static_cast<const __half*>(index_input) +
                                 uint64_t(tile_c + warp * 16u) * 128u +
                                 step * 16u,
                             128);
  }
  if constexpr (!kCached) {
    if (n_head != 0u)
      stage_q_head(0u, a_sh[0]);
    __syncthreads();
  }

  for (uint32_t h = 0; h < n_head; h++) {
    const uint32_t cur = h & 1u;

    if constexpr (!kCached)
      if (h + 1u < n_head)
        stage_q_head(h + 1u, a_sh[cur ^ 1u]);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    const uint32_t comp_row0 = warp * 16u;
#pragma unroll
    for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
      if constexpr (kCached)
        a_frag = keys[k0 / 16u];
      else
        wmma::load_matrix_sync(a_frag, b_sh + comp_row0 * 128u + k0, 128);
      if constexpr (kCached)
        wmma::load_matrix_sync(
            b_frag,
            static_cast<const __half*>(query_input) +
                (uint64_t(tile_t / 16u) * n_head + h) * 128u * 16u + k0 * 16u,
            16u);
      else
        wmma::load_matrix_sync(b_frag, a_sh[cur] + k0 * QT_PITCH, QT_PITCH);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    if (token_own < n_tokens) {
      const float w = weights[(uint64_t)token_own * n_head + h];
#pragma unroll
      for (uint32_t e = 0; e < 8u; e++) {
        acc[e] += fmaxf(c_frag.x[e], 0.0f) * w;
      }
    }

    if constexpr (!kCached)
      __syncthreads();
  }

  if (token_own < n_tokens) {
    const uint32_t visible = causal ? (pos0 + token_own + 1u) / ratio : n_comp;
#pragma unroll
    for (uint32_t e = 0; e < 8u; e++) {
      const uint32_t comp = comp_base + 2u * e;
      if (comp < n_comp) {
        float out = acc[e] * scale;
        if (causal && comp >= visible)
          out = -INFINITY;
        scores[(uint64_t)token_own * n_comp + comp] = out;
      }
    }
  }
#endif
}
