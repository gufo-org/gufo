#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::deepseek_v4_flash::kernels {

__device__ static float dev_f16_to_f32(uint16_t v) {
  return __half2float(*reinterpret_cast<const __half*>(&v));
}
__device__ static float warp_sum_f32(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    v += __shfl_down(v, offset, 32);
#else
    v += __shfl_down_sync(FULL_WARP_MASK, v, offset, 32);
#endif
  }
  return v;
}
__device__ __forceinline__ static void q2_K_scale_broadcast_w32(
    const unsigned char* blk, float* d, float* dmin) {
  float vd = 0.0f;
  float vm = 0.0f;
  if ((threadIdx.x & 31u) == 0u) {
    const uint16_t d_bits = (uint16_t)blk[80] | ((uint16_t)blk[81] << 8);
    const uint16_t dmin_bits = (uint16_t)blk[82] | ((uint16_t)blk[83] << 8);
    vd = dev_f16_to_f32(d_bits);
    vm = dev_f16_to_f32(dmin_bits);
  }
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  *d = __shfl(vd, 0, 32);
  *dmin = __shfl(vm, 0, 32);
#else
  *d = __shfl_sync(FULL_WARP_MASK, vd, 0, 32);
  *dmin = __shfl_sync(FULL_WARP_MASK, vm, 0, 32);
#endif
}
__device__ __forceinline__ static float q2_K_dequant_256_scaled_w32(
    const unsigned char* blk, uint32_t lane, uint32_t kk, float d, float dmin) {
  const unsigned char* sc = blk;
  const unsigned char* qs = blk + 16u;
  const uint32_t g = (lane >> 4u) + (kk << 1u);
  const uint32_t within = g & 7u;
  const uint32_t qi = (g >> 3u) * 32u + (within & 1u) * 16u + (lane & 15u);
  const uint32_t shift = (within >> 1u) * 2u;
  const float q = (float)((qs[qi] >> shift) & 3u);
  const float scale = (float)(sc[g] & 0x0fu);
  const float mn = (float)(sc[g] >> 4u);
  return d * scale * q - dmin * mn;
}
template<uint32_t TILES_PER_PASS>
__global__ static void GroupedQ2Down(
    __half* down_out_h, const char* down_base, const float* mid,
    const uint32_t* expert_offsets, const uint32_t* sorted_tiles,
    const uint32_t* active_count, const uint32_t* active_experts,
    const uint32_t* tile_pair_counts, const uint32_t* tile_pairs,
    uint64_t down_expert_bytes, uint64_t down_row_bytes,
    uint32_t expert_mid_dim, uint32_t out_dim) {
  static_assert(TILES_PER_PASS >= 2u && TILES_PER_PASS <= 4u);
  constexpr uint32_t ROWS_PER_WARP = 2u;
  constexpr uint32_t PAIRS_PER_PASS = TILES_PER_PASS * 4u;
  const uint32_t active = blockIdx.y;
  if (active >= *active_count)
    return;

  const uint32_t expert = active_experts[active];
  const uint32_t first = expert_offsets[expert];
  const uint32_t count = expert_offsets[expert + 1u] - first;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t rows_per_block = blockDim.x >> 5u;
  const uint32_t warp = tid >> 5u;
  const uint32_t row0 = blockIdx.x * rows_per_block * ROWS_PER_WARP + warp;
  uint32_t rows[ROWS_PER_WARP] = {};
  const unsigned char* down_rows[ROWS_PER_WARP] = {};
#pragma unroll
  for (uint32_t row_slot = 0u; row_slot < ROWS_PER_WARP; ++row_slot) {
    rows[row_slot] = row0 + row_slot * rows_per_block;
    down_rows[row_slot] =
        (const unsigned char*)down_base + (uint64_t)expert * down_expert_bytes +
        (uint64_t)(rows[row_slot] < out_dim ? rows[row_slot] : 0u) *
            down_row_bytes;
  }
  const uint32_t n_blocks = expert_mid_dim >> 8u;
  extern __shared__ float shmid[];

  for (uint32_t tile_index = 0u; tile_index < count;
       tile_index += TILES_PER_PASS) {
    uint32_t pair[PAIRS_PER_PASS] = {};
    uint32_t np = 0u;
#pragma unroll
    for (uint32_t tile_slot = 0u; tile_slot < TILES_PER_PASS; ++tile_slot) {
      if (tile_index + tile_slot >= count)
        break;
      const uint32_t tile = sorted_tiles[first + tile_index + tile_slot];
      const uint32_t tile_np = tile_pair_counts[tile];
#pragma unroll
      for (uint32_t p = 0u; p < 4u; ++p) {
        if (p < tile_np) {
          pair[np++] = tile_pairs[(uint64_t)tile * 4u + p];
        }
      }
    }

    float acc[ROWS_PER_WARP][PAIRS_PER_PASS] = {};
    for (uint32_t b = 0u; b < n_blocks; ++b) {
      const uint64_t mid_base = (uint64_t)b * 256u;
      for (uint32_t j = tid; j < np * 256u; j += blockDim.x) {
        const uint32_t p = j >> 8u;
        const uint32_t k = j & 255u;
        shmid[j] = mid[(uint64_t)pair[p] * expert_mid_dim + mid_base + k];
      }
      __syncthreads();

      const unsigned char* weight_blocks[ROWS_PER_WARP] = {};
      float d[ROWS_PER_WARP] = {};
      float dmin[ROWS_PER_WARP] = {};
#pragma unroll
      for (uint32_t r = 0u; r < ROWS_PER_WARP; ++r) {
        weight_blocks[r] = down_rows[r] + static_cast<uint64_t>(b) * 84u;
        q2_K_scale_broadcast_w32(weight_blocks[r], &d[r], &dmin[r]);
      }
      /*
       * Load each staged activation once for both output rows.
       * Each accumulator still visits b/k in the original order,
       * including the same dequantization and F16 boundary.
       */
#pragma unroll
      for (uint32_t k = 0u; k < 8u; ++k) {
        const uint32_t i = lane + (k << 5u);
        float weights[ROWS_PER_WARP] = {};
#pragma unroll
        for (uint32_t r = 0u; r < ROWS_PER_WARP; ++r) {
          weights[r] = q2_K_dequant_256_scaled_w32(weight_blocks[r], lane, k,
                                                   d[r], dmin[r]);
        }
#pragma unroll
        for (uint32_t p = 0u; p < PAIRS_PER_PASS; ++p) {
          if (p < np) {
            const float value = shmid[(p << 8u) + i];
#pragma unroll
            for (uint32_t r = 0u; r < ROWS_PER_WARP; ++r) {
              if (rows[r] < out_dim) {
                acc[r][p] += weights[r] * value;
              }
            }
          }
        }
      }

      __syncthreads();
    }
#pragma unroll
    for (uint32_t row_slot = 0u; row_slot < ROWS_PER_WARP; ++row_slot) {
#pragma unroll
      for (uint32_t p = 0u; p < PAIRS_PER_PASS; ++p) {
        if (p < np) {
          acc[row_slot][p] = warp_sum_f32(acc[row_slot][p]);
          if (lane == 0u && rows[row_slot] < out_dim) {
            down_out_h[(uint64_t)pair[p] * out_dim + rows[row_slot]] =
                __float2half(acc[row_slot][p]);
          }
        }
      }
    }
  }
}

}  // namespace gufo::models::deepseek_v4_flash::kernels
