#pragma once
#include "../iq2_tables.inc"
#include "ds4_rocm_device.hip.hpp"

__device__ static float dev_f16_to_f32(uint16_t v) {
  return __half2float(*reinterpret_cast<const __half*>(&v));
}

__device__ __forceinline__ static uint32_t dev_pack_half2_bits(float x,
                                                               float y) {
  const __half2 h = __floats2half2_rn(x, y);
  return *reinterpret_cast<const uint32_t*>(&h);
}

__device__ __forceinline__ static uint32_t dev_unpack_iq2_signs(uint32_t v) {
  const uint32_t p = __popc(v) & 1u;
  const uint32_t s = v ^ (p << 7u);
  return s * 0x01010101u;
}

__device__ __forceinline__ static int32_t dev_iq2_dp4a_8(uint64_t grid,
                                                         uint32_t sign,
                                                         const int8_t* q8,
                                                         int32_t acc) {
  const uint32_t signs = dev_unpack_iq2_signs(sign);
  const int32_t sm0 = __vcmpne4(signs & 0x08040201u, 0);
  const int32_t sm1 = __vcmpne4(signs & 0x80402010u, 0);
  const int32_t g0 = __vsub4((int32_t)(uint32_t)grid ^ sm0, sm0);
  const int32_t g1 = __vsub4((int32_t)(uint32_t)(grid >> 32) ^ sm1, sm1);
  acc = __dp4a(g0, *(const int32_t*)(q8 + 0), acc);
  acc = __dp4a(g1, *(const int32_t*)(q8 + 4), acc);
  return acc;
}

__device__ static int32_t dev_dot_q2_16(const uint8_t* q2, const int8_t* q8,
                                        int shift) {
  int32_t sum = 0;
#pragma unroll
  for (uint32_t i = 0; i < 16; i += 4) {
    const int32_t v = (*(const int32_t*)(q2 + i) >> shift) & 0x03030303;
    sum = __dp4a(v, *(const int32_t*)(q8 + i), sum);
  }
  return sum;
}

__device__ static int32_t dev_dot_iq2_pair_16(uint8_t grid0, uint32_t sign0,
                                              uint8_t grid1, uint32_t sign1,
                                              const int8_t* q8) {
  int32_t sum = 0;
  sum =
      dev_iq2_dp4a_8(hip_iq2xxs_grid[grid0], hip_ksigns_iq2xs[sign0], q8, sum);
  sum = dev_iq2_dp4a_8(hip_iq2xxs_grid[grid1], hip_ksigns_iq2xs[sign1], q8 + 8,
                       sum);
  return sum;
}

__device__ __forceinline__ static void dev_iq2_i8x8_lut(const uint64_t* grid,
                                                        uint8_t grid_idx,
                                                        uint32_t sign_idx,
                                                        int32_t* w0,
                                                        int32_t* w1) {
  const uint32_t signs = sign_idx | ((__popc(sign_idx) & 1u) << 7u);
  const uint32_t add0 = ((signs & 15u) * 0x00204081u) & 0x01010101u;
  const uint32_t add1 = ((signs >> 4u) * 0x00204081u) & 0x01010101u;
  const uint64_t g = grid[grid_idx];
  // IQ2_XXS grid bytes are nonzero: each negated byte is 256 - g, so these
  // packed additions cannot carry into adjacent bytes. Keep the dot and
  // floating-point reduction order unchanged.
  *w0 = static_cast<int32_t>((static_cast<uint32_t>(g) ^ (add0 * 255u)) + add0);
  *w1 = static_cast<int32_t>((static_cast<uint32_t>(g >> 32u) ^ (add1 * 255u)) +
                             add1);
}

__device__ static float dev_dot_iq2_xxs_q8_K_block_lut(
    const hip_block_iq2_xxs* x, const hip_block_q8_K* y, const uint64_t* grid) {
  const float xd = dev_f16_to_f32(x->d);
  const uint16_t* q2 = x->qs;
  const int8_t* q8 = y->qs;
  int32_t bsum = 0;
  for (int ib32 = 0; ib32 < ROCM_QK_K / 32; ib32++) {
    const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
    const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
    q2 += 4;
    const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
    int32_t w[8];
    dev_iq2_i8x8_lut(grid, (uint8_t)(aux0 & 0xffu), (aux1 >> 0) & 127u, &w[0],
                     &w[1]);
    dev_iq2_i8x8_lut(grid, (uint8_t)((aux0 >> 8) & 0xffu), (aux1 >> 7) & 127u,
                     &w[2], &w[3]);
    dev_iq2_i8x8_lut(grid, (uint8_t)((aux0 >> 16) & 0xffu), (aux1 >> 14) & 127u,
                     &w[4], &w[5]);
    dev_iq2_i8x8_lut(grid, (uint8_t)((aux0 >> 24) & 0xffu), (aux1 >> 21) & 127u,
                     &w[6], &w[7]);
    int32_t sumi = 0;
    sumi = __dp4a(w[0], *(const int32_t*)(q8 + ib32 * 32u + 0), sumi);
    sumi = __dp4a(w[1], *(const int32_t*)(q8 + ib32 * 32u + 4), sumi);
    sumi = __dp4a(w[2], *(const int32_t*)(q8 + ib32 * 32u + 8), sumi);
    sumi = __dp4a(w[3], *(const int32_t*)(q8 + ib32 * 32u + 12), sumi);
    sumi = __dp4a(w[4], *(const int32_t*)(q8 + ib32 * 32u + 16), sumi);
    sumi = __dp4a(w[5], *(const int32_t*)(q8 + ib32 * 32u + 20), sumi);
    sumi = __dp4a(w[6], *(const int32_t*)(q8 + ib32 * 32u + 24), sumi);
    sumi = __dp4a(w[7], *(const int32_t*)(q8 + ib32 * 32u + 28), sumi);
    bsum += sumi * ls;
  }
  return 0.125f * xd * y->d * (float)bsum;
}

__device__ static float dev_dot_iq2_xxs_q8_K_block(const hip_block_iq2_xxs* x,
                                                   const hip_block_q8_K* y) {
  const float d = dev_f16_to_f32(x->d) * y->d;
  const uint16_t* q2 = x->qs;
  const int8_t* q8 = y->qs;
  int32_t bsum = 0;
  for (int ib32 = 0; ib32 < ROCM_QK_K / 32; ib32++) {
    const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
    const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
    q2 += 4;
    const uint32_t ls = 2u * (aux1 >> 28) + 1u;
    const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
    const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
    const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
    const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
    int32_t sumi = 0;
    sumi +=
        dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8);
    q8 += 16;
    sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3,
                                (aux1 >> 21) & 127u, q8);
    q8 += 16;
    bsum += sumi * (int32_t)ls;
  }
  return 0.125f * d * (float)bsum;
}

__device__ static void dev_dot_iq2_xxs_q8_K_block8_deq_lut(
    const hip_block_iq2_xxs* x, const hip_block_q8_K* y0,
    const hip_block_q8_K* y1, const hip_block_q8_K* y2,
    const hip_block_q8_K* y3, const hip_block_q8_K* y4,
    const hip_block_q8_K* y5, const hip_block_q8_K* y6,
    const hip_block_q8_K* y7, uint32_t n, float acc[8], const uint64_t* grid) {
  const float xd = dev_f16_to_f32(x->d);
  const uint16_t* q2 = x->qs;
  int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const int8_t* q8[8] = {
      y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL,
      y3 ? y3->qs : NULL, y4 ? y4->qs : NULL, y5 ? y5->qs : NULL,
      y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
  };
  for (int ib32 = 0; ib32 < ROCM_QK_K / 32; ib32++) {
    const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
    const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
    q2 += 4;
    const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
    int32_t w[8];
    dev_iq2_i8x8_lut(grid, (uint8_t)(aux0 & 0xffu), (aux1 >> 0) & 127u, &w[0],
                     &w[1]);
    dev_iq2_i8x8_lut(grid, (uint8_t)((aux0 >> 8) & 0xffu), (aux1 >> 7) & 127u,
                     &w[2], &w[3]);
    dev_iq2_i8x8_lut(grid, (uint8_t)((aux0 >> 16) & 0xffu), (aux1 >> 14) & 127u,
                     &w[4], &w[5]);
    dev_iq2_i8x8_lut(grid, (uint8_t)((aux0 >> 24) & 0xffu), (aux1 >> 21) & 127u,
                     &w[6], &w[7]);
    for (uint32_t p = 0; p < n; p++) {
      const int8_t* q = q8[p] + ib32 * 32;
      int32_t sumi = 0;
      sumi = __dp4a(w[0], *(const int32_t*)(q + 0), sumi);
      sumi = __dp4a(w[1], *(const int32_t*)(q + 4), sumi);
      sumi = __dp4a(w[2], *(const int32_t*)(q + 8), sumi);
      sumi = __dp4a(w[3], *(const int32_t*)(q + 12), sumi);
      sumi = __dp4a(w[4], *(const int32_t*)(q + 16), sumi);
      sumi = __dp4a(w[5], *(const int32_t*)(q + 20), sumi);
      sumi = __dp4a(w[6], *(const int32_t*)(q + 24), sumi);
      sumi = __dp4a(w[7], *(const int32_t*)(q + 28), sumi);
      bsum[p] += sumi * ls;
    }
  }
  const hip_block_q8_K* ys[8] = {y0, y1, y2, y3, y4, y5, y6, y7};
  for (uint32_t p = 0; p < n; p++)
    acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

template<uint32_t N>
__device__ __forceinline__ static void dev_dot_iq2_xxs_q8_K_block4(
    const hip_block_iq2_xxs* x, const hip_block_q8_K* y0,
    const hip_block_q8_K* y1, const hip_block_q8_K* y2,
    const hip_block_q8_K* y3, float acc[4]) {
  static_assert(N >= 1u && N <= 4u);
  const float xd = dev_f16_to_f32(x->d);
  const uint16_t* q2 = x->qs;
  int32_t bsum[4] = {0, 0, 0, 0};
  const int8_t* q8[4] = {
      y0 ? y0->qs : NULL,
      y1 ? y1->qs : NULL,
      y2 ? y2->qs : NULL,
      y3 ? y3->qs : NULL,
  };
  for (int ib32 = 0; ib32 < ROCM_QK_K / 32; ib32++) {
    const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
    const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
    q2 += 4;
    const uint32_t ls = 2u * (aux1 >> 28) + 1u;
    const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
    const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
    const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
    const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
#pragma unroll
    for (uint32_t p = 0; p < N; p++) {
      int32_t sumi = 0;
      sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1,
                                  (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
      sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3,
                                  (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
      bsum[p] += sumi * (int32_t)ls;
    }
  }
  const hip_block_q8_K* ys[4] = {y0, y1, y2, y3};
#pragma unroll
  for (uint32_t p = 0; p < N; p++) {
    // Preserve the scalar LUT kernel's scale-product rounding before FMA.
    acc[p] = fmaf(__fmul_rn(0.125f * xd, ys[p]->d), (float)bsum[p], acc[p]);
  }
}

__global__ static void moe_gate_up_mid_decode_lut_qwarp32_kernel(
    float* gate_out, float* up_out, float* mid_out, const char* gate_base,
    const char* up_base, const hip_block_q8_K* xq, const int32_t* selected,
    const float* weights, uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
    uint32_t xq_blocks, uint32_t expert_mid_dim, uint32_t n_expert,
    uint32_t write_aux, float clamp) {
  uint32_t lane = threadIdx.x & 7u;
  uint32_t row_lane = threadIdx.x >> 3u;
  uint32_t pair = blockIdx.y;
  uint32_t tok = pair / n_expert;
  uint32_t slot = pair - tok * n_expert;
  int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
  if (expert_i < 0)
    expert_i = 0;
  uint32_t expert = (uint32_t)expert_i;
  const hip_block_q8_K* xqb = xq + (uint64_t)tok * xq_blocks;
  __shared__ hip_block_q8_K sxq[16];
  __shared__ uint64_t s_iq2_grid[256];
  if (xq_blocks <= 16u) {
    for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x)
      sxq[i] = xqb[i];
    xqb = sxq;
  }
  for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x)
    s_iq2_grid[i] = hip_iq2xxs_grid[i];
  __syncthreads();
  for (uint32_t rr = 0; rr < 4u; rr++) {
    uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
    if (row >= expert_mid_dim)
      continue;
    const hip_block_iq2_xxs* gr =
        (const hip_block_iq2_xxs*)(gate_base +
                                   (uint64_t)expert * gate_expert_bytes +
                                   (uint64_t)row * gate_row_bytes);
    const hip_block_iq2_xxs* ur =
        (const hip_block_iq2_xxs*)(up_base +
                                   (uint64_t)expert * gate_expert_bytes +
                                   (uint64_t)row * gate_row_bytes);
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
      gate += dev_dot_iq2_xxs_q8_K_block_lut(gr + b, xqb + b, s_iq2_grid);
      up += dev_dot_iq2_xxs_q8_K_block_lut(ur + b, xqb + b, s_iq2_grid);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
      if (clamp > 1.0e-6f) {
        if (gate > clamp)
          gate = clamp;
        if (up > clamp)
          up = clamp;
        if (up < -clamp)
          up = -clamp;
      }
      const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
      if (write_aux) {
        gate_out[off] = gate;
        up_out[off] = up;
      }
      mid_out[off] = (gate / (1.0f + expf(-gate))) * up *
                     weights[(uint64_t)tok * n_expert + slot];
    }
  }
}

__global__ static void moe_gate_up_mid_expert_tile4_row32_kernel(
    float* gate_out, float* up_out, float* mid_out, const char* gate_base,
    const char* up_base, const hip_block_q8_K* xq, const uint32_t* sorted_pairs,
    const uint32_t* offsets, const uint32_t* counts, const uint32_t* tile_total,
    const uint32_t* tile_experts, const uint32_t* tile_starts,
    const float* weights, uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
    uint32_t xq_blocks, uint32_t expert_mid_dim, uint32_t n_expert,
    uint32_t max_count, uint32_t write_aux, float clamp) {
  uint32_t tile = blockIdx.y;
  if (tile >= *tile_total)
    return;
  uint32_t lane = threadIdx.x & 7u;
  uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
  uint32_t expert = tile_experts[tile];
  uint32_t count = counts[expert];
  if (max_count != 0u && count >= max_count)
    return;
  uint32_t local_start = tile_starts[tile];
  uint32_t pair[4] = {0, 0, 0, 0};
  uint32_t tok[4] = {0, 0, 0, 0};
  uint32_t slot[4] = {0, 0, 0, 0};
  const hip_block_q8_K* xqb[4] = {NULL, NULL, NULL, NULL};
  uint32_t np = 0;
  for (; np < 4u; np++) {
    uint32_t local_pair = local_start + np;
    if (local_pair >= count)
      break;
    pair[np] = sorted_pairs[offsets[expert] + local_pair];
    tok[np] = pair[np] / n_expert;
    slot[np] = pair[np] - tok[np] * n_expert;
    xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
  }
  if (row >= expert_mid_dim)
    return;
  const hip_block_iq2_xxs* gr =
      (const hip_block_iq2_xxs*)(gate_base +
                                 (uint64_t)expert * gate_expert_bytes +
                                 (uint64_t)row * gate_row_bytes);
  const hip_block_iq2_xxs* ur =
      (const hip_block_iq2_xxs*)(up_base +
                                 (uint64_t)expert * gate_expert_bytes +
                                 (uint64_t)row * gate_row_bytes);
  float gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint32_t b = lane; b < xq_blocks; b += 8u) {
    const hip_block_q8_K* y0 = xqb[0] ? xqb[0] + b : NULL;
    const hip_block_q8_K* y1 = xqb[1] ? xqb[1] + b : NULL;
    const hip_block_q8_K* y2 = xqb[2] ? xqb[2] + b : NULL;
    const hip_block_q8_K* y3 = xqb[3] ? xqb[3] + b : NULL;
    if (np == 1u) {
      dev_dot_iq2_xxs_q8_K_block4<1u>(gr + b, y0, y1, y2, y3, gate);
      dev_dot_iq2_xxs_q8_K_block4<1u>(ur + b, y0, y1, y2, y3, up);
    } else if (np == 2u) {
      dev_dot_iq2_xxs_q8_K_block4<2u>(gr + b, y0, y1, y2, y3, gate);
      dev_dot_iq2_xxs_q8_K_block4<2u>(ur + b, y0, y1, y2, y3, up);
    } else if (np == 3u) {
      dev_dot_iq2_xxs_q8_K_block4<3u>(gr + b, y0, y1, y2, y3, gate);
      dev_dot_iq2_xxs_q8_K_block4<3u>(ur + b, y0, y1, y2, y3, up);
    } else {
      dev_dot_iq2_xxs_q8_K_block4<4u>(gr + b, y0, y1, y2, y3, gate);
      dev_dot_iq2_xxs_q8_K_block4<4u>(ur + b, y0, y1, y2, y3, up);
    }
  }
  for (uint32_t p = 0; p < np; p++) {
    gate[p] = quarter_warp_sum_f32(gate[p], lane);
    up[p] = quarter_warp_sum_f32(up[p], lane);
    if (lane == 0) {
      if (clamp > 1.0e-6f) {
        if (gate[p] > clamp)
          gate[p] = clamp;
        if (up[p] > clamp)
          up[p] = clamp;
        if (up[p] < -clamp)
          up[p] = -clamp;
      }
      const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
      if (write_aux) {
        gate_out[off] = gate[p];
        up_out[off] = up[p];
      }
      mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] *
                     weights[(uint64_t)tok[p] * n_expert + slot[p]];
    }
  }
}
