// Wave32 split batched Q4_K MoE gate/up kernels (compiled without -mwavefrontsize64).
#include "qfn_mmq_prelude.h"
namespace qfn_mmq {
#include "mmvq.hpp"
#include "unary.hpp"
#include "vecdotq.hpp"

namespace {

static __device__ __forceinline__ void load_q8_pair(
    const block_q8_1* __restrict__ bq8_1, int iqs, int u[4], float d8[2],
    int dot2[2]) {
  const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
#pragma unroll
  for (int i = 0; i < QR4_K; ++i) {
    const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
    d8[i] = __low2float(bq8i->ds);
    const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
    u[2 * i + 0] = q8[0];
    u[2 * i + 1] = q8[4];
    dot2[i] = ggml_hip_dp4a(0x01010101, u[2 * i + 1],
                            ggml_hip_dp4a(0x01010101, u[2 * i + 0], 0));
  }
}

static __device__ __forceinline__ float dot_q4k_preloaded(
    const void* __restrict__ vbq, int kbx, int iqs, const int u[4],
    const float d8[2], const int dot2[2]) {
  const block_q4_K* bq4_K = static_cast<const block_q4_K*>(vbq) + kbx;
  const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
  const int* q4 = reinterpret_cast<const int*>(
      bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  const int v0 = q4[0];
  const int v1 = q4[4];
  uint16_t aux[2];
  q4_K_scale_pair(bq4_K, bq8_offset / 2, aux);
  const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
  const uint8_t* m = sc + 2;
  float sumf_d = 0.0f;
  float sumf_m = 0.0f;
#pragma unroll
  for (int i = 0; i < QR4_K; ++i) {
    const int v0i = (v0 >> (4 * i)) & 0x0F0F0F0F;
    const int v1i = (v1 >> (4 * i)) & 0x0F0F0F0F;
    const int dot1 = ggml_hip_dp4a(v1i, u[2 * i + 1],
                                   ggml_hip_dp4a(v0i, u[2 * i + 0], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2[i] * m[i]);
  }
  const float2 dm4f = __half22float2(bq4_K->dm);
  return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

static __device__ __forceinline__ float dot_q4k_frag_preloaded(
    const Q4MoeFragment& frag, const int u[4], const float d8[2],
    const int dot2[2]) {
  const uint8_t* sc = reinterpret_cast<const uint8_t*>(frag.aux);
  const uint8_t* m = sc + 2;
  float sumf_d = 0.0f;
  float sumf_m = 0.0f;
#pragma unroll
  for (int i = 0; i < QR4_K; ++i) {
    const int v0i = (frag.v[0] >> (4 * i)) & 0x0F0F0F0F;
    const int v1i = (frag.v[1] >> (4 * i)) & 0x0F0F0F0F;
    const int dot1 = ggml_hip_dp4a(v1i, u[2 * i + 1],
                                   ggml_hip_dp4a(v0i, u[2 * i + 0], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2[i] * m[i]);
  }
  const float2 dm4f = __half22float2(frag.dm);
  return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

}  // namespace

__launch_bounds__(32) static __global__ void mul_mat_vec_moe_batch_split_single_w32(
    const void* __restrict__ gate, const void* __restrict__ up,
    const block_q8_1* __restrict__ input, const int32_t* __restrict__ storage,
    float* __restrict__ output, int k, int rows, int n_tokens, int experts_used,
    int input_stride) {
  const int count = storage[1];
  if (blockIdx.y >= count)
    return;
  const auto* multi_groups = reinterpret_cast<const MoeBatchGroup*>(storage + 2);
  const auto* groups = multi_groups + n_tokens * experts_used;
  constexpr int qk = QK_K;
  constexpr int qi = QI4_K;
  constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
  constexpr int blocks_per_iter = vdr * 32 / qi;
  const int lane = threadIdx.x;
  const int row0 = blockIdx.x * 4;
  const int r0 = row0;
  const int r1 = min(row0 + 1, rows - 1);
  const int r2 = min(row0 + 2, rows - 1);
  const int r3 = min(row0 + 3, rows - 1);
  const int blocks_per_row = k / qk;
  const int kqs = vdr * (lane % (qi / vdr));
  for (int anchor = blockIdx.y; anchor < count; anchor += gridDim.y) {
    const auto& group = groups[anchor];
    const int expert = group.expert;
    const int token = group.token[0];
    const uint32_t mask = group.slots[0];
    float sum_g0 = 0.0f, sum_u0 = 0.0f;
    float sum_g1 = 0.0f, sum_u1 = 0.0f;
    float sum_g2 = 0.0f, sum_u2 = 0.0f;
    float sum_g3 = 0.0f, sum_u3 = 0.0f;
    for (int kb = lane / (qi / vdr); kb < blocks_per_row;
         kb += blocks_per_iter) {
      const auto* x = input + token * input_stride + kb * (qk / QK8_1);
      int u[4];
      float d8[2];
      int dot2[2];
      load_q8_pair(x, kqs, u, d8, dot2);
      const int b0 = (expert * rows + r0) * blocks_per_row + kb;
      const int b1 = (expert * rows + r1) * blocks_per_row + kb;
      const int b2 = (expert * rows + r2) * blocks_per_row + kb;
      const int b3 = (expert * rows + r3) * blocks_per_row + kb;
      sum_g0 += dot_q4k_preloaded(gate, b0, kqs, u, d8, dot2);
      sum_u0 += dot_q4k_preloaded(up, b0, kqs, u, d8, dot2);
      sum_g1 += dot_q4k_preloaded(gate, b1, kqs, u, d8, dot2);
      sum_u1 += dot_q4k_preloaded(up, b1, kqs, u, d8, dot2);
      sum_g2 += dot_q4k_preloaded(gate, b2, kqs, u, d8, dot2);
      sum_u2 += dot_q4k_preloaded(up, b2, kqs, u, d8, dot2);
      sum_g3 += dot_q4k_preloaded(gate, b3, kqs, u, d8, dot2);
      sum_u3 += dot_q4k_preloaded(up, b3, kqs, u, d8, dot2);
    }
    sum_g0 = warp_reduce_sum<32>(sum_g0);
    sum_u0 = warp_reduce_sum<32>(sum_u0);
    sum_g1 = warp_reduce_sum<32>(sum_g1);
    sum_u1 = warp_reduce_sum<32>(sum_u1);
    sum_g2 = warp_reduce_sum<32>(sum_g2);
    sum_u2 = warp_reduce_sum<32>(sum_u2);
    sum_g3 = warp_reduce_sum<32>(sum_g3);
    sum_u3 = warp_reduce_sum<32>(sum_u3);
    if (lane < 4 && row0 + lane < rows) {
      const float rg = lane == 0 ? sum_g0 : lane == 1 ? sum_g1 : lane == 2 ? sum_g2 : sum_g3;
      const float ru = lane == 0 ? sum_u0 : lane == 1 ? sum_u1 : lane == 2 ? sum_u2 : sum_u3;
      const float g = isfinite(rg) ? rg : 0.0f;
      const float u = isfinite(ru) ? ru : 0.0f;
      const float val = (g * (1.0f / (1.0f + __expf(-g)))) * u;
      uint32_t bits = mask;
      while (bits) {
        const int slot = __ffs(static_cast<int>(bits)) - 1;
        output[(token * experts_used + slot) * rows + row0 + lane] = val;
        bits &= bits - 1;
      }
    }
  }
}

__launch_bounds__(32) static __global__ void mul_mat_vec_moe_batch_split_multi_w32(
    const void* __restrict__ gate, const void* __restrict__ up,
    const block_q8_1* __restrict__ input, const int32_t* __restrict__ storage,
    float* __restrict__ output, int k, int rows, int experts_used,
    int input_stride) {
  constexpr int tokens = MMVQ_MAX_BATCH_SIZE;
  const int count = storage[0];
  if (blockIdx.y >= count)
    return;
  const auto* groups = reinterpret_cast<const MoeBatchGroup*>(storage + 2);
  constexpr int qk = QK_K;
  constexpr int qi = QI4_K;
  constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
  constexpr int blocks_per_iter = vdr * 32 / qi;
  const int lane = threadIdx.x;
  const int row = blockIdx.x;
  const int blocks_per_row = k / qk;
  const int kqs = vdr * (lane % (qi / vdr));
  for (int anchor = blockIdx.y; anchor < count; anchor += gridDim.y) {
    const auto& group = groups[anchor];
    const int expert = group.expert;
    if (expert < 0) {
      if (expert == -2 && lane == 0)
        output[group.anchor * rows + row] = 0.0f;
      continue;
    }
    int tok[tokens];
    uint32_t slots[tokens];
    int ntok = 0;
#pragma unroll
    for (int t = 0; t < tokens; ++t) {
      tok[t] = group.token[t];
      slots[t] = group.slots[t];
      if (slots[t] != 0)
        ++ntok;
    }
    float sum_g[tokens] = {};
    float sum_u[tokens] = {};
    for (int kb = lane / (qi / vdr); kb < blocks_per_row;
         kb += blocks_per_iter) {
      const int blk = (expert * rows + row) * blocks_per_row + kb;
      const Q4MoeFragment wg(gate, blk, kqs);
      const Q4MoeFragment wu(up, blk, kqs);
#pragma unroll
      for (int t = 0; t < tokens; ++t) {
        if (t >= ntok)
          break;
        int u[4];
        float d8[2];
        int dot2[2];
        load_q8_pair(input + tok[t] * input_stride + kb * (qk / QK8_1), kqs, u,
                     d8, dot2);
        sum_g[t] += dot_q4k_frag_preloaded(wg, u, d8, dot2);
        sum_u[t] += dot_q4k_frag_preloaded(wu, u, d8, dot2);
      }
    }
#pragma unroll
    for (int t = 0; t < tokens; ++t) {
      if (t >= ntok)
        break;
      const float rg = warp_reduce_sum<32>(sum_g[t]);
      const float ru = warp_reduce_sum<32>(sum_u[t]);
      if (lane == 0) {
        const float g = isfinite(rg) ? rg : 0.0f;
        const float u = isfinite(ru) ? ru : 0.0f;
        const float val = (g * (1.0f / (1.0f + __expf(-g)))) * u;
        uint32_t bits = slots[t];
        while (bits) {
          const int slot = __ffs(static_cast<int>(bits)) - 1;
          output[(tok[t] * experts_used + slot) * rows + row] = val;
          bits &= bits - 1;
        }
      }
    }
  }
}

void mul_mat_vec_moe_batch_wave32_split(const void* gate, const void* up,
                                       const block_q8_1* input,
                                       const int32_t* storage, float* output,
                                       int k, int rows, int tokens,
                                       int experts_used, int input_stride,
                                       hipStream_t stream) {
  const int total_slots = tokens * experts_used;
  const int multi_y = min(64, max(1, total_slots / 2));
  const int single_y = min(256, total_slots);
  mul_mat_vec_moe_batch_split_multi_w32<<<dim3(rows, multi_y), 32, 0, stream>>>(
      gate, up, input, storage, output, k, rows, experts_used, input_stride);
  mul_mat_vec_moe_batch_split_single_w32
      <<<dim3((rows + 3) / 4, single_y), 32, 0, stream>>>(
          gate, up, input, storage, output, k, rows, tokens, experts_used,
          input_stride);
}

}  // namespace qfn_mmq
