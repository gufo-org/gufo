// Batched Q4 experts share one wave64 while retaining 32-lane dot reductions.
#include "qfn_mmq_prelude.h"
#include <cstdlib>
namespace qfn_mmq {
#include "mmvq.hpp"
#include "unary.hpp"
#include "vecdotq.hpp"

namespace {

constexpr bool FastMoeGuEnabled() noexcept { return true; }

}  // namespace

template<bool single_request>
__launch_bounds__(64) static __global__ void mul_mat_vec_moe_batch(
    const void* __restrict__ gate, const void* __restrict__ up,
    const block_q8_1* __restrict__ input, const int32_t* __restrict__ storage,
    float* __restrict__ output, int k, int rows, int experts_used, int input_stride) {
  constexpr int tokens = MMVQ_MAX_BATCH_SIZE;
  const int count = storage[0];
  const int32_t* groups = storage + 1;
  // Every block consumes a bounded slice of the compact device list.
  // Keep arithmetic in the kernel body: extracting a shared device helper
  // changes FP32 contraction on the production compiler.
  for (int anchor = blockIdx.y; anchor < count; anchor += gridDim.y) {
    constexpr int qk = QK_K;
    constexpr int qi = QI4_K;
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int blocks_per_iter = vdr * 32 / qi;
    constexpr auto dot = vec_dot_q4_K_q8_1;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const bool is_up = tid >= 32;
    const int row0 = blockIdx.x * 2;
    const auto& group = reinterpret_cast<const MoeBatchGroup*>(groups)[anchor];
    const int expert = group.expert;
    const auto token_row = [&](int t) { return group.token[t]; };
    const auto token_slots = [&](int t) { return group.slots[t]; };
    if (expert < 0) {
      if (!single_request && expert == -2 && tid < 2 && row0 + tid < rows)
        output[group.anchor * rows + row0 + tid] = 0.0f;
      continue;
    }

    constexpr int columns = single_request ? 1 : tokens;
    int token = 0;
    {
      int active = 0;
#pragma unroll
      for (int t = 0; t < tokens; ++t) {
        if (token_slots(t) != 0) {
          ++active;
          token = t;
        }
      }
      if (single_request ? active != 1 : active <= 1)
        continue;
    }
    const void* weights = is_up ? up : gate;
    const int blocks_per_row = k / qk;
    uint32_t slots[columns];
#pragma unroll
    for (int t = 0; t < columns; ++t)
      slots[t] = token_slots(single_request ? token : t);
    float sum[columns][2] = {};
    for (int kb = lane / (qi / vdr); kb < blocks_per_row; kb += blocks_per_iter) {
      const int kqs = vdr * (lane % (qi / vdr));
      if constexpr (!single_request) {
        const Q4MoeFragment w0(
            weights, (expert * rows + row0) * blocks_per_row + kb, kqs);
        const Q4MoeFragment w1(
            weights, (expert * rows + min(row0 + 1, rows - 1)) * blocks_per_row + kb,
            kqs);
#pragma unroll
        for (int t = 0; t < columns; ++t) {
          if (slots[t] == 0)
            continue;
          const auto* x = input + token_row(t) * input_stride + kb * (qk / QK8_1);
          sum[t][0] += w0.Dot(x, kqs);
          sum[t][1] += w1.Dot(x, kqs);
        }
      } else {
#pragma unroll
        for (int t = 0; t < columns; ++t) {
          if (slots[t] == 0)
            continue;
#pragma unroll
          for (int r = 0; r < 2; ++r)
            sum[t][r] += dot(
                weights,
                input + token_row(single_request ? token : t) * input_stride +
                    kb * (qk / QK8_1),
                (expert * rows + min(row0 + r, rows - 1)) * blocks_per_row + kb,
                kqs);
        }
      }
    }

    // Gate and up stay in separate 32-lane halves to retain the scalar projection's
    // sum order, including the treatment of nonfinite quantization scales.
    __shared__ float values[columns][2][2];
#pragma unroll
    for (int t = 0; t < columns; ++t) {
      if (slots[t] == 0)
        continue;
#pragma unroll
      for (int r = 0; r < 2; ++r)
        sum[t][r] = warp_reduce_sum<32>(sum[t][r]);
      if (lane < 2)
        values[t][is_up][lane] = isfinite(sum[t][lane]) ? sum[t][lane] : 0.0f;
    }
    __syncthreads();
    if (!is_up && lane < 2 && row0 + lane < rows) {
#pragma unroll
      for (int t = 0; t < columns; ++t) {
        if (slots[t] == 0)
          continue;
        const float g = values[t][0][lane];
        const float u = values[t][1][lane];
        uint32_t bits = slots[t];
        while (bits) {
          const int slot = __ffs(static_cast<int>(bits)) - 1;
          output[(token_row(single_request ? token : t) * experts_used + slot) * rows +
                 row0 + lane] =
              (g * (1.0f / (1.0f + __expf(-g)))) * u;
          bits &= bits - 1;
        }
      }
    }
    __syncthreads();
  }
}

void mul_mat_vec_moe_batch_wave64(const void* gate, const void* up,
                                 const block_q8_1* input, const int32_t* groups,
                                 float* output, int k, int rows,
                                 int experts_used, int input_stride,
                                 hipStream_t stream) {
  const dim3 grid((rows + 1) / 2, 32);
  mul_mat_vec_moe_batch<false><<<grid, 64, 0, stream>>>(
      gate, up, input, groups, output, k, rows, experts_used, input_stride);
  mul_mat_vec_moe_batch<true><<<grid, 64, 0, stream>>>(
      gate, up, input, groups, output, k, rows, experts_used, input_stride);
}

template<bool single_request>
__launch_bounds__(64) static __global__ void mul_mat_vec_moe_batch_split(
    const void* __restrict__ gate, const void* __restrict__ up,
    const block_q8_1* __restrict__ input, const int32_t* __restrict__ storage,
    float* __restrict__ output, int k, int rows, int n_tokens, int experts_used,
    int input_stride) {
  constexpr int tokens = MMVQ_MAX_BATCH_SIZE;
  const int count = storage[single_request ? 1 : 0];
  if (blockIdx.y >= count)
    return;
  const auto* multi_groups = reinterpret_cast<const MoeBatchGroup*>(storage + 2);
  const auto* groups =
      single_request ? multi_groups + n_tokens * experts_used : multi_groups;
  constexpr int qk = QK_K;
  constexpr int qi = QI4_K;
  constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
  constexpr int blocks_per_iter = vdr * 32 / qi;
  constexpr auto dot = vec_dot_q4_K_q8_1;
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const bool is_up = tid >= 32;
  const int row0 = blockIdx.x * 2;
  const void* weights = is_up ? up : gate;
  const int blocks_per_row = k / qk;
  const int kqs = vdr * (lane % (qi / vdr));
  for (int anchor = blockIdx.y; anchor < count; anchor += gridDim.y) {
    const auto& group = groups[anchor];
    const int expert = group.expert;
    if constexpr (single_request) {
      const int token = group.token[0];
      const uint32_t mask = group.slots[0];
      float sum[2] = {};
      for (int kb = lane / (qi / vdr); kb < blocks_per_row;
           kb += blocks_per_iter) {
        const auto* x = input + token * input_stride + kb * (qk / QK8_1);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          sum[r] += dot(
              weights, x,
              (expert * rows + min(row0 + r, rows - 1)) * blocks_per_row + kb,
              kqs);
        }
      }
      __shared__ float values[2][2];
#pragma unroll
      for (int r = 0; r < 2; ++r)
        sum[r] = warp_reduce_sum<32>(sum[r]);
      if (lane < 2)
        values[is_up][lane] = isfinite(sum[lane]) ? sum[lane] : 0.0f;
      __syncthreads();
      if (!is_up && lane < 2 && row0 + lane < rows) {
        const float g = values[0][lane];
        const float u = values[1][lane];
        uint32_t bits = mask;
        while (bits) {
          const int slot = __ffs(static_cast<int>(bits)) - 1;
          output[(token * experts_used + slot) * rows + row0 + lane] =
              (g * (1.0f / (1.0f + __expf(-g)))) * u;
          bits &= bits - 1;
        }
      }
      __syncthreads();
    } else {
      if (expert < 0) {
        if (expert == -2 && tid < 2 && row0 + tid < rows)
          output[group.anchor * rows + row0 + tid] = 0.0f;
        continue;
      }
      uint32_t slots[tokens];
#pragma unroll
      for (int t = 0; t < tokens; ++t)
        slots[t] = group.slots[t];
      float sum[tokens][2] = {};
      for (int kb = lane / (qi / vdr); kb < blocks_per_row;
           kb += blocks_per_iter) {
        const Q4MoeFragment w0(
            weights, (expert * rows + row0) * blocks_per_row + kb, kqs);
        const Q4MoeFragment w1(
            weights, (expert * rows + min(row0 + 1, rows - 1)) * blocks_per_row + kb,
            kqs);
#pragma unroll
        for (int t = 0; t < tokens; ++t) {
          if (slots[t] == 0)
            continue;
          const auto* x = input + group.token[t] * input_stride + kb * (qk / QK8_1);
          sum[t][0] += w0.Dot(x, kqs);
          sum[t][1] += w1.Dot(x, kqs);
        }
      }
      __shared__ float values[tokens][2][2];
#pragma unroll
      for (int t = 0; t < tokens; ++t) {
        if (slots[t] == 0)
          continue;
#pragma unroll
        for (int r = 0; r < 2; ++r)
          sum[t][r] = warp_reduce_sum<32>(sum[t][r]);
        if (lane < 2)
          values[t][is_up][lane] = isfinite(sum[t][lane]) ? sum[t][lane] : 0.0f;
      }
      __syncthreads();
      if (!is_up && lane < 2 && row0 + lane < rows) {
#pragma unroll
        for (int t = 0; t < tokens; ++t) {
          if (slots[t] == 0)
            continue;
          const float g = values[t][0][lane];
          const float u = values[t][1][lane];
          uint32_t bits = slots[t];
          while (bits) {
            const int slot = __ffs(static_cast<int>(bits)) - 1;
            output[(group.token[t] * experts_used + slot) * rows + row0 + lane] =
                (g * (1.0f / (1.0f + __expf(-g)))) * u;
            bits &= bits - 1;
          }
        }
      }
      __syncthreads();
    }
  }
}

void mul_mat_vec_moe_batch_wave64_split_ref(const void* gate, const void* up,
                                           const block_q8_1* input,
                                           const int32_t* storage,
                                           float* output, int k, int rows,
                                           int tokens, int experts_used,
                                           int input_stride,
                                           hipStream_t stream) {
  const dim3 grid((rows + 1) / 2, 32);
  mul_mat_vec_moe_batch_split<false><<<grid, 64, 0, stream>>>(
      gate, up, input, storage, output, k, rows, tokens, experts_used,
      input_stride);
  mul_mat_vec_moe_batch_split<true><<<grid, 64, 0, stream>>>(
      gate, up, input, storage, output, k, rows, tokens, experts_used,
      input_stride);
}

void mul_mat_vec_moe_batch_wave64_split(const void* gate, const void* up,
                                       const block_q8_1* input,
                                       const int32_t* storage, float* output,
                                       int k, int rows, int tokens,
                                       int experts_used, int input_stride,
                                       hipStream_t stream) {
  if (FastMoeGuEnabled()) {
    mul_mat_vec_moe_batch_wave32_split(gate, up, input, storage, output, k,
                                       rows, tokens, experts_used, input_stride,
                                       stream);
    return;
  }
  mul_mat_vec_moe_batch_wave64_split_ref(gate, up, input, storage, output, k,
                                         rows, tokens, experts_used,
                                         input_stride, stream);
}

}  // namespace qfn_mmq
