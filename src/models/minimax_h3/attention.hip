// Short-path softmax ordering adapted from Composable Kernel, ROCm 7.2.3.
// Copyright (c) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
// MIT; see licenses/composable_kernel.txt. No CK headers/runtime are required.
#include <cstddef>

#include "src/models/minimax_h3/attention.hip.hpp"

namespace gufo::minimax_h3::dit_ops {
using BufferResource = __attribute__((ext_vector_type(4))) unsigned;
__device__ BufferResource
LoadBuffer(BufferResource, unsigned, unsigned,
           unsigned) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

namespace {

using Fragment = __attribute__((ext_vector_type(16))) unsigned short;
using Accumulator = __attribute__((ext_vector_type(8))) float;

__device__ BufferResource MakeBuffer(const std::uint16_t* pointer,
                                     unsigned bytes) {
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  return {unsigned(address), unsigned(address >> 32), bytes, 0x31004000U};
}
__device__ uint4 LoadBufferVector(BufferResource resource, unsigned offset) {
  union {
    BufferResource vector;
    uint4 packed;
  } result;
  result.vector = LoadBuffer(resource, offset, 0, 0);
  return result.packed;
}

__device__ Fragment LoadFragment(const std::uint16_t* source) {
  union {
    Fragment fragment;
    uint4 words[2];
  } value;
  value.words[0] = *reinterpret_cast<const uint4*>(source);
  value.words[1] = *reinterpret_cast<const uint4*>(source + 8);
  return value.fragment;
}

__device__ std::uint16_t Bf16(float value) {
  const unsigned bits = __float_as_uint(value);
  const unsigned nan = 0U - unsigned((bits & 0x7fffffffU) > 0x7f800000U);
  const unsigned rounded = bits + 0x7fffU + ((bits >> 16U) & 1U);
  return ((rounded & ~nan) | ((bits | 0x00400000U) & nan)) >> 16U;
}

// Four waves own sixteen queries each. Transposed QK/PV tiles keep each
// query's softmax reduction within two lanes. Q/K/V and PV inputs use BF16;
// scores, normalization and value accumulation use FP32. No quadratic scratch
// or global probability/LSE buffer. Qualified against the independent teacher.
__global__ __launch_bounds__(128, 1) void WmmaAttention(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value, std::uint16_t* output, std::uint32_t rows,
    std::uint32_t heads) {
  __shared__ std::uint16_t staged_key[32][136];
  __shared__ std::uint16_t staged_value[32][136];
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int first_query = blockIdx.x * 64 + wave * 16;
  const std::size_t head_base = std::size_t(blockIdx.y) * rows * 128;
  query += head_base;
  key += head_base;
  value += head_base;
  // Bounded per-head descriptors zero masked key tails without reading the
  // next head. The admitted <=131072 rows fit 32-bit byte offsets.
  const auto key_buffer = MakeBuffer(key, rows * 256);
  const auto value_buffer = MakeBuffer(value, rows * 256);
  Fragment q[8];
#pragma unroll
  for (int d = 0; d < 8; ++d) {
    q[d] = {};
    if (static_cast<std::uint32_t>(first_query + lane % 16) < rows)
      q[d] = LoadFragment(query + std::size_t(first_query + lane % 16) * 128 +
                          d * 16);
  }
  Accumulator result[8]{};
  float maximum = -INFINITY, sum = 0;

  for (std::uint32_t base = 0; base < rows; base += 32) {
    // Share each global K/V vector across all four query waves.
#pragma unroll
    for (int z = 0; z < 4; ++z) {
      const int kr = tid / 16 + z * 8, d = (tid % 16) * 8;
      const unsigned offset = ((base + kr) * 128 + d) * 2;
      const uint4 k = LoadBufferVector(key_buffer, offset);
      const uint4 v = LoadBufferVector(value_buffer, offset);
      *reinterpret_cast<uint4*>(staged_key[kr] + d) = k;
      *reinterpret_cast<uint4*>(staged_value[kr] + d) = v;
    }
    __syncthreads();
    Accumulator scores[2]{};
#pragma unroll
    for (int d = 0; d < 8; ++d) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const Fragment k =
            LoadFragment(staged_key[half * 16 + lane % 16] + d * 16);
        scores[half] =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(k, q[d], scores[half]);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    if (base + 32 > rows) {
#pragma unroll
      for (int half = 0; half < 2; ++half)
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if (base + half * 16 + 2 * i + lane / 16 >= rows)
            scores[half][i] = -INFINITY;
    }
    float next = maximum;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        // log2(e) / sqrt(128), for the base-two online softmax.
        scores[half][i] *= 0.12751743082459868F;
        next = fmaxf(next, scores[half][i]);
      }
    }
    next = fmaxf(next, __uint_as_float(__builtin_amdgcn_permlanex16(
                           __float_as_uint(next), __float_as_uint(next),
                           0x76543210U, 0xfedcba98U, false, false)));
    const float correction = __builtin_amdgcn_exp2f(maximum - next);
    float subtotal = 0;
    Fragment probability_fragments[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float p = __builtin_amdgcn_exp2f(scores[half][i] - next);
        subtotal += p;
        const unsigned own = Bf16(p);
        const unsigned peer = __builtin_amdgcn_permlanex16(
            own, own, 0x76543210U, 0xfedcba98U, false, false);
        probability_fragments[half][2 * i] = lane < 16 ? own : peer;
        probability_fragments[half][2 * i + 1] = lane < 16 ? peer : own;
      }
    }
    subtotal += __uint_as_float(__builtin_amdgcn_permlanex16(
        __float_as_uint(subtotal), __float_as_uint(subtotal), 0x76543210U,
        0xfedcba98U, false, false));
    sum = sum * correction + subtotal;
    maximum = next;
#pragma unroll
    for (int d = 0; d < 8; ++d)
      result[d] *= correction;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const Fragment p = probability_fragments[half];
#pragma unroll
      for (int d = 0; d < 8; ++d) {
        Fragment v;
#pragma unroll
        for (int i = 0; i < 16; ++i)
          v[i] = staged_value[half * 16 + i][d * 16 + lane % 16];
        result[d] =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(v, p, result[d]);
        if (d % 2 == 1)
          __builtin_amdgcn_sched_barrier(0);
      }
    }
    __syncthreads();
  }
  const std::uint32_t row = first_query + lane % 16;
  if (row < rows) {
    const float reciprocal = __builtin_amdgcn_rcpf(sum);
#pragma unroll
    for (int d = 0; d < 8; ++d) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        output[(std::size_t(row) * heads + blockIdx.y) * 128 + d * 16 + 2 * i +
               lane / 16] = Bf16(result[d][i] * reciprocal);
    }
  }
}

// Short sequences retain 128-key local softmax tiles and normalize after each
// tile. In particular, round local (not running-max-scaled) probabilities to
// BF16 before PV. Contiguous eight-key lane groups preserve the sum tree.
// Explicit FMAs below preserve the rounded recurrence across all 50 blocks.
__global__ __launch_bounds__(128) void ShortAttention(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value, std::uint16_t* output, std::uint32_t rows,
    std::uint32_t heads) {
#pragma clang fp contract(off)
  union Staging {
    std::uint16_t key[128][136];
    struct {
      std::uint16_t probability[64][136];
      std::uint16_t value[128][136];
    } pv;
  };
  __shared__ __align__(16) Staging staging;
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int swizzled_lane = ((lane & 1) << 3) | ((lane % 16) >> 1);
  const int first_query = blockIdx.x * 64 + wave * 16;
  const std::size_t head_base = std::size_t(blockIdx.y) * rows * 128;
  query += head_base;
  const auto key_buffer = MakeBuffer(key + head_base, rows * 256);
  const auto value_buffer = MakeBuffer(value + head_base, rows * 256);
  Accumulator result[8]{};
  float running_sum = 0, running_max = -INFINITY;
  for (std::uint32_t base = 0; base < rows; base += 128) {
#pragma unroll
    for (int z = 0; z < 16; ++z) {
      const int r = tid / 16 + z * 8, d = (tid % 16) * 8;
      *reinterpret_cast<uint4*>(staging.key[r] + d) =
          LoadBufferVector(key_buffer, ((base + r) * 128 + d) * 2);
    }
    __syncthreads();
    Accumulator scores[8]{};
#pragma unroll 1
    for (int d = 0; d < 8; ++d) {
      Fragment q{};
      if (static_cast<std::uint32_t>(first_query + lane % 16) < rows)
        q = LoadFragment(query + std::size_t(first_query + lane % 16) * 128 +
                         d * 16);
#pragma unroll
      for (int half = 0; half < 8; ++half) {
        const auto k =
            LoadFragment(staging.key[half * 16 + swizzled_lane] + d * 16);
        scores[half] =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(k, q, scores[half]);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    float local_max = -INFINITY, local_sum = 0;
#pragma unroll
    for (int half = 0; half < 8; ++half)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        scores[half][i] = base + half * 16 + i + (lane / 16) * 8 < rows
                              ? scores[half][i] * 0.08838834764831845F
                              : -INFINITY;
        local_max = fmaxf(local_max, scores[half][i]);
      }
    local_max = fmaxf(local_max, __shfl_xor(local_max, 16));
    __syncthreads();
#pragma unroll
    for (int half = 0; half < 8; ++half)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float p = expf(scores[half][i] - local_max);
        local_sum += p;
        staging.pv.probability[wave * 16 + lane % 16]
                              [half * 16 + i + (lane / 16) * 8] = Bf16(p);
      }
    local_sum += __shfl_xor(local_sum, 16);
#pragma unroll
    for (int z = 0; z < 16; ++z) {
      const int r = tid / 16 + z * 8, d = (tid % 16) * 8;
      *reinterpret_cast<uint4*>(staging.pv.value[r] + d) =
          LoadBufferVector(value_buffer, ((base + r) * 128 + d) * 2);
    }
    __syncthreads();
    const float next_max = fmaxf(running_max, local_max);
    const float previous_scale = running_sum * expf(running_max - next_max);
    const float local_scale = expf(local_max - next_max);
    const float next_sum = fmaf(local_scale, local_sum, previous_scale);
#pragma unroll
    for (int d = 0; d < 8; ++d) {
      Accumulator partial{};
#pragma unroll 1
      for (int half = 0; half < 8; ++half) {
        const auto p = LoadFragment(
            staging.pv.probability[wave * 16 + lane % 16] + half * 16);
        Fragment v;
#pragma unroll
        for (int i = 0; i < 16; ++i)
          v[i] = staging.pv.value[half * 16 + i][d * 16 + swizzled_lane];
        partial = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(v, p, partial);
        __builtin_amdgcn_sched_barrier(0);
      }
      for (int i = 0; i < 8; ++i)
        result[d][i] =
            fmaf(previous_scale, result[d][i], local_scale * partial[i]) /
            next_sum;
    }
    running_sum = next_sum;
    running_max = next_max;
    __syncthreads();
  }
  const std::uint32_t row = first_query + lane % 16;
  if (row < rows)
#pragma unroll
    for (int d = 0; d < 8; ++d)
#pragma unroll
      for (int i = 0; i < 8; ++i)
        output[(std::size_t(row) * heads + blockIdx.y) * 128 + d * 16 + i +
               (lane / 16) * 8] = Bf16(result[d][i]);
}

}  // namespace

void LaunchWmmaAttention(const std::uint16_t* query, const std::uint16_t* key,
                         const std::uint16_t* value, std::uint16_t* output,
                         std::uint32_t rows, std::uint32_t heads,
                         hipStream_t stream) {
  if (rows <= 4096) {
    ShortAttention<<<dim3((rows + 63) / 64, heads), 128, 0, stream>>>(
        query, key, value, output, rows, heads);
    return;
  }
  WmmaAttention<<<dim3((rows + 63) / 64, heads), 128, 0, stream>>>(
      query, key, value, output, rows, heads);
}

}  // namespace gufo::minimax_h3::dit_ops
