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

}  // namespace

void LaunchWmmaAttention(const std::uint16_t* query, const std::uint16_t* key,
                         const std::uint16_t* value, std::uint16_t* output,
                         std::uint32_t rows, std::uint32_t heads,
                         hipStream_t stream) {
  WmmaAttention<<<dim3((rows + 63) / 64, heads), 128, 0, stream>>>(
      query, key, value, output, rows, heads);
}

}  // namespace gufo::minimax_h3::dit_ops
