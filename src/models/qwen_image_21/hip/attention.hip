#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "src/models/qwen_image_21/hip/bf16.hpp"
#include "src/models/qwen_image_21/hip/runtime.hpp"

namespace gufo::models::qwen_image_21::hip {
namespace {

using Fragment = __attribute__((ext_vector_type(16))) unsigned short;
using Accumulator = __attribute__((ext_vector_type(8))) float;

__device__ Fragment Load(const std::uint16_t* x) {
  union {
    Fragment b;
    uint4 u[2];
  } value;
  value.u[0] = *reinterpret_cast<const uint4*>(x);
  value.u[1] = *reinterpret_cast<const uint4*>(x + 8);
  return value.b;
}

// Online softmax with BF16 QK matrix instructions and an FP32 value product.
// No probability narrowing and no global score/probability allocation.
// Each block owns 128 queries by all 128 output channels, consuming 32 keys at
// a time. Split value staging into two halves to bound LDS usage.
template<bool Masked>
__global__ void FusedAttention(const std::uint16_t* queries,
                               const std::uint16_t* keys,
                               const std::uint16_t* values,
                               hip_bfloat16* output, int heads, int rows,
                               int count, int query_start, int query_count,
                               const int* limits) {
  __shared__ float p[32][129], v[16][129], stats[128][3];
  __shared__ int last_key;
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int col = blockIdx.x * 128, head = blockIdx.y;
  const int query_end = query_start + query_count;
  queries += static_cast<std::size_t>(head) * rows * 128;
  keys += static_cast<std::size_t>(head) * count * 128;
  values += static_cast<std::size_t>(head) * count * 128;
  if (tid < 128) {
    stats[tid][0] = -INFINITY;
    stats[tid][1] = 0;
  }
  if constexpr (Masked) {
    if (tid < 32) {
      int visible = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int q = query_start + col + tid + i * 32;
        if (q < query_end)
          visible = max(visible, limits[q]);
      }
#pragma unroll
      for (int offset = 16; offset; offset /= 2)
        visible = max(visible, __shfl_xor(visible, offset, 32));
      if (tid == 0)
        last_key = visible;
    }
  }
  __syncthreads();
  float accum[8][8]{};
  const int visible = Masked ? last_key : count;
  // Keep the next tile in registers while QK, softmax and FP32 PV execute.
  unsigned short prefetch_v[16];
#pragma unroll
  for (int z = 0; z < 16; ++z) {
    const int i = tid + z * 256;
    prefetch_v[z] =
        i / 128 < count
            ? values[static_cast<std::size_t>(i / 128) * 128 + i % 128]
            : 0;
  }
  for (int base = 0; base < visible; base += 32) {
    for (int qoff = 0; qoff < 128; qoff += 64) {
      Accumulator dot{};
      const int qr = query_start + col + qoff + (wave / 2) * 16 + lane % 16;
      const int kr = base + (wave % 2) * 16 + lane % 16;
// Keep two fragment pairs live; full unrolling raises register pressure.
#pragma unroll 2
      for (int d = 0; d < 128; d += 16) {
        Fragment q{};
        if (qr < query_end)
          q = Load(queries + static_cast<std::size_t>(qr) * 128 + d);
        Fragment k{};
        if (kr < count)
          k = Load(keys + static_cast<std::size_t>(kr) * 128 + d);
        dot = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(q, k, dot);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int q = qoff + (wave / 2) * 16 + 2 * i + lane / 16;
        const int k = (wave % 2) * 16 + lane % 16;
        int limit = count;
        if constexpr (Masked) {
          const int position = query_start + col + q;
          if (position < query_end)
            limit = limits[position];
        }
        p[k][q] = base + k < limit ? dot[i] * 0.08838834764831845F : -INFINITY;
      }
    }
    __syncthreads();
    for (int qoff = 0; qoff < 128; qoff += 64) {
      const int q = qoff + tid / 4, l = tid % 4;
      float maximum = -INFINITY;
#pragma unroll
      for (int k = l; k < 32; k += 4)
        maximum = fmaxf(maximum, p[k][q]);
      maximum = fmaxf(maximum, __shfl_xor(maximum, 2, 4));
      maximum = fmaxf(maximum, __shfl_xor(maximum, 1, 4));
      const float previous = stats[q][0];
      maximum = fmaxf(maximum, previous);
      const float correction = expf(previous - maximum);
      float sum = 0;
#pragma unroll
      for (int k = l; k < 32; k += 4) {
        const float value = expf(p[k][q] - maximum);
        p[k][q] = value;
        sum += value;
      }
      sum += __shfl_xor(sum, 2, 4);
      sum += __shfl_xor(sum, 1, 4);
      if (l == 0) {
        stats[q][0] = maximum;
        stats[q][1] = stats[q][1] * correction + sum;
        stats[q][2] = correction;
      }
    }
    unsigned short current_v[16];
#pragma unroll
    for (int z = 0; z < 16; z++)
      current_v[z] = prefetch_v[z];
#pragma unroll
    for (int z = 0; z < 16; ++z) {
      const int i = tid + z * 256;
      prefetch_v[z] =
          base + 32 + i / 128 < count
              ? values[std::size_t(base + 32 + i / 128) * 128 + i % 128]
              : 0;
    }
    __syncthreads();
#pragma unroll
    for (int y = 0; y < 8; y++)
#pragma unroll
      for (int x = 0; x < 8; x++)
        accum[y][x] *= stats[tid / 16 + y * 16][2];
    for (int half = 0; half < 2; half++) {
#pragma unroll
      for (int z = 0; z < 8; z++) {
        const int i = tid + z * 256;
        v[i / 128][i % 128] =
            __uint_as_float(unsigned(current_v[half * 8 + z]) << 16);
      }
      __syncthreads();
#pragma unroll 2
      for (int k = 0; k < 16; k++)
#pragma unroll
        for (int y = 0; y < 8; y++) {
          float probability = p[half * 16 + k][tid / 16 + y * 16];
#pragma unroll
          for (int x = 0; x < 8; x++)
            accum[y][x] =
                fmaf(v[k][tid % 16 + x * 16], probability, accum[y][x]);
        }
      __syncthreads();
    }
    __syncthreads();
  }
#pragma unroll
  for (int y = 0; y < 8; ++y)
#pragma unroll
    for (int x = 0; x < 8; ++x) {
      const int query = query_start + col + tid / 16 + y * 16;
      if (query < query_end)
        output[(static_cast<std::size_t>(query) * heads + head) * 128 +
               tid % 16 + x * 16] =
            Bf16(accum[y][x] / stats[tid / 16 + y * 16][1]);
    }
}
}  // namespace

void FusedAttention128(const std::uint16_t* queries, const std::uint16_t* keys,
                       const std::uint16_t* values, std::uint16_t* output,
                       int heads, int rows, int count, int query_start,
                       hipStream_t stream, int query_count, const int* limits) {
  const dim3 grid((query_count + 127) / 128, heads);
  if (limits)
    FusedAttention<true><<<grid, 256, 0, stream>>>(
        queries, keys, values, reinterpret_cast<hip_bfloat16*>(output), heads,
        rows, count, query_start, query_count, limits);
  else
    FusedAttention<false><<<grid, 256, 0, stream>>>(
        queries, keys, values, reinterpret_cast<hip_bfloat16*>(output), heads,
        rows, count, query_start, query_count, nullptr);
}

}  // namespace gufo::models::qwen_image_21::hip
