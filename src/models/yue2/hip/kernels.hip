#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>

#include "src/models/yue2/hip/kernels.hpp"

namespace gufo::models::yue2::hip {
namespace {

constexpr unsigned int kHidden = 2048;
constexpr unsigned int kLatent = 64;

__device__ __forceinline__ float Round(float value) {
  return static_cast<float>(hip_bfloat16(value));
}

__device__ __forceinline__ float Load(const hip_bfloat16* values,
                                      std::size_t index) {
  return static_cast<float>(values[index]);
}

// One workgroup per row: the BF16 latent lives in LDS and each thread forms
// eight hidden channels from their 64-wide `vae2llm` rows.
__launch_bounds__(256) __global__
    void LatentInputKernel(const float* __restrict__ state,
                           std::uint32_t frames,
                           const hip_bfloat16* __restrict__ weight,
                           const hip_bfloat16* __restrict__ bias,
                           const hip_bfloat16* __restrict__ time_row,
                           const hip_bfloat16* __restrict__ positions,
                           float* __restrict__ hidden) {
  const std::uint32_t row = blockIdx.x;
  __shared__ float latent[kLatent];
  if (threadIdx.x < kLatent) {
    const bool content = row >= 1 && row <= frames;
    latent[threadIdx.x] =
        content ? Round(state[(row - 1) * kLatent + threadIdx.x]) : 0.0F;
  }
  __syncthreads();
  for (unsigned int c = threadIdx.x; c < kHidden; c += blockDim.x) {
    const hip_bfloat16* w = weight + static_cast<std::size_t>(c) * kLatent;
    float sum = 0.0F;
#pragma unroll 16
    for (unsigned int k = 0; k < kLatent; ++k)
      sum = fmaf(latent[k], Load(w, k), sum);
    float value = Round(sum + Load(bias, c));
    value = Round(value + Load(time_row, c));
    value = Round(value +
                  Load(positions, static_cast<std::size_t>(row) * kHidden + c));
    hidden[static_cast<std::size_t>(row) * kHidden + c] = value;
  }
}

__global__ void VelocityOutputKernel(const float* __restrict__ projected,
                                     const hip_bfloat16* __restrict__ bias,
                                     std::size_t count,
                                     float* __restrict__ velocity) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    velocity[index] =
        Round(projected[index + kLatent] + Load(bias, index % kLatent));
}

__global__ void OdeStepKernel(const float* __restrict__ state,
                              const float* __restrict__ velocity, float scale,
                              std::size_t count, float* __restrict__ output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    output[index] = Round(state[index] - Round(scale * velocity[index]));
}

__global__ void Bfloat16LogitsKernel(const float* __restrict__ input,
                                     std::size_t count,
                                     float* __restrict__ output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    output[index] = Round(input[index]);
}

__global__ void SnakeKernel(const float* __restrict__ input,
                            float* __restrict__ output, std::size_t pitch,
                            std::size_t length, std::size_t pad,
                            const float* __restrict__ alpha,
                            const float* __restrict__ inverse_beta) {
  const std::size_t channel = blockIdx.y;
  const std::size_t base = channel * pitch;
  const float a = alpha[channel];
  const float b = inverse_beta[channel];
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < pitch; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (i < pad || i >= pad + length) {
      output[base + i] = 0.0F;
      continue;
    }
    const float x = input[base + i];
    const float s = sinf(x * a);
    output[base + i] = x + b * (s * s);
  }
}

__global__ void FillBiasKernel(float* __restrict__ output, std::size_t pitch,
                               std::size_t length, std::size_t pad,
                               const float* __restrict__ bias) {
  const std::size_t channel = blockIdx.y;
  const float value = bias != nullptr ? bias[channel] : 0.0F;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < pitch; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    output[channel * pitch + i] = i < pad || i >= pad + length ? 0.0F : value;
}

__global__ void AddBiasKernel(float* __restrict__ values, std::size_t pitch,
                              std::size_t length, std::size_t pad,
                              const float* __restrict__ bias) {
  const std::size_t channel = blockIdx.y;
  const float value = bias[channel];
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < length; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    values[channel * pitch + pad + i] += value;
}

__global__ void InterleavePhasesKernel(const float* __restrict__ phases,
                                       std::size_t phase_length,
                                       std::size_t stride, std::size_t padding,
                                       std::size_t channels, std::size_t pitch,
                                       std::size_t length, std::size_t pad,
                                       const float* __restrict__ bias,
                                       float* __restrict__ output) {
  const std::size_t channel = blockIdx.y;
  const float value = bias[channel];
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < pitch; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (i < pad || i >= pad + length) {
      output[channel * pitch + i] = 0.0F;
      continue;
    }
    const std::size_t shifted = i - pad + padding;
    const std::size_t phase = shifted % stride;
    const std::size_t q = shifted / stride;
    output[channel * pitch + i] =
        phases[(phase * channels + channel) * phase_length + q] + value;
  }
}

__global__ void InterleaveStereoKernel(const float* __restrict__ input,
                                       std::size_t pitch, std::size_t pad,
                                       std::size_t first, std::size_t count,
                                       float* __restrict__ output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= 2 * count)
    return;
  const std::size_t frame = index / 2;
  const std::size_t channel = index % 2;
  output[index] = input[channel * pitch + pad + first + frame];
}

using Fragment = __attribute__((ext_vector_type(16))) short;
using Accumulator = __attribute__((ext_vector_type(8))) float;

__device__ __forceinline__ Fragment LoadFragment(const std::uint16_t* source) {
  union {
    Fragment fragment;
    uint4 words[2];
  } value;
  value.words[0] = *reinterpret_cast<const uint4*>(source);
  value.words[1] = *reinterpret_cast<const uint4*>(source + 8);
  return value.fragment;
}

__device__ __forceinline__ unsigned Bf16Bits(float value) {
  return static_cast<unsigned>(hip_bfloat16(value).data);
}

__device__ __forceinline__ unsigned SwapHalves(unsigned value) {
  return __builtin_amdgcn_permlanex16(value, value, 0x76543210U, 0xfedcba98U,
                                      false, false);
}

__launch_bounds__(256) __global__
    void TransposeValuesKernel(const std::uint16_t* __restrict__ input,
                               std::uint32_t stride, std::uint32_t count,
                               std::uint32_t pitch,
                               std::uint16_t* __restrict__ output) {
  __shared__ __align__(16) std::uint16_t tile[32][136];
  const unsigned tid = threadIdx.x;
  const std::uint32_t base = blockIdx.x * 32;
  input += static_cast<std::size_t>(blockIdx.y) * stride * 128;
  output += static_cast<std::size_t>(blockIdx.y) * 128 * pitch;
#pragma unroll
  for (unsigned z = 0; z < 2; ++z) {
    const unsigned r = tid / 16 + z * 16, d = (tid % 16) * 8;
    uint4 x{};
    if (base + r < count)
      x = *reinterpret_cast<const uint4*>(
          input + (static_cast<std::size_t>(base) + r) * 128 + d);
    *reinterpret_cast<uint4*>(&tile[r][d]) = x;
  }
  __syncthreads();
#pragma unroll
  for (unsigned z = 0; z < 2; ++z) {
    const unsigned d = tid / 4 + z * 64, r = (tid % 4) * 8;
    union {
      uint4 packed;
      std::uint16_t half[8];
    } value;
#pragma unroll
    for (unsigned i = 0; i < 8; ++i)
      value.half[i] = tile[r + i][d];
    *reinterpret_cast<uint4*>(output + static_cast<std::size_t>(d) * pitch +
                              base + r) = value.packed;
  }
}

constexpr unsigned kNarKeys = 64;

// Eight waves own sixteen queries each; Q stays in registers. Each 64-key
// tile of K and of transposed V is staged once in LDS for all waves, and the
// next tile is already loading into registers while this one computes.
// Scores are computed transposed (keys x queries), so each lane owns one
// query and its softmax needs only a cross-half exchange.
__launch_bounds__(256) __global__
    void NarAttentionKernel(const std::uint16_t* __restrict__ queries,
                            NarSegment prefix, NarSegment latent,
                            std::uint16_t* __restrict__ output,
                            std::uint32_t heads, std::uint32_t rows,
                            std::uint32_t group) {
  __shared__ __align__(16) std::uint16_t staged_key[kNarKeys][136];
  __shared__ __align__(16) std::uint16_t staged_value[128][kNarKeys + 8];
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const std::uint32_t head = blockIdx.y, kv_head = head / group;
  const std::uint32_t query_row = blockIdx.x * 128 + wave * 16 + lane % 16;
  queries += static_cast<std::size_t>(head) * rows * 128;
  Fragment q[8];
#pragma unroll
  for (int d = 0; d < 8; ++d) {
    q[d] = {};
    if (query_row < rows)
      q[d] = LoadFragment(queries + static_cast<std::size_t>(query_row) * 128 +
                          d * 16);
  }
  // Tile t < prefix_tiles is prefix tile t, else latent tile t - prefix_tiles.
  const std::uint32_t prefix_tiles = (prefix.count + kNarKeys - 1) / kNarKeys;
  const std::uint32_t tiles =
      prefix_tiles + (latent.count + kNarKeys - 1) / kNarKeys;
  const auto key_base = [&](const NarSegment& segment) {
    return segment.keys +
           static_cast<std::size_t>(kv_head) * segment.key_stride * 128;
  };
  const auto value_base = [&](const NarSegment& segment) {
    return segment.values +
           static_cast<std::size_t>(kv_head) * 128 * segment.value_pitch;
  };
  const std::uint16_t* prefix_keys = key_base(prefix);
  const std::uint16_t* latent_keys = key_base(latent);
  const std::uint16_t* prefix_values = value_base(prefix);
  const std::uint16_t* latent_values = value_base(latent);
  uint4 next_key[4], next_value[4];
  const auto fetch = [&](std::uint32_t tile) {
    const bool first = tile < prefix_tiles;
    const std::uint32_t base = (first ? tile : tile - prefix_tiles) * kNarKeys;
    const std::uint32_t count = first ? prefix.count : latent.count;
    const std::uint32_t pitch = first ? prefix.value_pitch : latent.value_pitch;
    const std::uint16_t* keys = first ? prefix_keys : latent_keys;
    const std::uint16_t* values = first ? prefix_values : latent_values;
#pragma unroll
    for (int z = 0; z < 4; ++z) {
      const int kr = tid / 16 + z * 16, d = (tid % 16) * 8;
      next_key[z] =
          base + kr < count
              ? *reinterpret_cast<const uint4*>(
                    keys + (static_cast<std::size_t>(base) + kr) * 128 + d)
              : uint4{};
      // Pitches are multiples of 32: the second half of a last tile past the
      // padding is never read.
      const int vd = tid / 8 + z * 32, vk = (tid % 8) * 8;
      next_value[z] =
          base + vk < pitch
              ? *reinterpret_cast<const uint4*>(
                    values + static_cast<std::size_t>(vd) * pitch + base + vk)
              : uint4{};
    }
  };
  Accumulator result[8]{};
  float maximum = -INFINITY, sum = 0.0F;
  if (tiles != 0)
    fetch(0);
#pragma unroll 1
  for (std::uint32_t tile = 0; tile < tiles; ++tile) {
    const bool first = tile < prefix_tiles;
    const std::uint32_t base = (first ? tile : tile - prefix_tiles) * kNarKeys;
    const std::uint32_t count = first ? prefix.count : latent.count;
#pragma unroll
    for (int z = 0; z < 4; ++z) {
      const int kr = tid / 16 + z * 16, d = (tid % 16) * 8;
      *reinterpret_cast<uint4*>(&staged_key[kr][d]) = next_key[z];
      const int vd = tid / 8 + z * 32, vk = (tid % 8) * 8;
      *reinterpret_cast<uint4*>(&staged_value[vd][vk]) = next_value[z];
    }
    __syncthreads();
    if (tile + 1 < tiles)
      fetch(tile + 1);
    Accumulator scores[4]{};
#pragma unroll
    for (int d = 0; d < 8; ++d) {
#pragma unroll
      for (int part = 0; part < 4; ++part)
        scores[part] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
            LoadFragment(&staged_key[part * 16 + lane % 16][d * 16]), q[d],
            scores[part]);
    }
    float next = maximum;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const std::uint32_t key = base + part * 16 + 2 * i + lane / 16;
        scores[part][i] =
            key < count ? scores[part][i] * 0.08838834764831845F : -INFINITY;
        next = fmaxf(next, scores[part][i]);
      }
    }
    next = fmaxf(next, __uint_as_float(SwapHalves(__float_as_uint(next))));
    const float correction = expf(maximum - next);
    maximum = next;
#pragma unroll
    for (int d = 0; d < 8; ++d)
      result[d] *= correction;
    float subtotal = 0.0F;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      Fragment probabilities;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float p = expf(scores[part][i] - next);
        subtotal += p;
        const unsigned own = Bf16Bits(p);
        const unsigned peer = SwapHalves(own);
        probabilities[2 * i] = static_cast<short>(lane < 16 ? own : peer);
        probabilities[2 * i + 1] = static_cast<short>(lane < 16 ? peer : own);
      }
#pragma unroll
      for (int d = 0; d < 8; ++d)
        result[d] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
            LoadFragment(&staged_value[d * 16 + lane % 16][part * 16]),
            probabilities, result[d]);
    }
    subtotal += __uint_as_float(SwapHalves(__float_as_uint(subtotal)));
    sum = sum * correction + subtotal;
    __syncthreads();
  }
  if (query_row >= rows)
    return;
#pragma unroll
  for (int d = 0; d < 8; ++d) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      output[(static_cast<std::size_t>(query_row) * heads + head) * 128 +
             d * 16 + 2 * i + lane / 16] =
          static_cast<std::uint16_t>(Bf16Bits(result[d][i] / sum));
  }
}

dim3 Rows(std::size_t pitch, std::size_t channels) {
  const std::size_t blocks = std::min<std::size_t>((pitch + 255) / 256, 1024);
  return dim3(static_cast<unsigned int>(blocks),
              static_cast<unsigned int>(channels));
}

dim3 Blocks(std::size_t count) {
  return dim3(static_cast<unsigned int>((count + 255) / 256));
}

}  // namespace

void LaunchSnake(const float* input, float* output, std::size_t channels,
                 std::size_t length, std::size_t pad, const float* alpha,
                 const float* inverse_beta, hipStream_t stream) {
  const std::size_t pitch = length + 2 * pad;
  SnakeKernel<<<Rows(pitch, channels), 256, 0, stream>>>(
      input, output, pitch, length, pad, alpha, inverse_beta);
}

void LaunchFillBias(float* output, std::size_t channels, std::size_t length,
                    std::size_t pad, const float* bias, hipStream_t stream) {
  const std::size_t pitch = length + 2 * pad;
  FillBiasKernel<<<Rows(pitch, channels), 256, 0, stream>>>(output, pitch,
                                                            length, pad, bias);
}

void LaunchAddBias(float* values, std::size_t channels, std::size_t length,
                   std::size_t pad, const float* bias, hipStream_t stream) {
  AddBiasKernel<<<Rows(length, channels), 256, 0, stream>>>(
      values, length + 2 * pad, length, pad, bias);
}

void LaunchInterleavePhases(const float* phases, std::size_t phase_length,
                            std::size_t stride, std::size_t padding,
                            std::size_t channels, std::size_t length,
                            std::size_t pad, const float* bias, float* output,
                            hipStream_t stream) {
  const std::size_t pitch = length + 2 * pad;
  InterleavePhasesKernel<<<Rows(pitch, channels), 256, 0, stream>>>(
      phases, phase_length, stride, padding, channels, pitch, length, pad, bias,
      output);
}

void LaunchInterleaveStereo(const float* input, std::size_t length,
                            std::size_t pad, std::size_t first,
                            std::size_t count, float* output,
                            hipStream_t stream) {
  InterleaveStereoKernel<<<Blocks(2 * count), 256, 0, stream>>>(
      input, length + 2 * pad, pad, first, count, output);
}

void LaunchLatentInput(const float* state, std::size_t frames,
                       const void* weight, const void* bias,
                       const void* time_row, const void* positions,
                       float* hidden, hipStream_t stream) {
  LatentInputKernel<<<static_cast<unsigned int>(frames + 2), 256, 0, stream>>>(
      state, static_cast<std::uint32_t>(frames),
      static_cast<const hip_bfloat16*>(weight),
      static_cast<const hip_bfloat16*>(bias),
      static_cast<const hip_bfloat16*>(time_row),
      static_cast<const hip_bfloat16*>(positions), hidden);
}

void LaunchVelocityOutput(const float* projected, const void* bias,
                          std::size_t frames, float* velocity,
                          hipStream_t stream) {
  const std::size_t count = frames * kLatent;
  VelocityOutputKernel<<<Blocks(count), 256, 0, stream>>>(
      projected, static_cast<const hip_bfloat16*>(bias), count, velocity);
}

void LaunchOdeStep(const float* state, const float* velocity, float scale,
                   std::size_t count, float* output, hipStream_t stream) {
  OdeStepKernel<<<Blocks(count), 256, 0, stream>>>(state, velocity, scale,
                                                   count, output);
}

void LaunchBfloat16Logits(const float* input, std::size_t count, float* output,
                          hipStream_t stream) {
  Bfloat16LogitsKernel<<<Blocks(count), 256, 0, stream>>>(input, count, output);
}

void LaunchTransposeValues(const std::uint16_t* values, std::uint32_t stride,
                           std::uint32_t count, std::uint32_t kv_heads,
                           std::uint16_t* output, hipStream_t stream) {
  const std::uint32_t pitch = NarValuePitch(count);
  TransposeValuesKernel<<<dim3(pitch / 32, kv_heads), 256, 0, stream>>>(
      values, stride, count, pitch, output);
}

void LaunchNarAttention(const std::uint16_t* queries, NarSegment prefix,
                        NarSegment latent, std::uint16_t* output,
                        std::uint32_t heads, std::uint32_t rows,
                        std::uint32_t group, hipStream_t stream) {
  NarAttentionKernel<<<dim3((rows + 127) / 128, heads), 256, 0, stream>>>(
      queries, prefix, latent, output, heads, rows, group);
}

}  // namespace gufo::models::yue2::hip
