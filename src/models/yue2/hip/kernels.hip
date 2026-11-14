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
    output[index] = fmaf(-scale, velocity[index], state[index]);
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

}  // namespace gufo::models::yue2::hip
