#ifndef GUFO_MODELS_MINIMAX_H3_DIT_OPS_HIP_HPP_
#define GUFO_MODELS_MINIMAX_H3_DIT_OPS_HIP_HPP_

// MiniMax H3 DiT operation boundaries translated from antirez/h3.c
// h3_shaders.metal at 8974cc055ea9c02fcd14cc27dfda3e1027c05153 (MIT).
// This is a model-private HIP implementation for gfx1151. It intentionally
// keeps BF16 write boundaries and F32 reductions visible.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::minimax_h3::dit_ops {

__device__ __forceinline__ float Bf16ToFloat(std::uint16_t value) {
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t FloatToBf16(float value) {
  std::uint32_t bits = __float_as_uint(value);
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

static __global__ void F32ToBf16Kernel(const float* input,
                                       std::uint16_t* output,
                                       std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = FloatToBf16(input[index]);
  }
}

static __global__ void Bf16ToF32Kernel(const std::uint16_t* input,
                                       float* output, std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = Bf16ToFloat(input[index]);
  }
}

inline void LaunchF32ToBf16(const float* input, std::uint16_t* output,
                            std::size_t elements, hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(F32ToBf16Kernel,
                     dim3((elements + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, input, output, elements);
}

inline void LaunchBf16ToF32(const std::uint16_t* input, float* output,
                            std::size_t elements, hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(Bf16ToF32Kernel,
                     dim3((elements + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, input, output, elements);
}

static __global__ void AddBiasF32Kernel(float* values, const float* bias,
                                        std::uint32_t rows,
                                        std::uint32_t width) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row < rows && column < width) {
    values[static_cast<std::size_t>(row) * width + column] += bias[column];
  }
}

inline void LaunchAddBiasF32(float* values, const float* bias,
                             std::uint32_t rows, std::uint32_t width,
                             hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  hipLaunchKernelGGL(AddBiasF32Kernel,
                     dim3((width + kThreads - 1U) / kThreads, rows),
                     dim3(kThreads), 0, stream, values, bias, rows, width);
}

static __global__ void AddBiasF32ToBf16Kernel(const float* input,
                                              const float* bias,
                                              std::uint16_t* output,
                                              std::uint32_t rows,
                                              std::uint32_t width) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row < rows && column < width) {
    const std::size_t index = static_cast<std::size_t>(row) * width + column;
    output[index] = FloatToBf16(input[index] + bias[column]);
  }
}

inline void LaunchAddBiasF32ToBf16(const float* input, const float* bias,
                                   std::uint16_t* output, std::uint32_t rows,
                                   std::uint32_t width, hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  hipLaunchKernelGGL(
      AddBiasF32ToBf16Kernel, dim3((width + kThreads - 1U) / kThreads, rows),
      dim3(kThreads), 0, stream, input, bias, output, rows, width);
}

static __global__ void FirstMismatchKernel(const std::uint16_t* left,
                                           const std::uint16_t* right,
                                           std::size_t elements,
                                           unsigned long long* first) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements && left[index] != right[index]) {
    atomicMin(first, static_cast<unsigned long long>(index));
  }
}

inline void LaunchFirstMismatch(const std::uint16_t* left,
                                const std::uint16_t* right,
                                std::size_t elements, unsigned long long* first,
                                hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(FirstMismatchKernel,
                     dim3((elements + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, left, right, elements, first);
}

static __global__ void FillBf16PatternKernel(std::uint16_t* output,
                                             std::size_t elements,
                                             std::uint32_t seed) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  std::uint32_t value = static_cast<std::uint32_t>(index) ^ seed;
  value ^= value >> 16U;
  value *= 0x7FEB352DU;
  value ^= value >> 15U;
  value *= 0x846CA68BU;
  value ^= value >> 16U;
  const float scaled =
      (static_cast<float>(value & 0xFFFFU) / 32767.5F - 1.0F) * 0.75F;
  output[index] = FloatToBf16(scaled);
}

inline void LaunchFillBf16Pattern(std::uint16_t* output, std::size_t elements,
                                  std::uint32_t seed, hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(FillBf16PatternKernel,
                     dim3((elements + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, output, elements, seed);
}

static __global__ void AddSubtractKernel(const std::uint16_t* left,
                                         const std::uint16_t* right,
                                         std::uint16_t* output,
                                         std::size_t elements, bool subtract) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float right_value = Bf16ToFloat(right[index]);
    output[index] = FloatToBf16(Bf16ToFloat(left[index]) +
                                (subtract ? -right_value : right_value));
  }
}

inline void LaunchAddSubtract(const std::uint16_t* left,
                              const std::uint16_t* right, std::uint16_t* output,
                              std::size_t elements, bool subtract,
                              hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(
      AddSubtractKernel, dim3((elements + kThreads - 1U) / kThreads),
      dim3(kThreads), 0, stream, left, right, output, elements, subtract);
}

static __global__ void SiluKernel(const std::uint16_t* input,
                                  std::uint16_t* output, std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float value = Bf16ToFloat(input[index]);
    output[index] = FloatToBf16(value / (1.0F + expf(-value)));
  }
}

inline void LaunchSilu(const std::uint16_t* input, std::uint16_t* output,
                       std::size_t elements, hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(SiluKernel, dim3((elements + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, input, output, elements);
}

static __global__ void SiluF32Kernel(const float* input, float* output,
                                     std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float value = input[index];
    output[index] = value / (1.0F + expf(-value));
  }
}

inline void LaunchSiluF32(const float* input, float* output,
                          std::size_t elements, hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(SiluF32Kernel, dim3((elements + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, input, output, elements);
}

static __global__ void AddBiasBf16Kernel(std::uint16_t* values,
                                         const std::uint16_t* bias,
                                         std::uint32_t rows,
                                         std::uint32_t width) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row < rows && column < width) {
    const std::size_t index = static_cast<std::size_t>(row) * width + column;
    values[index] =
        FloatToBf16(Bf16ToFloat(values[index]) + Bf16ToFloat(bias[column]));
  }
}

inline void LaunchAddBiasBf16(std::uint16_t* values, const std::uint16_t* bias,
                              std::uint32_t rows, std::uint32_t width,
                              hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  hipLaunchKernelGGL(AddBiasBf16Kernel,
                     dim3((width + kThreads - 1U) / kThreads, rows),
                     dim3(kThreads), 0, stream, values, bias, rows, width);
}

static __global__ void F32ProjectionKernel(
    const float* input, const float* weight, const float* bias, float* output,
    std::uint32_t rows, std::uint32_t input_width, std::uint32_t output_width) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || column >= output_width) {
    return;
  }
  float sum = bias == nullptr ? 0.0F : bias[column];
  const std::size_t input_base = static_cast<std::size_t>(row) * input_width;
  const std::size_t weight_base =
      static_cast<std::size_t>(column) * input_width;
  for (std::uint32_t index = 0; index < input_width; ++index) {
    sum = fmaf(input[input_base + index], weight[weight_base + index], sum);
  }
  output[static_cast<std::size_t>(row) * output_width + column] = sum;
}

inline void LaunchF32Projection(const float* input, const float* weight,
                                const float* bias, float* output,
                                std::uint32_t rows, std::uint32_t input_width,
                                std::uint32_t output_width,
                                hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  hipLaunchKernelGGL(F32ProjectionKernel,
                     dim3((output_width + kThreads - 1U) / kThreads, rows),
                     dim3(kThreads), 0, stream, input, weight, bias, output,
                     rows, input_width, output_width);
}

static __global__ void PatchProjectionKernel(
    const float* input, const float* weight, const float* bias,
    std::uint16_t* output, std::uint32_t rows, std::uint32_t input_width,
    std::uint32_t output_width) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || column >= output_width) {
    return;
  }
  float sum = bias == nullptr ? 0.0F : bias[column];
  const std::size_t input_base = static_cast<std::size_t>(row) * input_width;
  const std::size_t weight_base =
      static_cast<std::size_t>(column) * input_width;
  for (std::uint32_t index = 0; index < input_width; ++index) {
    sum = fmaf(input[input_base + index], weight[weight_base + index], sum);
  }
  output[static_cast<std::size_t>(row) * output_width + column] =
      FloatToBf16(sum);
}

inline void LaunchPatchProjection(const float* input, const float* weight,
                                  const float* bias, std::uint16_t* output,
                                  std::uint32_t rows, std::uint32_t input_width,
                                  std::uint32_t output_width,
                                  hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  hipLaunchKernelGGL(PatchProjectionKernel,
                     dim3((output_width + kThreads - 1U) / kThreads, rows),
                     dim3(kThreads), 0, stream, input, weight, bias, output,
                     rows, input_width, output_width);
}

static __global__ void FinalProjectionKernel(const std::uint16_t* input,
                                             const float* weight,
                                             const float* bias, float* output,
                                             std::uint32_t rows,
                                             std::uint32_t input_width,
                                             std::uint32_t output_width) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || column >= output_width) {
    return;
  }
  float sum = bias == nullptr ? 0.0F : bias[column];
  const std::size_t input_base = static_cast<std::size_t>(row) * input_width;
  const std::size_t weight_base =
      static_cast<std::size_t>(column) * input_width;
  for (std::uint32_t index = 0; index < input_width; ++index) {
    sum = fmaf(Bf16ToFloat(input[input_base + index]),
               weight[weight_base + index], sum);
  }
  output[static_cast<std::size_t>(row) * output_width + column] = sum;
}

inline void LaunchFinalProjection(const std::uint16_t* input,
                                  const float* weight, const float* bias,
                                  float* output, std::uint32_t rows,
                                  std::uint32_t input_width,
                                  std::uint32_t output_width,
                                  hipStream_t stream) {
  constexpr std::uint32_t kThreads = 128;
  hipLaunchKernelGGL(FinalProjectionKernel,
                     dim3((output_width + kThreads - 1U) / kThreads, rows),
                     dim3(kThreads), 0, stream, input, weight, bias, output,
                     rows, input_width, output_width);
}

static __global__ void SwiGluKernel(const std::uint16_t* __restrict__ fused,
                                    std::uint16_t* __restrict__ output,
                                    std::uint32_t rows, std::uint32_t width) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || column >= width) {
    return;
  }
  const std::size_t base = static_cast<std::size_t>(row) * width * 2U;
  const float gate = Bf16ToFloat(fused[base + column]);
  const float up = Bf16ToFloat(fused[base + width + column]);
  output[static_cast<std::size_t>(row) * width + column] =
      FloatToBf16(gate / (1.0F + expf(-gate)) * up);
}

static __global__ void SwiGluVector8Kernel(
    const std::uint16_t* __restrict__ fused, std::uint16_t* __restrict__ output,
    std::uint32_t rows, std::uint32_t width) {
  constexpr std::uint32_t kValuesPerThread = 8;
  const std::uint32_t vector_column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t column = vector_column * kValuesPerThread;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || column >= width) {
    return;
  }
  const std::size_t gate_base =
      static_cast<std::size_t>(row) * width * 2U + column;
  const std::size_t up_base = gate_base + width;
  const std::size_t output_base =
      static_cast<std::size_t>(row) * width + column;
#pragma unroll 1
  for (std::uint32_t offset = 0; offset < kValuesPerThread; offset += 2U) {
    const std::uint32_t gate_pair =
        *reinterpret_cast<const std::uint32_t*>(fused + gate_base + offset);
    const std::uint32_t up_pair =
        *reinterpret_cast<const std::uint32_t*>(fused + up_base + offset);
    const float gate0 =
        Bf16ToFloat(static_cast<std::uint16_t>(gate_pair & 0xFFFFU));
    const float gate1 =
        Bf16ToFloat(static_cast<std::uint16_t>(gate_pair >> 16U));
    const float up0 =
        Bf16ToFloat(static_cast<std::uint16_t>(up_pair & 0xFFFFU));
    const float up1 = Bf16ToFloat(static_cast<std::uint16_t>(up_pair >> 16U));
    const std::uint32_t result =
        static_cast<std::uint32_t>(
            FloatToBf16(gate0 / (1.0F + __expf(-gate0)) * up0)) |
        (static_cast<std::uint32_t>(
             FloatToBf16(gate1 / (1.0F + __expf(-gate1)) * up1))
         << 16U);
    *reinterpret_cast<std::uint32_t*>(output + output_base + offset) = result;
  }
}

static __global__ void Bf16SiluLookupKernel(float* __restrict__ lookup) {
  const std::uint32_t bits = blockIdx.x * blockDim.x + threadIdx.x;
  if (bits >= (1U << 16U)) {
    return;
  }
  const float gate = Bf16ToFloat(static_cast<std::uint16_t>(bits));
  lookup[bits] = gate / (1.0F + __expf(-gate));
}

inline void LaunchBf16SiluLookup(float* lookup, hipStream_t stream) {
  constexpr std::uint32_t kEntries = 1U << 16U;
  constexpr std::uint32_t kThreads = 256;
  hipLaunchKernelGGL(Bf16SiluLookupKernel,
                     dim3((kEntries + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, lookup);
}

__device__ __forceinline__ std::uint16_t ApplySwiGluLookup(
    std::uint16_t gate, std::uint16_t up, const float* silu_lookup) {
  return FloatToBf16(silu_lookup[gate] * Bf16ToFloat(up));
}

inline void LaunchSwiGlu(const std::uint16_t* fused, std::uint16_t* output,
                         std::uint32_t rows, std::uint32_t width,
                         hipStream_t stream) {
  if (width % 8U == 0U) {
    constexpr std::uint32_t kShortSequenceThreads = 128;
    constexpr std::uint32_t kLongSequenceThreads = 256;
    constexpr std::uint32_t kValuesPerThread = 8;
    const std::uint32_t threads =
        rows >= 32'768 ? kLongSequenceThreads : kShortSequenceThreads;
    hipLaunchKernelGGL(SwiGluVector8Kernel,
                       dim3((width + threads * kValuesPerThread - 1U) /
                                (threads * kValuesPerThread),
                            rows),
                       dim3(threads), 0, stream, fused, output, rows, width);
  } else {
    constexpr std::uint32_t kThreads = 256;
    hipLaunchKernelGGL(SwiGluKernel,
                       dim3((width + kThreads - 1U) / kThreads, rows),
                       dim3(kThreads), 0, stream, fused, output, rows, width);
  }
}

static __global__ void RmsNormKernel(const std::uint16_t* input,
                                     const std::uint16_t* weight,
                                     std::uint16_t* output, std::uint32_t rows,
                                     std::uint32_t width, float epsilon) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float reductions[256];
  const std::size_t base = static_cast<std::size_t>(row) * width;
  float local = 0.0F;
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    const float value = Bf16ToFloat(input[base + column]);
    local = fmaf(value, value, local);
  }
  reductions[lane] = local;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
    if (lane < stride) {
      reductions[lane] += reductions[lane + stride];
    }
    __syncthreads();
  }
  const float inverse =
      rsqrtf(reductions[0] / static_cast<float>(width) + epsilon);
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    output[base + column] = FloatToBf16(Bf16ToFloat(input[base + column]) *
                                        inverse * Bf16ToFloat(weight[column]));
  }
}

inline void LaunchRmsNorm(const std::uint16_t* input,
                          const std::uint16_t* weight, std::uint16_t* output,
                          std::uint32_t rows, std::uint32_t width,
                          float epsilon, hipStream_t stream) {
  hipLaunchKernelGGL(RmsNormKernel, dim3(rows), dim3(256), 0, stream, input,
                     weight, output, rows, width, epsilon);
}

static __global__ void LayerNormKernel(const std::uint16_t* input,
                                       const std::uint16_t* weight,
                                       const std::uint16_t* bias,
                                       std::uint16_t* output,
                                       std::uint32_t rows, std::uint32_t width,
                                       float epsilon) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float sums[256];
  __shared__ float squares[256];
  const std::size_t base = static_cast<std::size_t>(row) * width;
  float local_sum = 0.0F;
  float local_square = 0.0F;
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    const float value = Bf16ToFloat(input[base + column]);
    local_sum += value;
    local_square = fmaf(value, value, local_square);
  }
  sums[lane] = local_sum;
  squares[lane] = local_square;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
    if (lane < stride) {
      sums[lane] += sums[lane + stride];
      squares[lane] += squares[lane + stride];
    }
    __syncthreads();
  }
  const float mean = sums[0] / static_cast<float>(width);
  const float variance =
      fmaxf(0.0F, squares[0] / static_cast<float>(width) - mean * mean);
  const float inverse = rsqrtf(variance + epsilon);
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    const float normalized =
        (Bf16ToFloat(input[base + column]) - mean) * inverse;
    output[base + column] =
        FloatToBf16(normalized * Bf16ToFloat(weight[column]) +
                    (bias == nullptr ? 0.0F : Bf16ToFloat(bias[column])));
  }
}

inline void LaunchLayerNorm(const std::uint16_t* input,
                            const std::uint16_t* weight,
                            const std::uint16_t* bias, std::uint16_t* output,
                            std::uint32_t rows, std::uint32_t width,
                            float epsilon, hipStream_t stream) {
  hipLaunchKernelGGL(LayerNormKernel, dim3(rows), dim3(256), 0, stream, input,
                     weight, bias, output, rows, width, epsilon);
}

static __global__ void FinalAdaLnKernel(
    const std::uint16_t* input, const std::uint16_t* weight,
    const std::uint16_t* modulation, std::uint32_t modulation_row,
    float* output, std::uint32_t rows, std::uint32_t width, std::uint32_t slots,
    std::uint32_t shift_slot, std::uint32_t scale_slot, float epsilon) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float reductions[256];
  const std::size_t row_base = static_cast<std::size_t>(row) * width;
  float local = 0.0F;
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    const float value = Bf16ToFloat(input[row_base + column]);
    local = fmaf(value, value, local);
  }
  reductions[lane] = local;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
    if (lane < stride) {
      reductions[lane] += reductions[lane + stride];
    }
    __syncthreads();
  }
  const float inverse =
      rsqrtf(reductions[0] / static_cast<float>(width) + epsilon);
  const std::size_t modulation_base =
      static_cast<std::size_t>(modulation_row) * slots * width;
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    const float normalized = Bf16ToFloat(input[row_base + column]) * inverse *
                             Bf16ToFloat(weight[column]);
    const float scale = Bf16ToFloat(
        modulation[modulation_base +
                   static_cast<std::size_t>(scale_slot) * width + column]);
    const float shift = Bf16ToFloat(
        modulation[modulation_base +
                   static_cast<std::size_t>(shift_slot) * width + column]);
    // The final F32 projection consumes the exact rounded BF16 boundary.
    // Widen in registers instead of storing and rereading a BF16 tensor.
    output[row_base + column] =
        Bf16ToFloat(FloatToBf16(normalized * (1.0F + scale) + shift));
  }
}

inline void LaunchFinalAdaLn(const std::uint16_t* input,
                             const std::uint16_t* weight,
                             const std::uint16_t* modulation,
                             std::uint32_t modulation_row, float* output,
                             std::uint32_t rows, std::uint32_t width,
                             std::uint32_t slots, std::uint32_t shift_slot,
                             std::uint32_t scale_slot, float epsilon,
                             hipStream_t stream) {
  hipLaunchKernelGGL(FinalAdaLnKernel, dim3(rows), dim3(256), 0, stream, input,
                     weight, modulation, modulation_row, output, rows, width,
                     slots, shift_slot, scale_slot, epsilon);
}

static __global__ void RmsInverseKernel(const std::uint16_t* input,
                                        float* inverse, std::uint32_t rows,
                                        std::uint32_t width, float epsilon) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float reductions[256];
  const std::size_t row_base = static_cast<std::size_t>(row) * width;
  float local = 0.0F;
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    const float value = Bf16ToFloat(input[row_base + column]);
    local = fmaf(value, value, local);
  }
  reductions[lane] = local;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
    if (lane < stride) {
      reductions[lane] += reductions[lane + stride];
    }
    __syncthreads();
  }
  if (lane == 0) {
    inverse[row] = rsqrtf(reductions[0] / static_cast<float>(width) + epsilon);
  }
}

static __global__ void RmsInverseLongKernel(const std::uint16_t* input,
                                            float* inverse, std::uint32_t rows,
                                            std::uint32_t width,
                                            float epsilon) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float reductions[256];
  const std::size_t row_base = static_cast<std::size_t>(row) * width;
  float local = 0.0F;
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    const float value = Bf16ToFloat(input[row_base + column]);
    local = fmaf(value, value, local);
  }
  reductions[lane] = local;
  __syncthreads();
  if (lane < 128U) {
    reductions[lane] += reductions[lane + 128U];
  }
  __syncthreads();
  if (lane < 64U) {
    reductions[lane] += reductions[lane + 64U];
  }
  __syncthreads();
  if (lane < 32U) {
    float value = reductions[lane] + reductions[lane + 32U];
    for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
      const float peer = __shfl_down(value, stride, 32);
      if (lane < stride) {
        value += peer;
      }
    }
    if (lane == 0U) {
      inverse[row] = rsqrtf(value / static_cast<float>(width) + epsilon);
    }
  }
}

static __global__ void AdaLnApplyVector8Kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint16_t* __restrict__ weight,
    const std::uint16_t* __restrict__ modulation,
    const std::uint32_t* __restrict__ row_map,
    const float* __restrict__ inverse, std::uint16_t* __restrict__ output,
    std::uint32_t rows, std::uint32_t width, std::uint32_t slots,
    std::uint32_t shift_slot, std::uint32_t scale_slot) {
  constexpr std::uint32_t kValuesPerThread = 8;
  const std::uint32_t vector_column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t column = vector_column * kValuesPerThread;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || column >= width) {
    return;
  }
  const std::size_t row_base = static_cast<std::size_t>(row) * width + column;
  const std::size_t modulation_base =
      static_cast<std::size_t>(row_map[row]) * slots * width;
  const std::size_t shift_base =
      modulation_base + static_cast<std::size_t>(shift_slot) * width + column;
  const std::size_t scale_base =
      modulation_base + static_cast<std::size_t>(scale_slot) * width + column;
  const float row_inverse = inverse[row];
#pragma unroll 1
  for (std::uint32_t offset = 0; offset < kValuesPerThread; offset += 2U) {
    const std::uint32_t input_pair =
        *reinterpret_cast<const std::uint32_t*>(input + row_base + offset);
    const std::uint32_t weight_pair =
        *reinterpret_cast<const std::uint32_t*>(weight + column + offset);
    const std::uint32_t shift_pair = *reinterpret_cast<const std::uint32_t*>(
        modulation + shift_base + offset);
    const std::uint32_t scale_pair = *reinterpret_cast<const std::uint32_t*>(
        modulation + scale_base + offset);
    const float normalized0 =
        Bf16ToFloat(static_cast<std::uint16_t>(input_pair & 0xFFFFU)) *
        row_inverse *
        Bf16ToFloat(static_cast<std::uint16_t>(weight_pair & 0xFFFFU));
    const float normalized1 =
        Bf16ToFloat(static_cast<std::uint16_t>(input_pair >> 16U)) *
        row_inverse *
        Bf16ToFloat(static_cast<std::uint16_t>(weight_pair >> 16U));
    const float scale0 =
        Bf16ToFloat(static_cast<std::uint16_t>(scale_pair & 0xFFFFU));
    const float scale1 =
        Bf16ToFloat(static_cast<std::uint16_t>(scale_pair >> 16U));
    const float shift0 =
        Bf16ToFloat(static_cast<std::uint16_t>(shift_pair & 0xFFFFU));
    const float shift1 =
        Bf16ToFloat(static_cast<std::uint16_t>(shift_pair >> 16U));
    const std::uint32_t result =
        static_cast<std::uint32_t>(
            FloatToBf16(normalized0 * (1.0F + scale0) + shift0)) |
        (static_cast<std::uint32_t>(
             FloatToBf16(normalized1 * (1.0F + scale1) + shift1))
         << 16U);
    *reinterpret_cast<std::uint32_t*>(output + row_base + offset) = result;
  }
}

static __global__ void AdaLnApplyRowKernel(
    const std::uint16_t* __restrict__ input,
    const std::uint16_t* __restrict__ weight,
    const std::uint16_t* __restrict__ modulation,
    const std::uint32_t* __restrict__ row_map,
    const float* __restrict__ inverse, std::uint16_t* __restrict__ output,
    std::uint32_t rows, std::uint32_t width, std::uint32_t slots,
    std::uint32_t shift_slot, std::uint32_t scale_slot) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (row >= rows) {
    return;
  }
  const std::size_t row_base = static_cast<std::size_t>(row) * width;
  const std::size_t modulation_base =
      static_cast<std::size_t>(row_map[row]) * slots * width;
  const std::size_t shift_base =
      modulation_base + static_cast<std::size_t>(shift_slot) * width;
  const std::size_t scale_base =
      modulation_base + static_cast<std::size_t>(scale_slot) * width;
  const float row_inverse = inverse[row];
#pragma unroll 1
  for (std::uint32_t column = lane * 2U; column < width;
       column += blockDim.x * 2U) {
    const std::uint32_t input_pair =
        *reinterpret_cast<const std::uint32_t*>(input + row_base + column);
    const std::uint32_t weight_pair =
        *reinterpret_cast<const std::uint32_t*>(weight + column);
    const std::uint32_t shift_pair = *reinterpret_cast<const std::uint32_t*>(
        modulation + shift_base + column);
    const std::uint32_t scale_pair = *reinterpret_cast<const std::uint32_t*>(
        modulation + scale_base + column);
    const float normalized0 =
        Bf16ToFloat(static_cast<std::uint16_t>(input_pair & 0xFFFFU)) *
        row_inverse *
        Bf16ToFloat(static_cast<std::uint16_t>(weight_pair & 0xFFFFU));
    const float normalized1 =
        Bf16ToFloat(static_cast<std::uint16_t>(input_pair >> 16U)) *
        row_inverse *
        Bf16ToFloat(static_cast<std::uint16_t>(weight_pair >> 16U));
    const float scale0 =
        Bf16ToFloat(static_cast<std::uint16_t>(scale_pair & 0xFFFFU));
    const float scale1 =
        Bf16ToFloat(static_cast<std::uint16_t>(scale_pair >> 16U));
    const float shift0 =
        Bf16ToFloat(static_cast<std::uint16_t>(shift_pair & 0xFFFFU));
    const float shift1 =
        Bf16ToFloat(static_cast<std::uint16_t>(shift_pair >> 16U));
    const std::uint32_t result =
        static_cast<std::uint32_t>(
            FloatToBf16(normalized0 * (1.0F + scale0) + shift0)) |
        (static_cast<std::uint32_t>(
             FloatToBf16(normalized1 * (1.0F + scale1) + shift1))
         << 16U);
    *reinterpret_cast<std::uint32_t*>(output + row_base + column) = result;
  }
}

inline void LaunchAdaLnSplit(const std::uint16_t* input,
                             const std::uint16_t* weight,
                             const std::uint16_t* modulation,
                             const std::uint32_t* row_map, float* inverse,
                             std::uint16_t* output, std::uint32_t rows,
                             std::uint32_t width, std::uint32_t slots,
                             std::uint32_t shift_slot, std::uint32_t scale_slot,
                             float epsilon, hipStream_t stream) {
  if (rows >= 32'768) {
    constexpr std::uint32_t kReductionThreads = 256;
    constexpr std::uint32_t kApplyThreads = 256;
    hipLaunchKernelGGL(RmsInverseLongKernel, dim3(rows),
                       dim3(kReductionThreads), 0, stream, input, inverse, rows,
                       width, epsilon);
    hipLaunchKernelGGL(AdaLnApplyRowKernel, dim3(rows), dim3(kApplyThreads), 0,
                       stream, input, weight, modulation, row_map, inverse,
                       output, rows, width, slots, shift_slot, scale_slot);
    return;
  }
  hipLaunchKernelGGL(RmsInverseKernel, dim3(rows), dim3(256), 0, stream, input,
                     inverse, rows, width, epsilon);
  constexpr std::uint32_t kThreads = 128;
  constexpr std::uint32_t kValuesPerThread = 8;
  hipLaunchKernelGGL(AdaLnApplyVector8Kernel,
                     dim3((width + kThreads * kValuesPerThread - 1U) /
                              (kThreads * kValuesPerThread),
                          rows),
                     dim3(kThreads), 0, stream, input, weight, modulation,
                     row_map, inverse, output, rows, width, slots, shift_slot,
                     scale_slot);
}

inline void LaunchAttentionRmsInverse(const std::uint16_t* input,
                                      float* inverse, std::uint32_t rows,
                                      std::uint32_t width, float epsilon,
                                      hipStream_t stream) {
  hipLaunchKernelGGL(RmsInverseLongKernel, dim3(rows), dim3(256), 0, stream,
                     input, inverse, rows, width, epsilon);
}

inline void LaunchAdaLnApplyPrepared(
    const std::uint16_t* input, const std::uint16_t* weight,
    const std::uint16_t* modulation, const std::uint32_t* row_map,
    const float* inverse, std::uint16_t* output, std::uint32_t rows,
    std::uint32_t width, std::uint32_t slots, std::uint32_t shift_slot,
    std::uint32_t scale_slot, hipStream_t stream) {
  if (rows >= 32'768) {
    constexpr std::uint32_t kThreads = 256;
    hipLaunchKernelGGL(AdaLnApplyRowKernel, dim3(rows), dim3(kThreads), 0,
                       stream, input, weight, modulation, row_map, inverse,
                       output, rows, width, slots, shift_slot, scale_slot);
    return;
  }
  constexpr std::uint32_t kThreads = 128;
  constexpr std::uint32_t kValuesPerThread = 8;
  hipLaunchKernelGGL(AdaLnApplyVector8Kernel,
                     dim3((width + kThreads * kValuesPerThread - 1U) /
                              (kThreads * kValuesPerThread),
                          rows),
                     dim3(kThreads), 0, stream, input, weight, modulation,
                     row_map, inverse, output, rows, width, slots, shift_slot,
                     scale_slot);
}

// Residual and output may be identical: each element is read before its
// sole writer updates it. Do not promise non-aliasing for these pointers.
static __global__ void GateKernel(const std::uint16_t* residual,
                                  const std::uint16_t* __restrict__ branch,
                                  const std::uint16_t* __restrict__ modulation,
                                  const std::uint32_t* __restrict__ row_map,
                                  std::uint16_t* output, std::uint32_t rows,
                                  std::uint32_t width, std::uint32_t slots,
                                  std::uint32_t gate_slot) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || column >= width) {
    return;
  }
  const std::size_t index = static_cast<std::size_t>(row) * width + column;
  const std::size_t modulation_index =
      static_cast<std::size_t>(row_map[row]) * slots * width +
      static_cast<std::size_t>(gate_slot) * width + column;
  output[index] = FloatToBf16(Bf16ToFloat(residual[index]) +
                              Bf16ToFloat(branch[index]) *
                                  Bf16ToFloat(modulation[modulation_index]));
}

static __global__ void GateVector8Kernel(
    const std::uint16_t* residual, const std::uint16_t* __restrict__ branch,
    const std::uint16_t* __restrict__ modulation,
    const std::uint32_t* __restrict__ row_map, std::uint16_t* output,
    std::uint32_t rows, std::uint32_t width, std::uint32_t slots,
    std::uint32_t gate_slot) {
  constexpr std::uint32_t kValuesPerThread = 8;
  const std::uint32_t vector_column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t column = vector_column * kValuesPerThread;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || column >= width) {
    return;
  }
  const std::size_t row_base = static_cast<std::size_t>(row) * width + column;
  const std::size_t modulation_base =
      static_cast<std::size_t>(row_map[row]) * slots * width +
      static_cast<std::size_t>(gate_slot) * width + column;
#pragma unroll 1
  for (std::uint32_t offset = 0; offset < kValuesPerThread; offset += 2U) {
    const std::uint32_t residual_pair =
        *reinterpret_cast<const std::uint32_t*>(residual + row_base + offset);
    const std::uint32_t branch_pair =
        *reinterpret_cast<const std::uint32_t*>(branch + row_base + offset);
    const std::uint32_t modulation_pair =
        *reinterpret_cast<const std::uint32_t*>(modulation + modulation_base +
                                                offset);
    const float residual0 =
        Bf16ToFloat(static_cast<std::uint16_t>(residual_pair & 0xFFFFU));
    const float residual1 =
        Bf16ToFloat(static_cast<std::uint16_t>(residual_pair >> 16U));
    const float branch0 =
        Bf16ToFloat(static_cast<std::uint16_t>(branch_pair & 0xFFFFU));
    const float branch1 =
        Bf16ToFloat(static_cast<std::uint16_t>(branch_pair >> 16U));
    const float modulation0 =
        Bf16ToFloat(static_cast<std::uint16_t>(modulation_pair & 0xFFFFU));
    const float modulation1 =
        Bf16ToFloat(static_cast<std::uint16_t>(modulation_pair >> 16U));
    const std::uint32_t result =
        static_cast<std::uint32_t>(
            FloatToBf16(residual0 + branch0 * modulation0)) |
        (static_cast<std::uint32_t>(
             FloatToBf16(residual1 + branch1 * modulation1))
         << 16U);
    *reinterpret_cast<std::uint32_t*>(output + row_base + offset) = result;
  }
}

static __global__ void GateRowKernel(
    const std::uint16_t* residual, const std::uint16_t* __restrict__ branch,
    const std::uint16_t* __restrict__ modulation,
    const std::uint32_t* __restrict__ row_map, std::uint16_t* output,
    std::uint32_t rows, std::uint32_t width, std::uint32_t slots,
    std::uint32_t gate_slot) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (row >= rows) {
    return;
  }
  const std::size_t row_base = static_cast<std::size_t>(row) * width;
  const std::size_t modulation_base =
      static_cast<std::size_t>(row_map[row]) * slots * width +
      static_cast<std::size_t>(gate_slot) * width;
#pragma unroll 1
  for (std::uint32_t column = lane * 2U; column < width;
       column += blockDim.x * 2U) {
    const std::uint32_t residual_pair =
        *reinterpret_cast<const std::uint32_t*>(residual + row_base + column);
    const std::uint32_t branch_pair =
        *reinterpret_cast<const std::uint32_t*>(branch + row_base + column);
    const std::uint32_t modulation_pair =
        *reinterpret_cast<const std::uint32_t*>(modulation + modulation_base +
                                                column);
    const float residual0 =
        Bf16ToFloat(static_cast<std::uint16_t>(residual_pair & 0xFFFFU));
    const float residual1 =
        Bf16ToFloat(static_cast<std::uint16_t>(residual_pair >> 16U));
    const float branch0 =
        Bf16ToFloat(static_cast<std::uint16_t>(branch_pair & 0xFFFFU));
    const float branch1 =
        Bf16ToFloat(static_cast<std::uint16_t>(branch_pair >> 16U));
    const float modulation0 =
        Bf16ToFloat(static_cast<std::uint16_t>(modulation_pair & 0xFFFFU));
    const float modulation1 =
        Bf16ToFloat(static_cast<std::uint16_t>(modulation_pair >> 16U));
    const std::uint32_t result =
        static_cast<std::uint32_t>(
            FloatToBf16(residual0 + branch0 * modulation0)) |
        (static_cast<std::uint32_t>(
             FloatToBf16(residual1 + branch1 * modulation1))
         << 16U);
    *reinterpret_cast<std::uint32_t*>(output + row_base + column) = result;
  }
}

static __global__ void GateRmsInverseLongKernel(
    const std::uint16_t* residual, const std::uint16_t* __restrict__ branch,
    const std::uint16_t* __restrict__ modulation,
    const std::uint32_t* __restrict__ row_map, std::uint16_t* output,
    float* __restrict__ inverse, std::uint32_t rows, std::uint32_t width,
    std::uint32_t slots, std::uint32_t gate_slot, float epsilon) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float reductions[256];
  const std::size_t row_base = static_cast<std::size_t>(row) * width;
  const std::size_t modulation_base =
      static_cast<std::size_t>(row_map[row]) * slots * width +
      static_cast<std::size_t>(gate_slot) * width;
  float local = 0.0F;
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    const float gated = Bf16ToFloat(residual[row_base + column]) +
                        Bf16ToFloat(branch[row_base + column]) *
                            Bf16ToFloat(modulation[modulation_base + column]);
    const std::uint16_t rounded = FloatToBf16(gated);
    output[row_base + column] = rounded;
    const float value = Bf16ToFloat(rounded);
    local = fmaf(value, value, local);
  }
  reductions[lane] = local;
  __syncthreads();
  if (lane < 128U) {
    reductions[lane] += reductions[lane + 128U];
  }
  __syncthreads();
  if (lane < 64U) {
    reductions[lane] += reductions[lane + 64U];
  }
  __syncthreads();
  if (lane < 32U) {
    float value = reductions[lane] + reductions[lane + 32U];
    for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
      const float peer = __shfl_down(value, stride, 32);
      if (lane < stride) {
        value += peer;
      }
    }
    if (lane == 0U) {
      inverse[row] = rsqrtf(value / static_cast<float>(width) + epsilon);
    }
  }
}

inline void LaunchGateRmsInverse(const std::uint16_t* residual,
                                 const std::uint16_t* branch,
                                 const std::uint16_t* modulation,
                                 const std::uint32_t* row_map,
                                 std::uint16_t* output, float* inverse,
                                 std::uint32_t rows, std::uint32_t width,
                                 std::uint32_t slots, std::uint32_t gate_slot,
                                 float epsilon, hipStream_t stream) {
  hipLaunchKernelGGL(GateRmsInverseLongKernel, dim3(rows), dim3(256), 0, stream,
                     residual, branch, modulation, row_map, output, inverse,
                     rows, width, slots, gate_slot, epsilon);
}

inline void LaunchGate(const std::uint16_t* residual,
                       const std::uint16_t* branch,
                       const std::uint16_t* modulation,
                       const std::uint32_t* row_map, std::uint16_t* output,
                       std::uint32_t rows, std::uint32_t width,
                       std::uint32_t slots, std::uint32_t gate_slot,
                       hipStream_t stream) {
  if (width % 8U == 0U) {
    if (rows >= 32'768) {
      constexpr std::uint32_t kThreads = 256;
      hipLaunchKernelGGL(GateRowKernel, dim3(rows), dim3(kThreads), 0, stream,
                         residual, branch, modulation, row_map, output, rows,
                         width, slots, gate_slot);
      return;
    }
    constexpr std::uint32_t kThreads = 128;
    constexpr std::uint32_t kValuesPerThread = 8;
    hipLaunchKernelGGL(GateVector8Kernel,
                       dim3((width + kThreads * kValuesPerThread - 1U) /
                                (kThreads * kValuesPerThread),
                            rows),
                       dim3(kThreads), 0, stream, residual, branch, modulation,
                       row_map, output, rows, width, slots, gate_slot);
  } else {
    constexpr std::uint32_t kThreads = 256;
    hipLaunchKernelGGL(GateKernel,
                       dim3((width + kThreads - 1U) / kThreads, rows),
                       dim3(kThreads), 0, stream, residual, branch, modulation,
                       row_map, output, rows, width, slots, gate_slot);
  }
}

static __global__ void GroupedQkvNormRopeKernel(
    const std::uint16_t* qkv, const std::uint16_t* query_weight,
    const std::uint16_t* key_weight, const std::uint16_t* rope_cos,
    const std::uint16_t* rope_sin, std::uint16_t* query, std::uint16_t* key,
    std::uint16_t* value, std::uint32_t sequence, std::uint32_t heads,
    std::uint32_t head_dimension, std::uint32_t rope_half, float epsilon,
    bool head_major) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t dimension = threadIdx.x;
  if (row >= sequence || head >= heads || dimension >= head_dimension) {
    return;
  }
  const std::size_t inner = static_cast<std::size_t>(heads) * head_dimension;
  const std::size_t row_base = static_cast<std::size_t>(row) * inner * 3U;
  const std::size_t query_base =
      row_base + static_cast<std::size_t>(head) * head_dimension * 3U;
  const std::size_t key_base = query_base + head_dimension;
  const std::size_t value_base = key_base + head_dimension;
  const float raw_query = Bf16ToFloat(qkv[query_base + dimension]);
  const float raw_key = Bf16ToFloat(qkv[key_base + dimension]);
  __shared__ float query_sums[128];
  __shared__ float key_sums[128];
  __shared__ float inverses[2];
  query_sums[dimension] = raw_query * raw_query;
  key_sums[dimension] = raw_key * raw_key;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2U; offset > 0; offset /= 2U) {
    if (dimension < offset) {
      query_sums[dimension] += query_sums[dimension + offset];
      key_sums[dimension] += key_sums[dimension + offset];
    }
    __syncthreads();
  }
  if (dimension == 0) {
    inverses[0] =
        rsqrtf(query_sums[0] / static_cast<float>(head_dimension) + epsilon);
    inverses[1] =
        rsqrtf(key_sums[0] / static_cast<float>(head_dimension) + epsilon);
  }
  __syncthreads();
  const std::size_t output =
      (head_major ? static_cast<std::size_t>(head) * sequence + row
                  : static_cast<std::size_t>(row) * heads + head) *
          head_dimension +
      dimension;
  value[output] = qkv[value_base + dimension];
  if (dimension < rope_half) {
    const std::uint32_t pair = dimension + rope_half;
    const float query_value =
        raw_query * inverses[0] * Bf16ToFloat(query_weight[dimension]);
    const float key_value =
        raw_key * inverses[1] * Bf16ToFloat(key_weight[dimension]);
    const float paired_query = Bf16ToFloat(qkv[query_base + pair]) *
                               inverses[0] * Bf16ToFloat(query_weight[pair]);
    const float paired_key = Bf16ToFloat(qkv[key_base + pair]) * inverses[1] *
                             Bf16ToFloat(key_weight[pair]);
    const float cosine = Bf16ToFloat(
        rope_cos[static_cast<std::size_t>(row) * rope_half + dimension]);
    const float sine = Bf16ToFloat(
        rope_sin[static_cast<std::size_t>(row) * rope_half + dimension]);
    query[output] = FloatToBf16(query_value * cosine - paired_query * sine);
    key[output] = FloatToBf16(key_value * cosine - paired_key * sine);
    query[output + rope_half] =
        FloatToBf16(paired_query * cosine + query_value * sine);
    key[output + rope_half] =
        FloatToBf16(paired_key * cosine + key_value * sine);
  } else if (dimension >= rope_half * 2U) {
    query[output] = FloatToBf16(raw_query * inverses[0] *
                                Bf16ToFloat(query_weight[dimension]));
    key[output] =
        FloatToBf16(raw_key * inverses[1] * Bf16ToFloat(key_weight[dimension]));
  }
}

// One wave owns a head. Retain the original four-term norm reduction and
// direct paired RoPE expressions: shuffling pre-normalized pairs changes
// compiler contraction at BF16 rounding boundaries.
static __global__ void GroupedQkvNormRopeWaveKernel(
    const std::uint16_t* __restrict__ qkv,
    const std::uint16_t* __restrict__ query_weight,
    const std::uint16_t* __restrict__ key_weight,
    const std::uint16_t* __restrict__ rope_cos,
    const std::uint16_t* __restrict__ rope_sin,
    std::uint16_t* __restrict__ query, std::uint16_t* __restrict__ key,
    std::uint16_t* __restrict__ value, std::uint32_t sequence,
    std::uint32_t heads, float epsilon, bool head_major) {
  constexpr std::uint32_t kHeadDimension = 128;
  constexpr std::uint32_t kRopeHalf = 48;
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t head = blockIdx.y * 4 + threadIdx.x / 32;
  const std::uint32_t lane = threadIdx.x % 32;
  if (row >= sequence || head >= heads) {
    return;
  }
  const std::size_t inner = static_cast<std::size_t>(heads) * kHeadDimension;
  const std::size_t row_base = static_cast<std::size_t>(row) * inner * 3U;
  const std::size_t query_base =
      row_base + static_cast<std::size_t>(head) * kHeadDimension * 3U;
  const std::size_t key_base = query_base + kHeadDimension;
  const std::size_t value_base = key_base + kHeadDimension;
  float query_sum = 0, key_sum = 0;
#pragma unroll
  for (unsigned dim = lane; dim < 128; dim += 32) {
    float q = Bf16ToFloat(qkv[query_base + dim]),
          k = Bf16ToFloat(qkv[key_base + dim]);
    query_sum = fmaf(q, q, query_sum);
    key_sum = fmaf(k, k, key_sum);
  }
#pragma unroll
  for (unsigned off = 16; off; off /= 2) {
    query_sum += __shfl_down(query_sum, off);
    key_sum += __shfl_down(key_sum, off);
  }
  const float query_inverse = rsqrtf(__shfl(query_sum, 0) / 128.f + epsilon);
  const float key_inverse = rsqrtf(__shfl(key_sum, 0) / 128.f + epsilon);
#pragma unroll
  for (unsigned dimension = lane; dimension < 128; dimension += 32) {
    const std::size_t output = (head_major ? std::size_t(head) * sequence + row
                                           : std::size_t(row) * heads + head) *
                                   128 +
                               dimension;
    value[output] = qkv[value_base + dimension];
    if (dimension < kRopeHalf) {
      const std::uint32_t pair = dimension + kRopeHalf;
      const float query_value = Bf16ToFloat(qkv[query_base + dimension]) *
                                query_inverse *
                                Bf16ToFloat(query_weight[dimension]);
      const float key_value = Bf16ToFloat(qkv[key_base + dimension]) *
                              key_inverse * Bf16ToFloat(key_weight[dimension]);
      const float paired_query = Bf16ToFloat(qkv[query_base + pair]) *
                                 query_inverse *
                                 Bf16ToFloat(query_weight[pair]);
      const float paired_key = Bf16ToFloat(qkv[key_base + pair]) * key_inverse *
                               Bf16ToFloat(key_weight[pair]);
      const std::size_t rope =
          static_cast<std::size_t>(row) * kRopeHalf + dimension;
      const float cosine = Bf16ToFloat(rope_cos[rope]);
      const float sine = Bf16ToFloat(rope_sin[rope]);
      query[output] = FloatToBf16(query_value * cosine - paired_query * sine);
      query[output + kRopeHalf] =
          FloatToBf16(paired_query * cosine + query_value * sine);
      key[output] = FloatToBf16(key_value * cosine - paired_key * sine);
      key[output + kRopeHalf] =
          FloatToBf16(paired_key * cosine + key_value * sine);
    } else if (dimension >= kRopeHalf * 2U) {
      query[output] =
          FloatToBf16(Bf16ToFloat(qkv[query_base + dimension]) * query_inverse *
                      Bf16ToFloat(query_weight[dimension]));
      key[output] =
          FloatToBf16(Bf16ToFloat(qkv[key_base + dimension]) * key_inverse *
                      Bf16ToFloat(key_weight[dimension]));
    }
  }
}

inline void LaunchGroupedQkvNormRope(
    const std::uint16_t* qkv, const std::uint16_t* query_weight,
    const std::uint16_t* key_weight, const std::uint16_t* rope_cos,
    const std::uint16_t* rope_sin, std::uint16_t* query, std::uint16_t* key,
    std::uint16_t* value, std::uint32_t sequence, std::uint32_t heads,
    std::uint32_t head_dimension, std::uint32_t rope_half, float epsilon,
    hipStream_t stream, bool head_major = false) {
  if (head_dimension == 128U && rope_half == 48U) {
    constexpr std::uint32_t kHeadsPerBlock = 4;
    hipLaunchKernelGGL(
        GroupedQkvNormRopeWaveKernel,
        dim3(sequence, (heads + kHeadsPerBlock - 1U) / kHeadsPerBlock),
        dim3(128), 0, stream, qkv, query_weight, key_weight, rope_cos, rope_sin,
        query, key, value, sequence, heads, epsilon, head_major);
  } else {
    hipLaunchKernelGGL(GroupedQkvNormRopeKernel, dim3(sequence, heads),
                       dim3(head_dimension), 0, stream, qkv, query_weight,
                       key_weight, rope_cos, rope_sin, query, key, value,
                       sequence, heads, head_dimension, rope_half, epsilon,
                       head_major);
  }
}

static __global__ void HeadMajorToRowMajorKernel(
    const std::uint16_t* __restrict__ input, std::uint16_t* __restrict__ output,
    std::uint32_t sequence, std::uint32_t heads) {
  constexpr std::uint32_t kHeadDimension = 128;
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t dimension = threadIdx.x;
  if (row >= sequence || head >= heads || dimension >= kHeadDimension) {
    return;
  }
  output[(static_cast<std::size_t>(row) * heads + head) * kHeadDimension +
         dimension] =
      input[(static_cast<std::size_t>(head) * sequence + row) * kHeadDimension +
            dimension];
}

inline void LaunchHeadMajorToRowMajor(const std::uint16_t* input,
                                      std::uint16_t* output,
                                      std::uint32_t sequence,
                                      std::uint32_t heads, hipStream_t stream) {
  hipLaunchKernelGGL(HeadMajorToRowMajorKernel, dim3(sequence, heads),
                     dim3(128), 0, stream, input, output, sequence, heads);
}

static __global__ void FullAttentionScalarKernel(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value, std::uint16_t* output, std::uint32_t sequence,
    std::uint32_t heads, std::uint32_t head_dimension, float scale) {
  const std::uint32_t query_row = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t lane = threadIdx.x;
  if (query_row >= sequence || head >= heads) {
    return;
  }
  extern __shared__ float scores[];
  const std::size_t query_base =
      (static_cast<std::size_t>(query_row) * heads + head) * head_dimension;
  if (lane == 0) {
    float maximum = -INFINITY;
    for (std::uint32_t key_row = 0; key_row < sequence; ++key_row) {
      const std::size_t key_base =
          (static_cast<std::size_t>(key_row) * heads + head) * head_dimension;
      float dot = 0.0F;
      for (std::uint32_t dimension = 0; dimension < head_dimension;
           ++dimension) {
        dot = fmaf(Bf16ToFloat(query[query_base + dimension]),
                   Bf16ToFloat(key[key_base + dimension]), dot);
      }
      scores[key_row] = dot * scale;
      maximum = fmaxf(maximum, scores[key_row]);
    }
    float sum = 0.0F;
    for (std::uint32_t key_row = 0; key_row < sequence; ++key_row) {
      scores[key_row] = expf(scores[key_row] - maximum);
      sum += scores[key_row];
    }
    const float inverse_sum = 1.0F / sum;
    for (std::uint32_t key_row = 0; key_row < sequence; ++key_row) {
      scores[key_row] *= inverse_sum;
    }
  }
  __syncthreads();
  for (std::uint32_t dimension = lane; dimension < head_dimension;
       dimension += blockDim.x) {
    float sum = 0.0F;
    for (std::uint32_t key_row = 0; key_row < sequence; ++key_row) {
      const std::size_t value_index =
          (static_cast<std::size_t>(key_row) * heads + head) * head_dimension +
          dimension;
      sum = fmaf(scores[key_row], Bf16ToFloat(value[value_index]), sum);
    }
    output[query_base + dimension] = FloatToBf16(sum);
  }
}

static __global__ void FullAttentionRowParallelKernel(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value, std::uint16_t* output, std::uint32_t sequence,
    std::uint32_t heads, std::uint32_t head_dimension, float scale) {
  const std::uint32_t query_row = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t lane = threadIdx.x;
  if (query_row >= sequence || head >= heads) {
    return;
  }
  extern __shared__ float scores[];
  const std::size_t query_base =
      (static_cast<std::size_t>(query_row) * heads + head) * head_dimension;
  // Key rows are independent. Parallelizing only this outer loop preserves
  // the released per-dot FMA order; lane 0 still performs the softmax
  // max/sum/normalization in the original key-row order, and each output
  // dimension retains its original V accumulation order.
  for (std::uint32_t key_row = lane; key_row < sequence;
       key_row += blockDim.x) {
    const std::size_t key_base =
        (static_cast<std::size_t>(key_row) * heads + head) * head_dimension;
    float dot = 0.0F;
    for (std::uint32_t dimension = 0; dimension < head_dimension; ++dimension) {
      dot = fmaf(Bf16ToFloat(query[query_base + dimension]),
                 Bf16ToFloat(key[key_base + dimension]), dot);
    }
    scores[key_row] = dot * scale;
  }
  __syncthreads();
  float score_zero = 0.0F;
  if (lane == 0) {
    score_zero = scores[0];
    float maximum = -INFINITY;
    for (std::uint32_t key_row = 0; key_row < sequence; ++key_row) {
      maximum = fmaxf(maximum, scores[key_row]);
    }
    scores[0] = maximum;
  }
  __syncthreads();
  const float maximum = scores[0];
  __syncthreads();
  for (std::uint32_t key_row = lane; key_row < sequence;
       key_row += blockDim.x) {
    const float score = key_row == 0 ? score_zero : scores[key_row];
    scores[key_row] = expf(score - maximum);
  }
  __syncthreads();
  float exponential_zero = 0.0F;
  if (lane == 0) {
    exponential_zero = scores[0];
    float sum = 0.0F;
    for (std::uint32_t key_row = 0; key_row < sequence; ++key_row) {
      sum += scores[key_row];
    }
    scores[0] = 1.0F / sum;
  }
  __syncthreads();
  const float inverse_sum = scores[0];
  __syncthreads();
  for (std::uint32_t key_row = lane; key_row < sequence;
       key_row += blockDim.x) {
    const float exponential = key_row == 0 ? exponential_zero : scores[key_row];
    scores[key_row] = exponential * inverse_sum;
  }
  __syncthreads();
  for (std::uint32_t dimension = lane; dimension < head_dimension;
       dimension += blockDim.x) {
    float sum = 0.0F;
    for (std::uint32_t key_row = 0; key_row < sequence; ++key_row) {
      const std::size_t value_index =
          (static_cast<std::size_t>(key_row) * heads + head) * head_dimension +
          dimension;
      sum = fmaf(scores[key_row], Bf16ToFloat(value[value_index]), sum);
    }
    output[query_base + dimension] = FloatToBf16(sum);
  }
}

inline void LaunchFullAttention(const std::uint16_t* query,
                                const std::uint16_t* key,
                                const std::uint16_t* value,
                                std::uint16_t* output, std::uint32_t sequence,
                                std::uint32_t heads,
                                std::uint32_t head_dimension, float scale,
                                hipStream_t stream, bool row_parallel = true) {
  constexpr std::uint32_t kThreads = 128;
  const dim3 grid(sequence, heads);
  const dim3 block(kThreads);
  const std::size_t shared_bytes =
      static_cast<std::size_t>(sequence) * sizeof(float);
  if (row_parallel) {
    hipLaunchKernelGGL(FullAttentionRowParallelKernel, grid, block,
                       shared_bytes, stream, query, key, value, output,
                       sequence, heads, head_dimension, scale);
  } else {
    hipLaunchKernelGGL(FullAttentionScalarKernel, grid, block, shared_bytes,
                       stream, query, key, value, output, sequence, heads,
                       head_dimension, scale);
  }
}

static __global__ void SoftmaxScoresToBf16SharedKernel(
    const float* scores, std::uint16_t* probabilities, std::uint32_t sequence,
    std::uint32_t heads) {
  const std::uint32_t query_row = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t lane = threadIdx.x;
  if (query_row >= sequence || head >= heads) {
    return;
  }
  extern __shared__ float row_scores[];
  __shared__ float reduction[128];
  const std::size_t base =
      (static_cast<std::size_t>(head) * sequence + query_row) * sequence;
  float local_maximum = -INFINITY;
  for (std::uint32_t key_row = lane; key_row < sequence;
       key_row += blockDim.x) {
    const float score = scores[base + key_row];
    row_scores[key_row] = score;
    local_maximum = fmaxf(local_maximum, score);
  }
  reduction[lane] = local_maximum;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2U; offset > 0; offset /= 2U) {
    if (lane < offset) {
      reduction[lane] = fmaxf(reduction[lane], reduction[lane + offset]);
    }
    __syncthreads();
  }
  const float maximum = reduction[0];
  float local_sum = 0.0F;
  for (std::uint32_t key_row = lane; key_row < sequence;
       key_row += blockDim.x) {
    const float value = expf(row_scores[key_row] - maximum);
    row_scores[key_row] = value;
    local_sum += value;
  }
  reduction[lane] = local_sum;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2U; offset > 0; offset /= 2U) {
    if (lane < offset) {
      reduction[lane] += reduction[lane + offset];
    }
    __syncthreads();
  }
  const float inverse_sum = 1.0F / reduction[0];
  for (std::uint32_t key_row = lane; key_row < sequence;
       key_row += blockDim.x) {
    probabilities[base + key_row] =
        FloatToBf16(row_scores[key_row] * inverse_sum);
  }
}

template<std::uint32_t kWaves>
__device__ __forceinline__ float SoftmaxBlockReduceMax(float value,
                                                       float* wave_values) {
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wave = threadIdx.x >> 5U;
#pragma unroll
  for (std::uint32_t offset = 16; offset > 0; offset /= 2U) {
    value = fmaxf(value, __shfl_down(value, offset));
  }
  if (lane == 0) {
    wave_values[wave] = value;
  }
  __syncthreads();
  value = threadIdx.x < kWaves ? wave_values[lane] : -INFINITY;
  if (wave == 0) {
#pragma unroll
    for (std::uint32_t offset = 16; offset > 0; offset /= 2U) {
      value = fmaxf(value, __shfl_down(value, offset));
    }
    if (lane == 0) {
      wave_values[0] = value;
    }
  }
  __syncthreads();
  return wave_values[0];
}

template<std::uint32_t kWaves>
__device__ __forceinline__ float SoftmaxBlockReduceSum(float value,
                                                       float* wave_values) {
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wave = threadIdx.x >> 5U;
#pragma unroll
  for (std::uint32_t offset = 16; offset > 0; offset /= 2U) {
    value += __shfl_down(value, offset);
  }
  if (lane == 0) {
    wave_values[wave] = value;
  }
  __syncthreads();
  value = threadIdx.x < kWaves ? wave_values[lane] : 0.0F;
  if (wave == 0) {
#pragma unroll
    for (std::uint32_t offset = 16; offset > 0; offset /= 2U) {
      value += __shfl_down(value, offset);
    }
    if (lane == 0) {
      wave_values[0] = value;
    }
  }
  __syncthreads();
  return wave_values[0];
}

__device__ __forceinline__ float LoadSoftmaxScore(const float* scores,
                                                  std::size_t base,
                                                  std::uint32_t key_row,
                                                  std::uint32_t sequence) {
  return key_row < sequence ? scores[base + key_row] : -INFINITY;
}

__device__ __forceinline__ void StoreSoftmaxProbability(
    std::uint16_t* probabilities, std::size_t base, std::uint32_t key_row,
    std::uint32_t sequence, float value) {
  if (key_row < sequence) {
    probabilities[base + key_row] = FloatToBf16(value);
  }
}

static __global__ void SoftmaxScoresToBf16Cached128Kernel(
    const float* scores, std::uint16_t* probabilities, std::uint32_t sequence,
    std::uint32_t heads) {
  constexpr std::uint32_t kThreads = 128;
  const std::uint32_t query_row = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t lane = threadIdx.x;
  if (query_row >= sequence || head >= heads) {
    return;
  }
  const std::size_t base =
      (static_cast<std::size_t>(head) * sequence + query_row) * sequence;
  float4 score0 = {
      LoadSoftmaxScore(scores, base, lane + 0U * kThreads, sequence),
      LoadSoftmaxScore(scores, base, lane + 1U * kThreads, sequence),
      LoadSoftmaxScore(scores, base, lane + 2U * kThreads, sequence),
      LoadSoftmaxScore(scores, base, lane + 3U * kThreads, sequence)};
  float4 score1 = {
      LoadSoftmaxScore(scores, base, lane + 4U * kThreads, sequence),
      LoadSoftmaxScore(scores, base, lane + 5U * kThreads, sequence),
      LoadSoftmaxScore(scores, base, lane + 6U * kThreads, sequence),
      LoadSoftmaxScore(scores, base, lane + 7U * kThreads, sequence)};
  float4 score2 = {
      LoadSoftmaxScore(scores, base, lane + 8U * kThreads, sequence),
      LoadSoftmaxScore(scores, base, lane + 9U * kThreads, sequence),
      LoadSoftmaxScore(scores, base, lane + 10U * kThreads, sequence),
      LoadSoftmaxScore(scores, base, lane + 11U * kThreads, sequence)};
  float4 score3 = {
      LoadSoftmaxScore(scores, base, lane + 12U * kThreads, sequence),
      LoadSoftmaxScore(scores, base, lane + 13U * kThreads, sequence),
      LoadSoftmaxScore(scores, base, lane + 14U * kThreads, sequence),
      LoadSoftmaxScore(scores, base, lane + 15U * kThreads, sequence)};
  float local_maximum = -INFINITY;
  local_maximum = fmaxf(local_maximum, score0.x);
  local_maximum = fmaxf(local_maximum, score0.y);
  local_maximum = fmaxf(local_maximum, score0.z);
  local_maximum = fmaxf(local_maximum, score0.w);
  local_maximum = fmaxf(local_maximum, score1.x);
  local_maximum = fmaxf(local_maximum, score1.y);
  local_maximum = fmaxf(local_maximum, score1.z);
  local_maximum = fmaxf(local_maximum, score1.w);
  local_maximum = fmaxf(local_maximum, score2.x);
  local_maximum = fmaxf(local_maximum, score2.y);
  local_maximum = fmaxf(local_maximum, score2.z);
  local_maximum = fmaxf(local_maximum, score2.w);
  local_maximum = fmaxf(local_maximum, score3.x);
  local_maximum = fmaxf(local_maximum, score3.y);
  local_maximum = fmaxf(local_maximum, score3.z);
  local_maximum = fmaxf(local_maximum, score3.w);
  __shared__ float wave_values[4];
  const float maximum = SoftmaxBlockReduceMax<4>(local_maximum, wave_values);
  score0.x = __expf(score0.x - maximum);
  score0.y = __expf(score0.y - maximum);
  score0.z = __expf(score0.z - maximum);
  score0.w = __expf(score0.w - maximum);
  score1.x = __expf(score1.x - maximum);
  score1.y = __expf(score1.y - maximum);
  score1.z = __expf(score1.z - maximum);
  score1.w = __expf(score1.w - maximum);
  score2.x = __expf(score2.x - maximum);
  score2.y = __expf(score2.y - maximum);
  score2.z = __expf(score2.z - maximum);
  score2.w = __expf(score2.w - maximum);
  score3.x = __expf(score3.x - maximum);
  score3.y = __expf(score3.y - maximum);
  score3.z = __expf(score3.z - maximum);
  score3.w = __expf(score3.w - maximum);
  float local_sum = 0.0F;
  local_sum += score0.x;
  local_sum += score0.y;
  local_sum += score0.z;
  local_sum += score0.w;
  local_sum += score1.x;
  local_sum += score1.y;
  local_sum += score1.z;
  local_sum += score1.w;
  local_sum += score2.x;
  local_sum += score2.y;
  local_sum += score2.z;
  local_sum += score2.w;
  local_sum += score3.x;
  local_sum += score3.y;
  local_sum += score3.z;
  local_sum += score3.w;
  const float inverse_sum =
      1.0F / SoftmaxBlockReduceSum<4>(local_sum, wave_values);
  StoreSoftmaxProbability(probabilities, base, lane + 0U * kThreads, sequence,
                          score0.x * inverse_sum);
  StoreSoftmaxProbability(probabilities, base, lane + 1U * kThreads, sequence,
                          score0.y * inverse_sum);
  StoreSoftmaxProbability(probabilities, base, lane + 2U * kThreads, sequence,
                          score0.z * inverse_sum);
  StoreSoftmaxProbability(probabilities, base, lane + 3U * kThreads, sequence,
                          score0.w * inverse_sum);
  StoreSoftmaxProbability(probabilities, base, lane + 4U * kThreads, sequence,
                          score1.x * inverse_sum);
  StoreSoftmaxProbability(probabilities, base, lane + 5U * kThreads, sequence,
                          score1.y * inverse_sum);
  StoreSoftmaxProbability(probabilities, base, lane + 6U * kThreads, sequence,
                          score1.z * inverse_sum);
  StoreSoftmaxProbability(probabilities, base, lane + 7U * kThreads, sequence,
                          score1.w * inverse_sum);
  StoreSoftmaxProbability(probabilities, base, lane + 8U * kThreads, sequence,
                          score2.x * inverse_sum);
  StoreSoftmaxProbability(probabilities, base, lane + 9U * kThreads, sequence,
                          score2.y * inverse_sum);
  StoreSoftmaxProbability(probabilities, base, lane + 10U * kThreads, sequence,
                          score2.z * inverse_sum);
  StoreSoftmaxProbability(probabilities, base, lane + 11U * kThreads, sequence,
                          score2.w * inverse_sum);
  StoreSoftmaxProbability(probabilities, base, lane + 12U * kThreads, sequence,
                          score3.x * inverse_sum);
  StoreSoftmaxProbability(probabilities, base, lane + 13U * kThreads, sequence,
                          score3.y * inverse_sum);
  StoreSoftmaxProbability(probabilities, base, lane + 14U * kThreads, sequence,
                          score3.z * inverse_sum);
  StoreSoftmaxProbability(probabilities, base, lane + 15U * kThreads, sequence,
                          score3.w * inverse_sum);
}

__device__ __forceinline__ float SoftmaxWaveReduceMax(float value) {
#pragma unroll
  for (std::uint32_t offset = 16; offset > 0; offset /= 2U) {
    value = fmaxf(value, __shfl_down(value, offset));
  }
  return __shfl(value, 0);
}

__device__ __forceinline__ float SoftmaxWaveReduceSum(float value) {
#pragma unroll
  for (std::uint32_t offset = 16; offset > 0; offset /= 2U) {
    value += __shfl_down(value, offset);
  }
  return __shfl(value, 0);
}

static __global__ void SoftmaxScoresToBf16WaveKernel(
    float* scores, std::uint16_t* probabilities, std::uint32_t sequence,
    std::uint32_t heads) {
  constexpr std::uint32_t kRowsPerBlock = 4;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wave = threadIdx.x >> 5U;
  const std::uint32_t query_row = blockIdx.x * kRowsPerBlock + wave;
  const std::uint32_t head = blockIdx.y;
  if (query_row >= sequence || head >= heads) {
    return;
  }
  const std::size_t base =
      (static_cast<std::size_t>(head) * sequence + query_row) * sequence;
  float local_maximum = -INFINITY;
  for (std::uint32_t key_row = lane; key_row < sequence; key_row += 32U) {
    local_maximum = fmaxf(local_maximum, scores[base + key_row]);
  }
  const float maximum = SoftmaxWaveReduceMax(local_maximum);
  float local_sum = 0.0F;
  for (std::uint32_t key_row = lane; key_row < sequence; key_row += 32U) {
    const float value = __expf(scores[base + key_row] - maximum);
    scores[base + key_row] = value;
    local_sum += value;
  }
  const float inverse_sum = 1.0F / SoftmaxWaveReduceSum(local_sum);
  for (std::uint32_t key_row = lane; key_row < sequence; key_row += 32U) {
    probabilities[base + key_row] =
        FloatToBf16(scores[base + key_row] * inverse_sum);
  }
}

inline void LaunchSoftmaxScoresToBf16(float* scores,
                                      std::uint16_t* probabilities,
                                      std::uint32_t sequence,
                                      std::uint32_t heads, hipStream_t stream) {
  constexpr std::uint32_t kFallbackThreads = 128;
  if (sequence <= 640U) {
    constexpr std::uint32_t kRowsPerBlock = 4;
    hipLaunchKernelGGL(
        SoftmaxScoresToBf16WaveKernel,
        dim3((sequence + kRowsPerBlock - 1U) / kRowsPerBlock, heads), dim3(128),
        0, stream, scores, probabilities, sequence, heads);
  } else if (sequence <= 128U * 16U) {
    hipLaunchKernelGGL(SoftmaxScoresToBf16Cached128Kernel,
                       dim3(sequence, heads), dim3(128), 0, stream, scores,
                       probabilities, sequence, heads);
  } else {
    hipLaunchKernelGGL(SoftmaxScoresToBf16SharedKernel, dim3(sequence, heads),
                       dim3(kFallbackThreads),
                       static_cast<std::size_t>(sequence) * sizeof(float),
                       stream, scores, probabilities, sequence, heads);
  }
}

}  // namespace gufo::minimax_h3::dit_ops

#endif  // GUFO_MODELS_MINIMAX_H3_DIT_OPS_HIP_HPP_
