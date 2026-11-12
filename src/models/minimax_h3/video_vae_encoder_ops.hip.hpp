#ifndef GUFO_MODELS_MINIMAX_H3_VIDEO_VAE_ENCODER_OPS_HIP_HPP_
#define GUFO_MODELS_MINIMAX_H3_VIDEO_VAE_ENCODER_OPS_HIP_HPP_

// Model-private kernels for the MiniMax H3 VisualVAE *encoder* (issue #177).
// These mirror the semantics of the checkpoint's shipped bundle
// (FL2VA/video_vae/vae_cnn.py EncoderFCN3D, conv.py BaseConv3d, norm.py
// get_group_norm_3d) for the single-frame keyframe path; nothing here is
// shared with any other model.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::minimax_h3::video_vae_encoder_ops {

// torch F.pad(mode="reflect") mirrors without repeating the edge sample. The
// encoder only ever pads by one, which is legal for extents of at least two.
__device__ __forceinline__ std::int32_t ReflectIndex(std::int32_t index,
                                                     std::int32_t extent) {
  if (index < 0) {
    return -index;
  }
  if (index >= extent) {
    return 2 * (extent - 1) - index;
  }
  return index;
}

// A T=1 causal 3x3x3 convolution prepends kernel_t-1 zero frames
// (conv.py:BaseConv3d._apply_temporal_padding), so only the last temporal tap
// survives. Copy weight[co][ci][kt-1][kh][kw] into the [co][ci*kk] row-major
// matrix the F32 GEMM consumes (columns are ci*kk + kh*3 + kw).
static __global__ void FlattenCausalWeightKernel(
    const float* source, float* destination, std::uint32_t out_channels,
    std::uint32_t in_channels, std::uint32_t taps, std::uint32_t temporal) {
  const std::size_t elements =
      static_cast<std::size_t>(out_channels) * in_channels * taps;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::uint32_t tap = static_cast<std::uint32_t>(index % taps);
  const std::size_t matrix = index / taps;
  const std::uint32_t in_channel =
      static_cast<std::uint32_t>(matrix % in_channels);
  const std::size_t out_channel = matrix / in_channels;
  const std::size_t origin =
      (out_channel * in_channels + in_channel) * temporal + (temporal - 1U);
  destination[index] = source[origin * taps + tap];
}

inline void LaunchFlattenCausalWeight(const float* source, float* destination,
                                      std::uint32_t out_channels,
                                      std::uint32_t in_channels,
                                      std::uint32_t taps,
                                      std::uint32_t temporal,
                                      hipStream_t stream) {
  const std::size_t elements =
      static_cast<std::size_t>(out_channels) * in_channels * taps;
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(
      FlattenCausalWeightKernel,
      dim3(static_cast<std::uint32_t>((elements + kThreads - 1U) / kThreads)),
      dim3(kThreads), 0, stream, source, destination, out_channels, in_channels,
      taps, temporal);
}

// im2col over a reflect-padded plane. Rows of the produced matrix are output
// positions, columns are [in_channel][kh][kw]; combined with the flattened
// weight matrix this reproduces the released 3x3 causal convolution exactly.
static __global__ void Im2ColReflectKernel(
    const float* input, float* columns, std::uint32_t in_height,
    std::uint32_t in_width, std::uint32_t out_width, std::uint32_t channels,
    std::uint32_t stride, std::int32_t pad_top, std::int32_t pad_left,
    std::uint32_t base, std::uint32_t count) {
  const std::size_t per_position = static_cast<std::size_t>(channels) * 9U;
  const std::size_t elements = static_cast<std::size_t>(count) * per_position;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::uint32_t tap = static_cast<std::uint32_t>(index % 9U);
  const std::size_t identity = index / 9U;
  const std::uint32_t channel = static_cast<std::uint32_t>(identity % channels);
  const std::uint32_t position =
      base + static_cast<std::uint32_t>(identity / channels);
  const std::uint32_t y = position / out_width;
  const std::uint32_t x = position % out_width;
  const std::int32_t source_y = ReflectIndex(
      static_cast<std::int32_t>(y) * static_cast<std::int32_t>(stride) +
          static_cast<std::int32_t>(tap / 3U) - pad_top,
      static_cast<std::int32_t>(in_height));
  const std::int32_t source_x = ReflectIndex(
      static_cast<std::int32_t>(x) * static_cast<std::int32_t>(stride) +
          static_cast<std::int32_t>(tap % 3U) - pad_left,
      static_cast<std::int32_t>(in_width));
  columns[index] =
      input[(static_cast<std::size_t>(source_y) * in_width + source_x) *
                channels +
            channel];
}

inline void LaunchIm2ColReflect(const float* input, float* columns,
                                std::uint32_t in_height, std::uint32_t in_width,
                                std::uint32_t out_height,
                                std::uint32_t out_width, std::uint32_t channels,
                                std::uint32_t stride, std::int32_t pad_top,
                                std::int32_t pad_left, std::uint32_t base,
                                std::uint32_t count, hipStream_t stream) {
  const std::size_t elements = static_cast<std::size_t>(count) * channels * 9U;
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(
      Im2ColReflectKernel,
      dim3(static_cast<std::uint32_t>((elements + kThreads - 1U) / kThreads)),
      dim3(kThreads), 0, stream, input, columns, in_height, in_width, out_width,
      channels, stride, pad_top, pad_left, base, count);
}

// GroupNorm(32, eps 1e-6, affine) over one plane. Statistics never mix
// frames (norm.py:_merge_time_to_batch), so with T=1 the reference set is the
// group's channels across every position of the tile. Partial sums are
// materialized per (position chunk, group) so the reduction order - and
// therefore the result - is fixed.
static __global__ void GroupNormPartialKernel(
    const float* input, float* sums, float* squares, std::uint32_t positions,
    std::uint32_t channels, std::uint32_t groups,
    std::uint32_t positions_per_block) {
  const std::uint32_t per_group = channels / groups;
  const std::uint32_t chunk = blockIdx.x;
  const std::uint32_t group = blockIdx.y;
  const std::size_t origin =
      static_cast<std::size_t>(chunk) * positions_per_block;
  if (origin >= positions) {
    return;
  }
  const std::uint32_t count = static_cast<std::uint32_t>(
      positions_per_block < positions - origin ? positions_per_block
                                               : positions - origin);
  const std::size_t elements = static_cast<std::size_t>(count) * per_group;
  float sum = 0.0F;
  float square = 0.0F;
  for (std::size_t index = threadIdx.x; index < elements; index += blockDim.x) {
    const std::size_t position = origin + index / per_group;
    const std::size_t channel =
        static_cast<std::size_t>(group) * per_group + index % per_group;
    const float value = input[position * channels + channel];
    sum += value;
    square = fmaf(value, value, square);
  }
  constexpr std::uint32_t kThreads = 256;
  __shared__ float shared_sum[kThreads];
  __shared__ float shared_square[kThreads];
  shared_sum[threadIdx.x] = sum;
  shared_square[threadIdx.x] = square;
  __syncthreads();
#pragma unroll
  for (std::uint32_t stride = kThreads / 2U; stride >= 32U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
      shared_square[threadIdx.x] += shared_square[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x < 32U) {
#pragma unroll
    for (std::uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
      shared_sum[threadIdx.x] +=
          __shfl_down(shared_sum[threadIdx.x], offset, 32);
      shared_square[threadIdx.x] +=
          __shfl_down(shared_square[threadIdx.x], offset, 32);
    }
    if (threadIdx.x == 0U) {
      const std::size_t slot = static_cast<std::size_t>(chunk) * groups + group;
      sums[slot] = shared_sum[0];
      squares[slot] = shared_square[0];
    }
  }
}

static __global__ void GroupNormStatsKernel(
    const float* sums, const float* squares, float* means, float* inverses,
    std::uint32_t chunks, std::uint32_t groups, float count, float epsilon) {
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t group = blockIdx.x;
  float sum = 0.0F;
  float square = 0.0F;
  for (std::uint32_t chunk = lane; chunk < chunks; chunk += 32U) {
    const std::size_t slot = static_cast<std::size_t>(chunk) * groups + group;
    sum += sums[slot];
    square += squares[slot];
  }
#pragma unroll
  for (std::uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    sum += __shfl_down(sum, offset, 32);
    square += __shfl_down(square, offset, 32);
  }
  if (lane == 0U) {
    const float mean = sum / count;
    const float variance = fmaxf(square / count - mean * mean, 0.0F);
    means[group] = mean;
    inverses[group] = rsqrtf(variance + epsilon);
  }
}

static __global__ void GroupNormSiluKernel(
    const float* input, const float* weight, const float* bias,
    const float* means, const float* inverses, float4* output,
    std::uint32_t positions, std::uint32_t vectors, std::uint32_t per_group) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t position = blockIdx.y;
  if (position >= positions || column >= vectors) {
    return;
  }
  const std::size_t index =
      static_cast<std::size_t>(position) * vectors + column;
  const float4 value = reinterpret_cast<const float4*>(input)[index];
  const std::size_t channel = static_cast<std::size_t>(column) * 4U;
  const float4 scale = reinterpret_cast<const float4*>(weight)[column];
  const float4 shift = reinterpret_cast<const float4*>(bias)[column];
  const float first = (value.x - means[channel / per_group]) *
                          inverses[channel / per_group] * scale.x +
                      shift.x;
  const float second = (value.y - means[(channel + 1U) / per_group]) *
                           inverses[(channel + 1U) / per_group] * scale.y +
                       shift.y;
  const float third = (value.z - means[(channel + 2U) / per_group]) *
                          inverses[(channel + 2U) / per_group] * scale.z +
                      shift.z;
  const float fourth = (value.w - means[(channel + 3U) / per_group]) *
                           inverses[(channel + 3U) / per_group] * scale.w +
                       shift.w;
  output[index] = {
      first / (1.0F + expf(-first)), second / (1.0F + expf(-second)),
      third / (1.0F + expf(-third)), fourth / (1.0F + expf(-fourth))};
}

static __global__ void GroupNormSiluScalarKernel(
    const float* input, const float* weight, const float* bias,
    const float* means, const float* inverses, float* output,
    std::uint32_t positions, std::uint32_t channels, std::uint32_t per_group) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(positions) * channels) {
    return;
  }
  const std::uint32_t channel = static_cast<std::uint32_t>(index % channels);
  const float normalized = (input[index] - means[channel / per_group]) *
                               inverses[channel / per_group] * weight[channel] +
                           bias[channel];
  output[index] = normalized / (1.0F + expf(-normalized));
}

inline void LaunchGroupNormSilu(const float* input, const float* weight,
                                const float* bias, float* sums, float* squares,
                                float* means, float* inverses, float* output,
                                std::uint32_t positions, std::uint32_t channels,
                                std::uint32_t groups, float epsilon,
                                hipStream_t stream) {
  constexpr std::uint32_t kPositionsPerBlock = 32;
  constexpr std::uint32_t kThreads = 256;
  const std::uint32_t chunks =
      (positions + kPositionsPerBlock - 1U) / kPositionsPerBlock;
  hipLaunchKernelGGL(GroupNormPartialKernel, dim3(chunks, groups),
                     dim3(kThreads), 0, stream, input, sums, squares, positions,
                     channels, groups, kPositionsPerBlock);
  const float count = static_cast<float>(static_cast<std::size_t>(positions) *
                                         (channels / groups));
  hipLaunchKernelGGL(GroupNormStatsKernel, dim3(groups), dim3(32), 0, stream,
                     sums, squares, means, inverses, chunks, groups, count,
                     epsilon);
  if (channels % 4U == 0U) {
    const std::uint32_t vectors = channels / 4U;
    hipLaunchKernelGGL(GroupNormSiluKernel,
                       dim3((vectors + kThreads - 1U) / kThreads, positions),
                       dim3(kThreads), 0, stream, input, weight, bias, means,
                       inverses, reinterpret_cast<float4*>(output), positions,
                       vectors, channels / groups);
    return;
  }
  const std::size_t elements = static_cast<std::size_t>(positions) * channels;
  hipLaunchKernelGGL(
      GroupNormSiluScalarKernel,
      dim3(static_cast<std::uint32_t>((elements + kThreads - 1U) / kThreads)),
      dim3(kThreads), 0, stream, input, weight, bias, means, inverses, output,
      positions, channels, channels / groups);
}

// Residual add: the block output accumulates either the block input or the
// 1x1 nin_shortcut projection of it (vae_cnn.py:ResnetBlock3D.forward).
static __global__ void AddInPlaceKernel(float* destination, const float* source,
                                        std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    destination[index] += source[index];
  }
}

static __global__ void AddInPlaceFloat4Kernel(float4* destination,
                                              const float4* source,
                                              std::uint32_t vectors) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= vectors) {
    return;
  }
  const float4 left = destination[index];
  const float4 right = source[index];
  destination[index] = {left.x + right.x, left.y + right.y, left.z + right.z,
                        left.w + right.w};
}

inline void LaunchAddInPlace(float* destination, const float* source,
                             std::size_t elements, hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  if (elements % 4U == 0U) {
    const auto vectors = static_cast<std::uint32_t>(elements / 4U);
    hipLaunchKernelGGL(
        AddInPlaceFloat4Kernel,
        dim3(static_cast<std::uint32_t>((vectors + kThreads - 1U) / kThreads)),
        dim3(kThreads), 0, stream, reinterpret_cast<float4*>(destination),
        reinterpret_cast<const float4*>(source), vectors);
    return;
  }
  hipLaunchKernelGGL(
      AddInPlaceKernel,
      dim3(static_cast<std::uint32_t>((elements + kThreads - 1U) / kThreads)),
      dim3(kThreads), 0, stream, destination, source, elements);
}

}  // namespace gufo::minimax_h3::video_vae_encoder_ops

#endif  // GUFO_MODELS_MINIMAX_H3_VIDEO_VAE_ENCODER_OPS_HIP_HPP_
