#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <stdexcept>

#include "src/models/qwen3_asr/hip/audio_ops.hpp"

namespace gufo::models::qwen3_asr::hip {
namespace {
using B = hip_bfloat16;
using Fragment = __attribute__((ext_vector_type(16))) unsigned short;
using Accumulator = __attribute__((ext_vector_type(8))) float;

__device__ Fragment LoadFragment(const unsigned short* source) {
  union {
    Fragment fragment;
    uint4 words[2];
  } value;
  value.words[0] = *reinterpret_cast<const uint4*>(source);
  value.words[1] = *reinterpret_cast<const uint4*>(source + 8);
  return value.fragment;
}

__global__ void PackWeights(const unsigned short* source,
                            unsigned short* output) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr unsigned kElements = 480 * 480 * 9;
  if (index < kElements) {
    const unsigned k = index / (480 * 16) * 16 + index % 16;
    const unsigned channel = index / 16 % 480;
    output[index] = source[channel * (480 * 9) + k];
  }
}

// NCHW, 3x3, stride two, one zero-padding pixel. Zero-pad the nine
// products to one WMMA instruction, preserving the matrix accumulation used
// by the BF16 checkpoint teacher while sharing each input across channels.
__global__ __launch_bounds__(128) void FirstConvolution(
    const unsigned short* input, const unsigned short* weight, B* output) {
  constexpr int Pixels = 64 * 50, M = 96, N = 64;
  const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  const int channel0 = blockIdx.x * M, pixel0 = blockIdx.y * N;
  input += std::size_t(blockIdx.z) * 128 * 100;
  output += std::size_t(blockIdx.z) * 480 * Pixels;
  Fragment a[3];
#pragma unroll
  for (int m = 0; m < 3; ++m) {
    a[m] = {};
    const int channel = channel0 + (wave / 2 * 3 + m) * 16 + lane % 16;
#pragma unroll
    for (int k = 0; k < 9; ++k)
      a[m][k] = weight[channel * 9 + k];
  }
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    Fragment b{};
    const int pixel = pixel0 + (wave % 2 * 2 + n) * 16 + lane % 16;
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      const int y = pixel / 50 * 2 - 1 + k / 3;
      const int x = pixel % 50 * 2 - 1 + k % 3;
      if (pixel < Pixels && y >= 0 && y < 128 && x >= 0 && x < 100)
        b[k] = input[y * 100 + x];
    }
#pragma unroll
    for (int m = 0; m < 3; ++m) {
      const auto sum =
          __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[m], b, Accumulator{});
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int channel =
            channel0 + (wave / 2 * 3 + m) * 16 + 2 * i + lane / 16;
        if (pixel < Pixels)
          output[channel * Pixels + pixel] = B(sum[i]);
      }
    }
  }
}

// Match the teacher's BF16 GEMM accumulation: 128-pixel tiles rotate the
// channel/tap reduction by 128 products, wrapping at K=4320. A fixed starting
// channel changes rounding in every tile after the first. Weights are packed
// losslessly once; im2col stays in LDS instead of a full device tensor.
template<int Height, int Width, int M, int N, int WM, int WN>
__global__ __launch_bounds__(WM * WN * 32) void Convolution(
    const unsigned short* input, const unsigned short* weight, B* output) {
  constexpr int C = 16, K = C * 9, T = WM * WN * 32;
  constexpr int OH = (Height + 1) / 2, OW = (Width + 1) / 2, Pixels = OH * OW;
  constexpr int MR = M / (16 * WM), NR = N / (16 * WN);
  __shared__ __align__(16) unsigned short tile[N][K + 8];
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int channel0 = blockIdx.x * M, pixel0 = blockIdx.y * N;
  input += std::size_t(blockIdx.z) * 480 * Height * Width;
  output += std::size_t(blockIdx.z) * 480 * Pixels;
  Accumulator acc[MR][NR]{};
  for (int cbase = 0; cbase < 480; cbase += C) {
    for (int i = tid; i < N * K; i += T) {
      const int p = i % N, k = i / N, pixel = pixel0 + p;
      const int reduction = (pixel0 / 128 * 128 + cbase * 9 + k) % (480 * 9);
      const int y = pixel / OW * 2 - 1 + (reduction % 9) / 3;
      const int x = pixel % OW * 2 - 1 + reduction % 3;
      tile[p][k] = pixel < Pixels && y >= 0 && y < Height && x >= 0 && x < Width
                       ? input[(reduction / 9 * Height + y) * Width + x]
                       : 0;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < K; k += 16) {
      Fragment a[MR];
#pragma unroll
      for (int m = 0; m < MR; ++m) {
        const int r = channel0 + (wave / WN * MR + m) * 16 + lane % 16;
        a[m] = LoadFragment(
            weight +
            std::size_t(((pixel0 / 128 * 128 + cbase * 9 + k) % (480 * 9)) /
                        16) *
                480 * 16 +
            r * 16);
      }
#pragma unroll
      for (int n = 0; n < NR; ++n) {
        const auto b =
            LoadFragment(tile[(wave % WN * NR + n) * 16 + lane % 16] + k);
#pragma unroll
        for (int m = 0; m < MR; ++m)
          acc[m][n] =
              __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b, a[m], acc[m][n]);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
  }
#pragma unroll
  for (int m = 0; m < MR; ++m)
#pragma unroll
    for (int n = 0; n < NR; ++n)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int p = pixel0 + (wave % WN * NR + n) * 16 + 2 * i + lane / 16;
        const int r = channel0 + (wave / WN * MR + m) * 16 + lane % 16;
        if (p < Pixels)
          output[r * Pixels + p] = B(acc[m][n][i]);
      }
}
}  // namespace

void LaunchPackConvolutionWeights(const void* input, void* output,
                                  hipStream_t stream) {
  PackWeights<<<(480 * 480 * 9 + 255) / 256, 256, 0, stream>>>(
      static_cast<const unsigned short*>(input),
      static_cast<unsigned short*>(output));
}

void LaunchConvolution3x3(const void* input, const void* weight, void* output,
                          std::size_t chunks, std::size_t input_channels,
                          std::size_t height, std::size_t width,
                          hipStream_t stream) {
  if (input_channels == 1 && height == 128 && width == 100) {
    FirstConvolution<<<dim3(5, (64 * 50 + 63) / 64, chunks), 128, 0, stream>>>(
        static_cast<const unsigned short*>(input),
        static_cast<const unsigned short*>(weight), static_cast<B*>(output));
    return;
  }
  constexpr int M = 480, N = 32, WM = 6, WN = 2;
  const auto launch = [&]<int H, int W>() {
    constexpr int pixels = ((H + 1) / 2) * ((W + 1) / 2);
    Convolution<H, W, M, N, WM, WN>
        <<<dim3(480 / M, (pixels + N - 1) / N, chunks), WM * WN * 32, 0,
           stream>>>(static_cast<const unsigned short*>(input),
                     static_cast<const unsigned short*>(weight),
                     static_cast<B*>(output));
  };
  if (input_channels == 480 && height == 64 && width == 50)
    launch.template operator()<64, 50>();
  else if (input_channels == 480 && height == 32 && width == 25)
    launch.template operator()<32, 25>();
  else
    throw std::invalid_argument("unsupported Qwen3-ASR convolution shape");
}
}  // namespace gufo::models::qwen3_asr::hip
