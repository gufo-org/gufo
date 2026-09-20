#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>

#include "src/models/qwen_image_21/hip/bf16.hpp"
#include "src/models/qwen_image_21/hip/runtime.hpp"

namespace gufo::models::qwen_image_21::hip {
namespace {
// Read pairs of adjacent NHWC channels, then transpose in LDS into the trained
// convolution's [channel][kernel_y][kernel_x] order. Each block covers four
// output pixels and 128 input channels. This changes only byte movement.
template<int Stride, int Upscale>
__global__ void TiledIm2col(const std::uint16_t* x, std::uint16_t* y,
                            int height, int width, int channels, int padding,
                            int out_width, int start, int count) {
  __shared__ __align__(16) std::uint16_t tile[4][128 * 9];
  int tid = threadIdx.x, pr = tid / 64,
      c = int(blockIdx.y) * 128 + (tid % 64) * 2,
      pixel = start + int(blockIdx.x) * 4 + pr;
  int py = pixel / out_width * Stride - padding,
      px = pixel % out_width * Stride - padding;
#pragma unroll
  for (int k = 0; k < 9; k++) {
    int sy = py + k / 3, sx = px + k % 3;
    unsigned pair = 0;
    if (int(blockIdx.x) * 4 + pr < count && c < channels && sy >= 0 &&
        sx >= 0 && sy < height * Upscale && sx < width * Upscale)
      pair = *reinterpret_cast<const unsigned*>(
          x + (std::size_t(sy / Upscale) * width + sx / Upscale) * channels +
          c);
    tile[pr][(tid % 64) * 18 + k] = pair & 65535;
    tile[pr][(tid % 64) * 18 + 9 + k] = pair >> 16;
  }
  __syncthreads();
#pragma unroll
  for (int flat = tid * 8; flat < 4 * 128 * 9; flat += 256 * 8) {
    int p = flat / (128 * 9), i = flat % (128 * 9);
    int valid = min(128, channels - int(blockIdx.y) * 128) * 9;
    if (int(blockIdx.x) * 4 + p < count && i < valid)
      *reinterpret_cast<uint4*>(
          y + std::size_t(int(blockIdx.x) * 4 + p) * channels * 9 +
          int(blockIdx.y) * 128 * 9 + i) =
          *reinterpret_cast<const uint4*>(&tile[p][i]);
  }
}
// Keep the im2col tile in LDS and consume packed weights with WMMA.
// Preserve channel/tap order and BF16 product rounding before bias.
using B = hip_bfloat16;
using Frag = __attribute__((ext_vector_type(16))) unsigned short;
using Acc = __attribute__((ext_vector_type(8))) float;
__device__ Frag Load(const unsigned short* p) {
  union {
    Frag f;
    uint4 v[2];
  } u;
  u.v[0] = *reinterpret_cast<const uint4*>(p);
  u.v[1] = *reinterpret_cast<const uint4*>(p + 8);
  return u.f;
}
template<int M, int N, int C, int WM, int WN, int Stride, int Upscale,
         bool Halo = false>
__launch_bounds__(WM * WN * 32) __global__
    void ImplicitConvolution3(const unsigned short* x, const unsigned short* w,
                              const B* bias, B* out, int height, int width,
                              int cin, int cout, int padding, int out_height,
                              int out_width) {
  constexpr int T = WM * WN * 32, WR = M / (16 * WM), WT = N / (16 * WN);
  __shared__ __align__(16) unsigned short tile[N][C * 9 + 8];
  int tid = threadIdx.x, lane = tid % 32, wave = tid / 32, sl = lane % 16,
      wr = wave / WN, wt = wave % WN;
  int pixel0 = blockIdx.y * N, channel0 = blockIdx.x * M;
  Acc acc[WR][WT]{};
  for (int cbase = 0; cbase < cin; cbase += C) {
    // Aligned strips share most of their 3x3 input windows. Read the three
    // source rows once, then form the original channel/tap layout in LDS.
    // The caller guarantees this output strip stays within one image row.
    __shared__ __align__(
        16) unsigned short halo[Halo ? 3 : 1][Halo ? N + 2 : 1][Halo ? C : 1];
    if constexpr (Halo) {
      static_assert(Stride == 1);
      for (int i = tid * 4; i < 3 * (N + 2) * C; i += T * 4) {
        const int c = i % C, p = i / C;
        const int iy = pixel0 / out_width - padding + p / (N + 2);
        const int ix = pixel0 % out_width - padding + p % (N + 2);
        uint2 value{};
        if (iy >= 0 && iy < height * Upscale && ix >= 0 &&
            ix < width * Upscale && cbase + c < cin)
          value = *reinterpret_cast<const uint2*>(
              x + (size_t(iy / Upscale) * width + ix / Upscale) * cin + cbase +
              c);
        *reinterpret_cast<uint2*>(&halo[p / (N + 2)][p % (N + 2)][c]) = value;
      }
      __syncthreads();
    }
    for (int i = tid * 4; i < N * 9 * C; i += T * 4) {
      int c = i % C, tap = i / C % 9, p = i / (9 * C), pixel = pixel0 + p,
          iy = pixel / out_width * Stride - padding + tap / 3,
          ix = pixel % out_width * Stride - padding + tap % 3;
      union {
        uint2 v;
        unsigned short h[4];
      } value{};
      if constexpr (Halo) {
        value.v =
            *reinterpret_cast<const uint2*>(&halo[tap / 3][p + tap % 3][c]);
      } else {
        if (pixel < out_height * out_width && iy >= 0 &&
            iy < height * Upscale && ix >= 0 && ix < width * Upscale &&
            cbase + c < cin)
          value.v = *reinterpret_cast<const uint2*>(
              x + (size_t(iy / Upscale) * width + ix / Upscale) * cin + cbase +
              c);
      }
#pragma unroll
      for (int j = 0; j < 4; j++)
        tile[p][(c + j) * 9 + tap] = value.h[j];
    }
    __syncthreads();
    int count = min(C, cin - cbase) * 9;
    for (int k = 0; k < count; k += 16) {
      Frag a[WR];
#pragma unroll
      for (int i = 0; i < WR; i++) {
        int r = min(cout - 1, channel0 + (wr * WR + i) * 16 + sl);
        a[i] = Load(w + size_t((cbase * 9 + k) / 16) * cout * 16 + r * 16);
      }
#pragma unroll
      for (int j = 0; j < WT; j++) {
        Frag b = Load(&tile[(wt * WT + j) * 16 + sl][k]);
#pragma unroll
        for (int i = 0; i < WR; i++)
          acc[i][j] =
              __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[i], b, acc[i][j]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < WR; i++)
#pragma unroll
    for (int j = 0; j < WT; j++)
#pragma unroll
      for (int z = 0; z < 8; z++) {
        int p = pixel0 + (wt * WT + j) * 16 + sl,
            r = channel0 + (wr * WR + i) * 16 + 2 * z + lane / 16;
        if (p < out_height * out_width && r < cout)
          out[size_t(p) * cout + r] =
              Bf16(float(Bf16(acc[i][j][z])) + (bias ? float(bias[r]) : 0.0F));
      }
}

template<int Stride, int Upscale>
void Convolve(const std::uint16_t* input, const std::uint16_t* weights,
              const B* bias, B* output, int height, int width, int channels,
              int output_channels, int padding, int out_height, int out_width,
              hipStream_t stream) {
  const int rows = out_height * out_width;
  auto launch = [&]<int M, int N, int C, int WM, int WN, bool Halo = false>() {
    ImplicitConvolution3<M, N, C, WM, WN, Stride, Upscale, Halo>
        <<<dim3((output_channels + M - 1) / M, (rows + N - 1) / N),
           WM * WN * 32, 0, stream>>>(input, weights, bias, output, height,
                                      width, channels, output_channels, padding,
                                      out_height, out_width);
  };
  if constexpr (Stride == 1) {
    if (out_width % 64 == 0) {
      if (output_channels == 144) {
        launch.template operator()<144, 64, 16, 3, 4, true>();
        return;
      }
      if (output_channels == 288) {
        launch.template operator()<288, 64, 16, 6, 2, true>();
        return;
      }
      if (output_channels == 576) {
        launch.template operator()<576, 64, 16, 6, 2, true>();
        return;
      }
    }
  }
  if (output_channels == 1152)
    launch.template operator()<576, 64, 16, 6, 2>();
  else if (output_channels == 576)
    launch.template operator()<576, 32, 32, 6, 2>();
  else if (output_channels == 288)
    launch.template operator()<288, 32, 32, 3, 2>();
  else if (output_channels == 144)
    launch.template operator()<144, 32, 32, 3, 2>();
  else if (output_channels == 96)
    launch.template operator()<96, 32, 32, 3, 2>();
  else
    launch.template operator()<192, 32, 32, 3, 2>();
}
}  // namespace

bool Convolve3Shape(int channels, int output_channels) {
  return (channels >= 96 && channels <= 384 && channels % 16 == 0 &&
          output_channels >= 96 && output_channels <= 384 &&
          output_channels % 16 == 0) ||
         ((channels == 576 || channels == 1152) &&
          (output_channels == 288 || output_channels == 576 ||
           output_channels == 1152));
}

bool Convolve3(const std::uint16_t* input, const std::uint16_t* weights,
               const std::uint16_t* bias, std::uint16_t* output, int height,
               int width, int channels, int output_channels, hipStream_t stream,
               int stride, int padding, int upscale) {
  if (!Convolve3Shape(channels, output_channels) || height < 1 || width < 1 ||
      (padding != 1 && padding != 0 && padding != -1) ||
      !((stride == 1 && (upscale == 1 || upscale == 2)) ||
        (stride == 2 && upscale == 1)))
    return false;
  const int total_padding = padding == -1 ? 1 : 2 * padding;
  const int out_height = (height * upscale + total_padding - 3) / stride + 1;
  const int out_width = (width * upscale + total_padding - 3) / stride + 1;
  if (out_height < 1 || out_width < 1)
    return false;
  const auto b = reinterpret_cast<const B*>(bias);
  const auto y = reinterpret_cast<B*>(output);
  if (stride == 2)
    Convolve<2, 1>(input, weights, b, y, height, width, channels,
                   output_channels, std::max(0, padding), out_height, out_width,
                   stream);
  else if (upscale == 2)
    Convolve<1, 2>(input, weights, b, y, height, width, channels,
                   output_channels, std::max(0, padding), out_height, out_width,
                   stream);
  else
    Convolve<1, 1>(input, weights, b, y, height, width, channels,
                   output_channels, std::max(0, padding), out_height, out_width,
                   stream);
  return true;
}

bool ExpandConvolution3(const std::uint16_t* input, std::uint16_t* output,
                        int height, int width, int channels, int stride,
                        int padding, int upscale, int output_width, int start,
                        int count, hipStream_t stream) {
  if (channels < 8 || channels % 8 ||
      !((stride == 1 && (upscale == 1 || upscale == 2)) ||
        (stride == 2 && upscale == 1)))
    return false;
  const dim3 grid((count + 3) / 4, (channels + 127) / 128);
  if (stride == 2)
    TiledIm2col<2, 1><<<grid, 256, 0, stream>>>(input, output, height, width,
                                                channels, padding, output_width,
                                                start, count);
  else if (upscale == 2)
    TiledIm2col<1, 2><<<grid, 256, 0, stream>>>(input, output, height, width,
                                                channels, padding, output_width,
                                                start, count);
  else
    TiledIm2col<1, 1><<<grid, 256, 0, stream>>>(input, output, height, width,
                                                channels, padding, output_width,
                                                start, count);
  return true;
}
}  // namespace gufo::models::qwen_image_21::hip
