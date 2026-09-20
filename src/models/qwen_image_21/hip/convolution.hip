#include <hip/hip_runtime.h>

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
}  // namespace

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
