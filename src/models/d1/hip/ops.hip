#include <cstddef>

#include "src/models/d1/hip/ops.hpp"
namespace gufo::models::d1::gpu {
namespace {
__device__ int Index(int y, int x, int width) {
  return ((y / 2) * (width / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2;
}
__global__ void Conv(const float* p, const float* w, float* out, int h,
                     int width, int hidden) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h * width * hidden)
    return;
  const int c = i % hidden, row = i / hidden;
  const int y = (row / 4) / (width / 2) * 2 + row % 4 / 2;
  const int x = (row / 4) % (width / 2) * 2 + row % 2;
  float sum = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
      if (y + dy >= 0 && y + dy < h && x + dx >= 0 && x + dx < width) {
        const auto at =
            std::size_t(Index(y + dy, x + dx, width)) * hidden * 3 + c;
        const float gated = p[at] * p[at + 2 * hidden];
        sum = fmaf(gated, w[c * 9 + (dy + 1) * 3 + dx + 1], sum);
      }
  out[i] = p[std::size_t(row) * 3 * hidden + hidden + c] * sum;
}
__global__ void Rope(float* x, int h, int w, int heads) {
  constexpr int dim = 64, half = 32;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h * w * heads * half)
    return;
  const int d = i % half, row = i / (heads * half);
  const int y = (row / 4) / (w / 2) * 2 + row % 4 / 2;
  const int col = (row / 4) % (w / 2) * 2 + row % 2;
  const float angle = (d < 16 ? y : col) * powf(1e6F, -float(d % 16) / 16);
  float sn, cs;
  sincosf(angle, &sn, &cs);
  const int at = (i / half) * dim + d;
  const float a = x[at], b = x[at + half];
  x[at] = a * cs - b * sn;
  x[at + half] = b * cs + a * sn;
}
}  // namespace
void SpatialConv(const float* p, const float* w, float* out, int h, int width,
                 int hidden, hipStream_t s) {
  Conv<<<(std::size_t(h) * width * hidden + 255) / 256, 256, 0, s>>>(
      p, w, out, h, width, hidden);
}
void AxialRope(float* x, int h, int w, int heads, hipStream_t s) {
  Rope<<<(std::size_t(h) * w * heads * 32 + 255) / 256, 256, 0, s>>>(x, h, w,
                                                                     heads);
}
}  // namespace gufo::models::d1::gpu
