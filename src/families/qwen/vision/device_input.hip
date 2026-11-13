#include <stdexcept>

#include "src/families/qwen/vision/device_input.hpp"

namespace gufo::models::qwen::vision {
namespace {
__global__ void InjectRowsKernel(float* hidden, const float* embedding,
                                 std::uint32_t rows, std::uint32_t width,
                                 std::uint32_t hc) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * width * hc)
    return;
  const auto row = i / (width * hc);
  hidden[i] = embedding[row * width + i % width];
}
}  // namespace

void DeviceInput::LaunchRows(float* hidden, const float* embedding,
                             std::uint32_t rows, std::uint32_t width,
                             std::uint32_t hc, hipStream_t stream) {
  InjectRowsKernel<<<(rows * width * hc + 255) / 256, 256, 0, stream>>>(
      hidden, embedding, rows, width, hc);
  const auto result = hipGetLastError();
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}
}  // namespace gufo::models::qwen::vision
