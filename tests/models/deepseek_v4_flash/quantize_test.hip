#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "src/models/deepseek_v4_flash/kernels/rocm/mmq/quantize.hip.hpp"

namespace {
void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(error));
    std::exit(1);
  }
}

void test_quantize(ggml_type type, int rows) {
  constexpr int width = 512;
  constexpr int block_width = 4 * QK8_1;
  std::vector<float> input(rows * width);
  for (int row = 0; row < rows; ++row) {
    if (row % 4 == 0)
      continue;  // Include an entirely empty activation row.
    for (int column = 0; column < width; column += QK8_1) {
      const float scale =
          std::ldexp(1.0F, (row + column / block_width) % 17 - 8);
      // Captured first-layer activations: rounding the FP32 product before
      // inspecting its fraction incorrectly produces 34 and -39.
      input[row * width + column] = 0.05716175213456154F * scale;
      input[row * width + column + 1] = -input[row * width + column];
      input[row * width + column + 2] = 0.015078099444508553F * scale;
      input[row * width + column + 3] = -0.01732856221497059F * scale;
      float positive = input[row * width + column + 2];
      float negative = input[row * width + column + 3];
      for (int step = 0; step < 4; ++step) {
        positive = std::nextafter(positive, INFINITY);
        negative = std::nextafter(negative, -INFINITY);
      }
      input[row * width + column + 4] = positive;
      input[row * width + column + 5] = negative;
      input[row * width + column + 6] = -0.0F;
    }
  }
  std::vector<block_q8_1_mmq> output(rows * width / block_width);
  float* device_input = nullptr;
  block_q8_1_mmq* device_output = nullptr;
  check(hipMalloc(&device_input, input.size() * sizeof(float)));
  check(hipMalloc(&device_output, output.size() * sizeof(block_q8_1_mmq)));
  check(hipMemcpy(device_input, input.data(), input.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  check(hipMemset(device_output, 0, output.size() * sizeof(block_q8_1_mmq)));
  ds4_quantize_mmq_q8_1_hip(device_input, nullptr, device_output, type, width,
                            width, 0, 0, width, rows, 1, 1, nullptr);
  check(hipGetLastError());
  check(hipMemcpy(output.data(), device_output,
                  output.size() * sizeof(block_q8_1_mmq),
                  hipMemcpyDeviceToHost));
  const int scale_width =
      mmq_get_q8_1_ds_layout(type) == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < width; ++column) {
      const int first = column / scale_width * scale_width;
      double maximum = 0;
      for (int i = first; i < first + scale_width; ++i)
        maximum = std::max(maximum, std::abs(double(input[row * width + i])));
      // Independent FP64 ratio and integer rounding; no production helper.
      const int expected =
          maximum == 0 ? 0
                       : int(std::round(double(input[row * width + column]) *
                                        127.0 / maximum));
      const int actual =
          output[column / block_width * rows + row].qs[column % block_width];
      if (actual != expected) {
        std::fprintf(stderr,
                     "MMQ quantization type=%d rows=%d row=%d column=%d: "
                     "%d != %d\n",
                     int(type), rows, row, column, actual, expected);
        std::exit(2);
      }
    }
  }
  check(hipFree(device_output));
  check(hipFree(device_input));
}

// One thread per byte pair. Each byte lane holds a different pair, so a carry
// or borrow leaking into a neighbouring lane also shows up.
__global__ void byte_intrinsics(unsigned* failures) {
  const unsigned pair = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned x = pair >> 8, y = pair & 255u;
  const unsigned a = x | y << 8 | (x ^ 0x80u) << 16 | (255u - y) << 24;
  const unsigned b = y | x << 8 | (y ^ 0x7fu) << 16 | (255u - x) << 24;
  const unsigned ne = __vcmpne4(a, b);
  const unsigned sub = unsigned(__vsub4(int(a), int(b)));
  for (unsigned shift = 0; shift < 32; shift += 8) {
    const unsigned ab = a >> shift & 255u, bb = b >> shift & 255u;
    if ((ne >> shift & 255u) != (ab != bb ? 255u : 0u) ||
        (sub >> shift & 255u) != ((ab - bb) & 255u))
      atomicAdd(failures, 1u);
  }
}

// The IQ2/IQ3 loaders apply signs with __vsub4(grid ^ mask, mask) on bytes
// built by __vcmpne4; both must match CUDA's per-byte definitions.
void test_byte_intrinsics() {
  unsigned* device_failures = nullptr;
  check(hipMalloc(&device_failures, sizeof(unsigned)));
  check(hipMemset(device_failures, 0, sizeof(unsigned)));
  byte_intrinsics<<<256, 256>>>(device_failures);
  check(hipGetLastError());
  unsigned failures = 0;
  check(hipMemcpy(&failures, device_failures, sizeof(unsigned),
                  hipMemcpyDeviceToHost));
  check(hipFree(device_failures));
  if (failures != 0) {
    std::fprintf(stderr, "__vcmpne4/__vsub4: %u byte mismatches\n", failures);
    std::exit(2);
  }
}
}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
    return 77;
  for (ggml_type type : {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q2_K})
    for (int rows : {1, 4, 32, 129})
      test_quantize(type, rows);
  std::puts("MMQ quantization rounding and zero/tail rows pass");
  test_byte_intrinsics();
  std::puts("__vcmpne4/__vsub4: all byte pairs exact");
}
