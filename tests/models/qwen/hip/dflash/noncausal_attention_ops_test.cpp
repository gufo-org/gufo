#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/kernels/dflash_kernels.hpp"
#include "tests/models/qwen/hip/support/comparisons.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

/// The shipped DFlash-2 topology: sixteen query heads over four key/value
/// heads at head_dim 256 (`decoder.Qcur` is 8x4096 and `decoder.Kcur` 8x1024
/// under `GUFO_DFLASH_DEBUG`), an eight-slot draft block and the 2,048-token
/// draft window.
constexpr std::uint32_t kNumQueryHeads = 16;
constexpr std::uint32_t kNumKvHeads = 4;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kBlockCount = 8;
constexpr std::uint32_t kSlidingWindow = 2048;

float Sample(std::size_t index, std::size_t salt) {
  return std::sin(static_cast<float>((index * 7U) + (salt * 13U) + 1U)) * 0.5F;
}

enum class Route { kWave, kGqa };

/// Runs the reference and one candidate route over the same operands and
/// reports the largest absolute difference. The kernels accumulate the QK dot
/// product in a different order -- the reference walks head_dim in one thread,
/// the candidates reduce across a wave -- so the comparison is a float
/// tolerance, not bit equality.
float MaxRouteDifference(Route route, std::uint32_t current_pos,
                         std::uint32_t history_length) {
  const std::size_t q_dim = static_cast<std::size_t>(kNumQueryHeads) * kHeadDim;
  const std::size_t kv_dim = static_cast<std::size_t>(kNumKvHeads) * kHeadDim;
  const std::size_t q_elements = kBlockCount * q_dim;
  const std::size_t block_elements = kBlockCount * kv_dim;
  const std::size_t history_elements =
      static_cast<std::size_t>(current_pos + kBlockCount) * kv_dim;

  std::vector<float> h_q(q_elements);
  std::vector<float> h_block_k(block_elements);
  std::vector<float> h_block_v(block_elements);
  std::vector<float> h_injected_k(history_elements);
  std::vector<float> h_injected_v(history_elements);
  for (std::size_t index = 0; index < q_elements; ++index) {
    h_q[index] = Sample(index, 1);
  }
  for (std::size_t index = 0; index < block_elements; ++index) {
    h_block_k[index] = Sample(index, 2);
    h_block_v[index] = Sample(index, 3);
  }
  for (std::size_t index = 0; index < history_elements; ++index) {
    h_injected_k[index] = Sample(index, 4);
    h_injected_v[index] = Sample(index, 5);
  }

  const auto upload = [](const std::vector<float>& host) {
    float* device = nullptr;
    HIP_CHECK(hipMalloc(&device, host.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(device, host.data(), host.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    return device;
  };

  float* d_q = upload(h_q);
  float* d_block_k = upload(h_block_k);
  float* d_block_v = upload(h_block_v);
  float* d_injected_k = upload(h_injected_k);
  float* d_injected_v = upload(h_injected_v);
  float* d_scalar_out = nullptr;
  float* d_wave_out = nullptr;
  HIP_CHECK(hipMalloc(&d_scalar_out, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_wave_out, q_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_scalar_out, 0, q_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_wave_out, 0, q_elements * sizeof(float)));

  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  gufo::hip::kernels::LaunchDFlashNonCausalAttentionScalar(
      d_q, d_injected_k, d_injected_v, d_block_k, d_block_v, d_scalar_out,
      current_pos, history_length, kBlockCount, kSlidingWindow, kNumQueryHeads,
      kNumKvHeads, kHeadDim, scale, nullptr);
  if (route == Route::kGqa) {
    gufo::test::Expect(
        gufo::hip::kernels::DFlashNonCausalAttentionGqaSupported(
            current_pos, history_length, kBlockCount, kSlidingWindow,
            kNumQueryHeads, kNumKvHeads, kHeadDim),
        "the shipped DFlash-2 shape must support the group-query route");
    gufo::hip::kernels::LaunchDFlashNonCausalAttentionGqa(
        d_q, d_injected_k, d_injected_v, d_block_k, d_block_v, d_wave_out,
        current_pos, history_length, kBlockCount, kSlidingWindow,
        kNumQueryHeads, kNumKvHeads, kHeadDim, scale, nullptr);
  } else {
    gufo::hip::kernels::LaunchDFlashNonCausalAttentionWave(
        d_q, d_injected_k, d_injected_v, d_block_k, d_block_v, d_wave_out,
        current_pos, history_length, kBlockCount, kSlidingWindow,
        kNumQueryHeads, kNumKvHeads, kHeadDim, scale, nullptr);
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> scalar_out(q_elements);
  std::vector<float> wave_out(q_elements);
  HIP_CHECK(hipMemcpy(scalar_out.data(), d_scalar_out,
                      q_elements * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(wave_out.data(), d_wave_out, q_elements * sizeof(float),
                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_block_k));
  HIP_CHECK(hipFree(d_block_v));
  HIP_CHECK(hipFree(d_injected_k));
  HIP_CHECK(hipFree(d_injected_v));
  HIP_CHECK(hipFree(d_scalar_out));
  HIP_CHECK(hipFree(d_wave_out));

  float max_difference = 0.0F;
  for (std::size_t index = 0; index < q_elements; ++index) {
    gufo::test::Expect(std::isfinite(wave_out[index]),
                       "DFlash candidate attention produced a non-finite "
                       "output");
    max_difference =
        std::max(max_difference, std::abs(scalar_out[index] - wave_out[index]));
  }
  return max_difference;
}

void TestRouteEquivalence(std::uint32_t current_pos,
                          std::uint32_t history_length, const char* label) {
  const float wave =
      MaxRouteDifference(Route::kWave, current_pos, history_length);
  const float gqa =
      MaxRouteDifference(Route::kGqa, current_pos, history_length);
  std::cout << "DFlash non-causal attention " << label
            << ": max |scalar - wave| = " << wave
            << ", |scalar - gqa| = " << gqa << "\n";
  gufo::test::Expect(wave < 1e-5F,
                     "DFlash wave attention diverges from the reference");
  gufo::test::Expect(gqa < 1e-5F,
                     "DFlash group-query attention diverges from the "
                     "reference");
}

}  // namespace

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen DFlash non-causal attention ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }

  // No injected history at all: only the eight in-block keys are attended.
  TestRouteEquivalence(0, 0, "empty history");
  // Shallow: the whole history is inside the sliding window.
  TestRouteEquivalence(128, 128, "128-token history");
  // Deep enough that the window clips the history, which is the case the
  // coalesced route exists for.
  TestRouteEquivalence(4096, 4096, "4096-token history, window-clipped");
  std::cout << "Qwen DFlash non-causal attention ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen DFlash non-causal attention ops "
               "test.\n";
  return 77;
#endif
}
