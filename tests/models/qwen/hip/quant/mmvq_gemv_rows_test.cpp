#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "qfn_mmq.h"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/gemm_route.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

// The Ornith-1.5-122B output projection shape that decode sends to the
// vendored vector kernel.
constexpr std::size_t kM = 3072;
constexpr std::size_t kK = 8192;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(EXIT_FAILURE);
  }
}

void CheckHip(hipError_t status, const char* message) {
  if (status != hipSuccess) {
    std::cerr << "FAIL: " << message << ": " << hipGetErrorString(status)
              << "\n";
    std::exit(EXIT_FAILURE);
  }
}

std::uint16_t HalfBits(float value) {
  const _Float16 half = static_cast<_Float16>(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

// Q4_K rows with random nibbles and 6-bit scales and small block scales.
std::vector<gufo::quant::block_q4_K> MakeWeights(std::mt19937& rng) {
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_real_distribution<float> scale(0.001F, 0.01F);
  std::vector<gufo::quant::block_q4_K> blocks(kM * kK / 256);
  for (auto& block : blocks) {
    block.d = HalfBits(scale(rng));
    block.dmin = HalfBits(scale(rng));
    for (auto& s : block.scales) {
      s = static_cast<std::uint8_t>(byte(rng));
    }
    for (auto& q : block.qs) {
      q = static_cast<std::uint8_t>(byte(rng));
    }
  }
  return blocks;
}

void TestRoute() {
  using gufo::models::qwen::QwenGemmMode;
  using gufo::models::qwen::QwenGemmRoute;
  const auto route = gufo::models::qwen::ResolveQwenGemmRoute(
      {.type = gufo::core::GgmlType::kQ4_K,
       .m = kM,
       .k = kK,
       .mode = QwenGemmMode::kHipDecode});
  Check(route.route == QwenGemmRoute::kHipQuantMmvq,
        "the test shape takes the vector route");
}

// LaunchGEMVRows must give each row the bits LaunchGEMV gives it alone, so
// drafted and verified tokens match plain decode. Rows carry distinct inputs
// to catch row or expert-id indexing errors.
void TestRowsMatchSingleRow(const void* weights,
                            const std::vector<gufo::quant::block_q4_K>& host,
                            std::size_t rows, std::mt19937& rng) {
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> x(rows * kK);
  for (auto& v : x) {
    v = normal(rng);
  }
  float* d_x = nullptr;
  float* d_rows = nullptr;
  float* d_single = nullptr;
  CheckHip(hipMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
  CheckHip(hipMalloc(&d_rows, rows * kM * sizeof(float)), "malloc rows");
  CheckHip(hipMalloc(&d_single, rows * kM * sizeof(float)), "malloc single");
  CheckHip(
      hipMemcpy(d_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice),
      "copy x");

  gufo::hip::LaunchGEMVRows(weights, gufo::core::GgmlType::kQ4_K, d_x, d_rows,
                            rows, kM, kK);
  for (std::size_t row = 0; row < rows; ++row) {
    gufo::hip::LaunchGEMV(weights, gufo::core::GgmlType::kQ4_K, d_x + row * kK,
                          d_single + row * kM, kM, kK);
  }
  CheckHip(hipDeviceSynchronize(), "launch");

  std::vector<float> y_rows(rows * kM);
  std::vector<float> y_single(rows * kM);
  CheckHip(hipMemcpy(y_rows.data(), d_rows, y_rows.size() * sizeof(float),
                     hipMemcpyDeviceToHost),
           "copy rows");
  CheckHip(hipMemcpy(y_single.data(), d_single, y_single.size() * sizeof(float),
                     hipMemcpyDeviceToHost),
           "copy single");
  Check(std::memcmp(y_rows.data(), y_single.data(),
                    y_rows.size() * sizeof(float)) == 0,
        "batched rows are bit-identical to single-row launches");

  // The vector kernel quantizes x to 8 bits, so it tracks the FP32 product
  // closely but not exactly.
  const std::size_t row_bytes = kK / 256 * sizeof(gufo::quant::block_q4_K);
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(host.data());
  for (std::size_t row = 0; row < rows; ++row) {
    double err = 0.0;
    double norm = 0.0;
    for (std::size_t m = 0; m < kM; ++m) {
      const double want = gufo::quant::DotProductQ4_K(
          bytes + m * row_bytes,
          std::span<const float>(x.data() + row * kK, kK), kK);
      const double got = y_rows[row * kM + m];
      Check(std::isfinite(got), "finite output");
      err += (got - want) * (got - want);
      norm += want * want;
    }
    Check(std::sqrt(err / norm) < 2e-2, "vector route tracks the FP32 product");
  }

  CheckHip(hipFree(d_x), "free x");
  CheckHip(hipFree(d_rows), "free rows");
  CheckHip(hipFree(d_single), "free single");
}

}  // namespace
#endif

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "Qwen MMVQ GEMV rows test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }
  Check(qfn_mmq_init(0) == 0, "qfn_mmq_init");
  TestRoute();

  std::mt19937 rng(829);
  const auto host = MakeWeights(rng);
  void* weights = nullptr;
  const std::size_t weight_bytes = host.size() * sizeof(host[0]);
  CheckHip(hipMalloc(&weights, weight_bytes), "malloc weights");
  CheckHip(hipMemcpy(weights, host.data(), weight_bytes, hipMemcpyHostToDevice),
           "copy weights");
  // 65 rows spans two vector launches.
  for (const std::size_t rows : {1U, 2U, 3U, 8U, 65U}) {
    TestRowsMatchSingleRow(weights, host, rows, rng);
  }
  CheckHip(hipFree(weights), "free weights");
  std::cout << "Qwen MMVQ GEMV rows tests passed.\n";
#endif
  return 0;
}
