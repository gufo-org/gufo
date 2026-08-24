#include "src/models/qwen/gemm_route.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {

using strix::core::GgmlType;
using strix::models::qwen::DescribeQwenGemmFormat;
using strix::models::qwen::QwenGemmCapabilities;
using strix::models::qwen::QwenGemmMode;
using strix::models::qwen::QwenGemmRejection;
using strix::models::qwen::QwenGemmRequest;
using strix::models::qwen::QwenGemmRoute;
using strix::models::qwen::ResolveQwenGemmRoute;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "Qwen GEMM route test failure: " << message << '\n';
    std::abort();
  }
}

QwenGemmRoute Route(GgmlType type, std::size_t m, std::size_t k,
                    QwenGemmMode mode,
                    QwenGemmCapabilities capabilities = {},
                    bool residual_epilogue = false,
                    std::size_t batch_size = 1) {
  return ResolveQwenGemmRoute({.type = type,
                               .batch_size = batch_size,
                               .m = m,
                               .k = k,
                               .mode = mode,
                               .capabilities = capabilities,
                               .residual_epilogue = residual_epilogue})
      .route;
}

void TestFormatCapabilities() {
  constexpr std::array all_types{
      GgmlType::kF32,          GgmlType::kF16,
      GgmlType::kQ4_0,         GgmlType::kQ4_1,
      GgmlType::kQ5_0,         GgmlType::kQ5_1,
      GgmlType::kQ8_0,         GgmlType::kQ8_1,
      GgmlType::kQ2_K,         GgmlType::kQ3_K,
      GgmlType::kQ4_K,         GgmlType::kQ5_K,
      GgmlType::kQ6_K,         GgmlType::kQ8_K,
      GgmlType::kIQ2_XXS,      GgmlType::kBF16,
      GgmlType::kStrixSHQ4_T16, GgmlType::kStrixSHQ6_T16,
      GgmlType::kStrixSHQ8_T16,
  };
  for (const auto type : all_types) {
    const auto descriptor = DescribeQwenGemmFormat(type);
    Check(descriptor.dense != descriptor.quantized,
          "every declared GGML type must classify as dense or quantized");
    const std::size_t k = descriptor.block_elements == 0
                              ? 32
                              : descriptor.block_elements;
    const auto cpu = ResolveQwenGemmRoute({.type = type,
                                           .batch_size = 1,
                                           .m = 1,
                                           .k = k,
                                           .mode = QwenGemmMode::kCpu});
    Check(cpu.accepted() == descriptor.cpu_direct,
          "CPU route must match the format capability descriptor");
  }

  Check(DescribeQwenGemmFormat(GgmlType::kQ8_0).block_elements == 32,
        "Q8_0 block geometry");
  Check(DescribeQwenGemmFormat(GgmlType::kQ6_K).block_elements == 256,
        "K-quant block geometry");
  Check(DescribeQwenGemmFormat(GgmlType::kF16).cpu_direct,
        "CPU F16 support");
  Check(!DescribeQwenGemmFormat(GgmlType::kF16).hip_decode_direct,
        "HIP F16 rejection");
  Check(DescribeQwenGemmFormat(GgmlType::kQ5_K).hip_prefill_direct,
        "HIP prefill K-quant support");
  Check(!DescribeQwenGemmFormat(GgmlType::kQ4_K).hip_prefill_direct,
        "HIP prefill Q4_K rejection");
}

void TestCpuRoutes() {
  Check(Route(GgmlType::kF32, 31, 31, QwenGemmMode::kCpu) ==
            QwenGemmRoute::kCpuF32Rows,
        "CPU F32 route");
  Check(Route(GgmlType::kBF16, 32, 32, QwenGemmMode::kCpu) ==
            QwenGemmRoute::kCpuBf16Rows,
        "CPU BF16 route");
  Check(Route(GgmlType::kF16, 255, 255, QwenGemmMode::kCpu) ==
            QwenGemmRoute::kCpuF16Rows,
        "CPU F16 route");
  Check(Route(GgmlType::kQ8_0, 1, 32, QwenGemmMode::kCpu) ==
            QwenGemmRoute::kCpuQuantDot,
        "CPU Q8_0 route");
  Check(Route(GgmlType::kQ3_K, 1, 256, QwenGemmMode::kCpu) ==
            QwenGemmRoute::kCpuQuantDot,
        "CPU K-quant route");
}

void TestDecodeAndMtpRoutes() {
  Check(Route(GgmlType::kBF16, 1, 31, QwenGemmMode::kHipDecode) ==
            QwenGemmRoute::kHipBf16Baseline256,
        "BF16 unaligned K baseline");
  Check(Route(GgmlType::kBF16, 1, 32, QwenGemmMode::kHipDecode) ==
            QwenGemmRoute::kHipBf16Wave32Single,
        "BF16 aligned K wave32");
  Check(Route(GgmlType::kBF16, 255, 4096, QwenGemmMode::kHipDecode) ==
            QwenGemmRoute::kHipBf16Wave32Single,
        "BF16 M=255 remains wave32 single-row");
  Check(Route(GgmlType::kBF16, 256, 4096, QwenGemmMode::kHipDecode) ==
            QwenGemmRoute::kHipBf16Wave32Single,
        "BF16 M=256 remains wave32 single-row");
  Check(Route(GgmlType::kBF16, 1, 8191, QwenGemmMode::kHipDecode) ==
            QwenGemmRoute::kHipBf16Baseline256,
        "BF16 K=8191 baseline");
  Check(Route(GgmlType::kBF16, 1, 8192, QwenGemmMode::kHipDecode) ==
            QwenGemmRoute::kHipBf16Baseline256,
        "BF16 K=8192 baseline");
  Check(Route(GgmlType::kBF16, 1, 16383, QwenGemmMode::kHipDecode) ==
            QwenGemmRoute::kHipBf16Baseline256,
        "BF16 K=16383 baseline256");
  Check(Route(GgmlType::kBF16, 1, 16384, QwenGemmMode::kHipDecode) ==
            QwenGemmRoute::kHipBf16Baseline512,
        "BF16 K=16384 baseline512");
  Check(Route(GgmlType::kF32, 1, 16383, QwenGemmMode::kHipDecode) ==
            QwenGemmRoute::kHipF32Baseline256,
        "F32 K=16383 baseline256");
  Check(Route(GgmlType::kF32, 1, 16384, QwenGemmMode::kHipDecode) ==
            QwenGemmRoute::kHipF32Baseline512,
        "F32 K=16384 baseline512");
  Check(Route(GgmlType::kQ8_0, 1, 32, QwenGemmMode::kHipDecode) ==
            QwenGemmRoute::kHipQuantDirect,
        "decode Q8_0 direct route");
  Check(Route(GgmlType::kQ6_K, 1, 256, QwenGemmMode::kHipMtp) ==
            QwenGemmRoute::kHipQuantDirect,
        "MTP quantized output route");
  Check(Route(GgmlType::kBF16, 256, 4096, QwenGemmMode::kHipMtp) ==
            QwenGemmRoute::kHipBf16Wave32Single,
        "MTP packed BF16 route");
}

void TestPrefillRoutes() {
  const QwenGemmCapabilities lt{.can_try_hipblaslt = true};
  Check(Route(GgmlType::kBF16, 1023, 1024, QwenGemmMode::kHipPrefill,
              lt) == QwenGemmRoute::kHipPrefillBf16Blas,
        "prefill M=1023 remains BLAS");
  Check(Route(GgmlType::kBF16, 1024, 1023, QwenGemmMode::kHipPrefill,
              lt) == QwenGemmRoute::kHipPrefillBf16Blas,
        "prefill K=1023 remains BLAS");
  Check(Route(GgmlType::kBF16, 1024, 1024, QwenGemmMode::kHipPrefill,
              lt) == QwenGemmRoute::kHipPrefillBf16LtTryThenBlas,
        "prefill threshold enables hipBLASLt try");
  Check(Route(GgmlType::kBF16, 1024, 1024, QwenGemmMode::kHipPrefill) ==
            QwenGemmRoute::kHipPrefillBf16Blas,
        "missing hipBLASLt capability uses BLAS");
  Check(Route(GgmlType::kF32, 1024, 1024, QwenGemmMode::kHipPrefill,
              lt) == QwenGemmRoute::kHipPrefillF32Blas,
        "prefill F32 BLAS route");
  Check(Route(GgmlType::kQ5_K, 1024, 256, QwenGemmMode::kHipPrefill,
              lt) == QwenGemmRoute::kHipPrefillQuantDirect,
        "prefill quant direct route");
}

void TestRejections() {
  auto result = ResolveQwenGemmRoute(
      {.type = GgmlType::kF32, .batch_size = 0, .m = 1, .k = 1});
  Check(result.rejection == QwenGemmRejection::kZeroShape,
        "zero batch rejection");
  result = ResolveQwenGemmRoute(
      {.type = GgmlType::kF32, .batch_size = 1, .m = 0, .k = 1});
  Check(result.rejection == QwenGemmRejection::kZeroShape,
        "zero M rejection");
  result = ResolveQwenGemmRoute(
      {.type = GgmlType::kF32,
       .batch_size = 1,
       .m = std::numeric_limits<std::size_t>::max(),
       .k = 2});
  Check(result.rejection == QwenGemmRejection::kShapeOverflow,
        "shape overflow rejection");
  result = ResolveQwenGemmRoute(
      {.type = GgmlType::kQ8_0, .batch_size = 1, .m = 1, .k = 31});
  Check(result.rejection == QwenGemmRejection::kMisalignedQuantK,
        "Q8_0 K=31 rejection");
  result = ResolveQwenGemmRoute(
      {.type = GgmlType::kQ5_K, .batch_size = 1, .m = 1, .k = 255});
  Check(result.rejection == QwenGemmRejection::kMisalignedQuantK,
        "K-quant K=255 rejection");
  Check(Route(GgmlType::kQ5_K, 1, 256, QwenGemmMode::kHipDecode, {}, true) ==
            QwenGemmRoute::kRejected,
        "quantized residual epilogue rejection");
  Check(Route(GgmlType::kQ4_K, 1, 256, QwenGemmMode::kHipDecode) ==
            QwenGemmRoute::kRejected,
        "unsupported HIP Q4_K rejection");
  Check(Route(GgmlType::kF16, 1, 256, QwenGemmMode::kHipMtp) ==
            QwenGemmRoute::kRejected,
        "unsupported MTP F16 rejection");
}

}  // namespace

int main() {
  TestFormatCapabilities();
  TestCpuRoutes();
  TestDecodeAndMtpRoutes();
  TestPrefillRoutes();
  TestRejections();
  std::cout << "Qwen GEMM route tests passed.\n";
  return 0;
}
