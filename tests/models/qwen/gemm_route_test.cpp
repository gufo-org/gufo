#include "src/models/qwen/gemm_route.hpp"

#include <array>
#include <cstdint>
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

auto Resolution(GgmlType type, std::size_t m, std::size_t k,
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
                               .residual_epilogue = residual_epilogue});
}

QwenGemmRoute Route(GgmlType type, std::size_t m, std::size_t k,
                    QwenGemmMode mode,
                    QwenGemmCapabilities capabilities = {},
                    bool residual_epilogue = false,
                    std::size_t batch_size = 1) {
  return Resolution(type, m, k, mode, capabilities, residual_epilogue,
                    batch_size)
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
    constexpr std::array modes{
        QwenGemmMode::kCpu,
        QwenGemmMode::kHipDecode,
        QwenGemmMode::kHipPrefill,
        QwenGemmMode::kHipMtp,
    };
    for (const auto mode : modes) {
      const auto resolution = Resolution(type, 1, k, mode);
      const bool expected =
          mode == QwenGemmMode::kCpu
              ? descriptor.cpu_direct
              : mode == QwenGemmMode::kHipPrefill
                    ? descriptor.hip_prefill_direct
                    : descriptor.hip_decode_direct;
      Check(resolution.accepted() == expected,
            "route must match the format/mode capability descriptor");
      Check(expected
                ? resolution.rejection == QwenGemmRejection::kNone
                : resolution.rejection ==
                      QwenGemmRejection::kUnsupportedFormat,
            "format/mode route must expose the expected rejection reason");
    }
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

void TestBackendDimensionBoundaries() {
  const std::size_t int_max =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  const std::size_t uint_max =
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());

  Check(Resolution(GgmlType::kBF16, int_max, 1,
                   QwenGemmMode::kHipPrefill)
            .accepted(),
        "hipBLAS M=INT_MAX must remain representable");
  Check(Resolution(GgmlType::kBF16, 1, int_max,
                   QwenGemmMode::kHipPrefill)
            .accepted(),
        "hipBLAS K=INT_MAX must remain representable");
  Check(Resolution(GgmlType::kBF16, 1, 1,
                   QwenGemmMode::kHipPrefill, {}, false, int_max)
            .accepted(),
        "hipBLAS batch=INT_MAX must remain representable");
  Check(Resolution(GgmlType::kBF16, int_max + 1U, 1,
                   QwenGemmMode::kHipPrefill)
            .rejection == QwenGemmRejection::kShapeOverflow,
        "hipBLAS M above INT_MAX must reject");
  Check(Resolution(GgmlType::kBF16, 1, int_max + 1U,
                   QwenGemmMode::kHipPrefill)
            .rejection == QwenGemmRejection::kShapeOverflow,
        "hipBLAS K above INT_MAX must reject");
  Check(Resolution(GgmlType::kBF16, 1, 1,
                   QwenGemmMode::kHipPrefill, {}, false, int_max + 1U)
            .rejection == QwenGemmRejection::kShapeOverflow,
        "hipBLAS batch above INT_MAX must reject");

  Check(Resolution(GgmlType::kF32, uint_max, 1,
                   QwenGemmMode::kHipDecode)
            .accepted(),
        "decode grid M=UINT_MAX must remain representable");
  Check(Resolution(GgmlType::kF32, uint_max + 1U, 1,
                   QwenGemmMode::kHipDecode)
            .rejection == QwenGemmRejection::kShapeOverflow,
        "decode grid M above UINT_MAX must reject");
  Check(Resolution(GgmlType::kBF16, uint_max, 1,
                   QwenGemmMode::kHipMtp)
            .accepted(),
        "MTP grid M=UINT_MAX must remain representable");
  Check(Resolution(GgmlType::kBF16, uint_max + 1U, 1,
                   QwenGemmMode::kHipMtp)
            .rejection == QwenGemmRejection::kShapeOverflow,
        "MTP grid M above UINT_MAX must reject");

  Check(Resolution(GgmlType::kQ8_0, 1, 32,
                   QwenGemmMode::kHipPrefill, {}, false, uint_max)
            .accepted(),
        "quant prefill batch=UINT_MAX must remain representable");
  Check(Resolution(GgmlType::kQ8_0, 1, 32,
                   QwenGemmMode::kHipPrefill, {}, false, uint_max + 1U)
            .rejection == QwenGemmRejection::kShapeOverflow,
        "quant prefill batch above UINT_MAX must reject");

  constexpr std::size_t rows_per_block = 4;
  if (uint_max <=
      std::numeric_limits<std::size_t>::max() / rows_per_block) {
    const std::size_t max_rounded_m = uint_max * rows_per_block;
    Check(Resolution(GgmlType::kQ8_0, max_rounded_m, 32,
                     QwenGemmMode::kHipPrefill)
              .accepted(),
          "quant prefill rounded grid at UINT_MAX must remain representable");
    Check(Resolution(GgmlType::kQ8_0, max_rounded_m + 1U, 32,
                     QwenGemmMode::kHipPrefill)
              .rejection == QwenGemmRejection::kShapeOverflow,
          "quant prefill rounded grid above UINT_MAX must reject");
  }
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
      {.type = GgmlType::kF32, .batch_size = 1, .m = 1, .k = 0});
  Check(result.rejection == QwenGemmRejection::kZeroShape,
        "zero K rejection");
  result = ResolveQwenGemmRoute(
      {.type = GgmlType::kF32,
       .batch_size = 1,
       .m = std::numeric_limits<std::size_t>::max(),
       .k = 2});
  Check(result.rejection == QwenGemmRejection::kShapeOverflow,
        "M*K overflow rejection");
  result = ResolveQwenGemmRoute(
      {.type = GgmlType::kF32,
       .batch_size = std::numeric_limits<std::size_t>::max(),
       .m = 2,
       .k = 1});
  Check(result.rejection == QwenGemmRejection::kShapeOverflow,
        "batch*M overflow rejection");
  result = ResolveQwenGemmRoute(
      {.type = GgmlType::kF32,
       .batch_size = std::numeric_limits<std::size_t>::max(),
       .m = 1,
       .k = 2});
  Check(result.rejection == QwenGemmRejection::kShapeOverflow,
        "batch*K overflow rejection");
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
  TestBackendDimensionBoundaries();
  TestRejections();
  std::cout << "Qwen GEMM route tests passed.\n";
  return 0;
}
