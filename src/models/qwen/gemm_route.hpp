#ifndef GUFO_MODELS_QWEN_GEMM_ROUTE_HPP_
#define GUFO_MODELS_QWEN_GEMM_ROUTE_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::qwen {

enum class QwenGemmMode : std::uint8_t {
  kCpu,
  kHipDecode,
  kHipPrefill,
  kHipMtp,
};

enum class QwenGemmRoute : std::uint8_t {
  kRejected,
  kCpuF32Rows,
  kCpuBf16Rows,
  kCpuF16Rows,
  kCpuQuantDot,
  kHipQuantDirect,
  kHipF32Baseline256,
  kHipF32Baseline512,
  kHipBf16Baseline256,
  kHipBf16Baseline512,
  kHipBf16Wave32Single,
  kHipPrefillF32Blas,
  kHipPrefillBf16Fp32,
  kHipPrefillQuantDirect,
};

enum class QwenGemmRejection : std::uint8_t {
  kNone,
  kZeroShape,
  kShapeOverflow,
  kUnsupportedFormat,
  kMisalignedQuantK,
};

struct QwenGemmFormatCapabilities {
  bool dense{false};
  bool quantized{false};
  std::size_t block_elements{0};
  bool cpu_direct{false};
  bool hip_decode_direct{false};
  bool hip_prefill_direct{false};
};

/// Static format capability metadata. Runtime byte geometry remains owned by
/// quant::QuantizedRowBytes/EncodedSizeBytes; this descriptor only states the
/// logical block alignment required before a route can be selected.
[[nodiscard]] constexpr QwenGemmFormatCapabilities DescribeQwenGemmFormat(
    core::GgmlType type) noexcept {
  switch (type) {
    case core::GgmlType::kI32:
      return {};
    case core::GgmlType::kF32:
    case core::GgmlType::kBF16:
      return {.dense = true,
              .cpu_direct = true,
              .hip_decode_direct = true,
              .hip_prefill_direct = true};
    case core::GgmlType::kF16:
      return {.dense = true, .cpu_direct = true};
    // Every quantized format below has an in-kernel decoder
    // (src/models/qwen/hip/quant_ops.hpp DecodeQuantSub16) and is therefore
    // direct on the CPU and on both GPU paths. opt-q4kxl added Q3_K, Q4_K,
    // IQ4_NL, IQ4_XS and IQ3_S to this set so the mixed low-bit UD-Q4_K_XL
    // shard runs packed instead of being pre-expanded to BF16.
    case core::GgmlType::kQ8_0:
    case core::GgmlType::kQ3_K:
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kIQ4_NL:
    case core::GgmlType::kIQ4_XS:
    case core::GgmlType::kIQ3_S:
    case core::GgmlType::kQ5_K:
    case core::GgmlType::kQ6_K:
    case core::GgmlType::kQ8_K:
      return {.quantized = true,
              .block_elements = ::gufo::quant::QuantizedBlockElements(type),
              .cpu_direct = true,
              .hip_decode_direct = true,
              .hip_prefill_direct = true};
    case core::GgmlType::kQ4_0:
    case core::GgmlType::kQ4_1:
    case core::GgmlType::kQ5_0:
    case core::GgmlType::kQ5_1:
    case core::GgmlType::kQ8_1:
    case core::GgmlType::kQ2_K:
    case core::GgmlType::kIQ2_XXS:
      return {.quantized = true};
  }
  return {};
}

struct QwenGemmRequest {
  core::GgmlType type{core::GgmlType::kF32};
  std::size_t batch_size{1};
  std::size_t m{0};
  std::size_t k{0};
  QwenGemmMode mode{QwenGemmMode::kCpu};
};

[[nodiscard]] constexpr std::string_view QwenGemmRouteName(
    QwenGemmRoute route) noexcept {
  switch (route) {
    case QwenGemmRoute::kRejected:
      return "rejected";
    case QwenGemmRoute::kCpuF32Rows:
      return "cpu_f32_rows";
    case QwenGemmRoute::kCpuBf16Rows:
      return "cpu_bf16_rows";
    case QwenGemmRoute::kCpuF16Rows:
      return "cpu_f16_rows";
    case QwenGemmRoute::kCpuQuantDot:
      return "cpu_quant_dot";
    case QwenGemmRoute::kHipQuantDirect:
    case QwenGemmRoute::kHipPrefillQuantDirect:
      return "quant_direct";
    case QwenGemmRoute::kHipF32Baseline256:
    case QwenGemmRoute::kHipF32Baseline512:
    case QwenGemmRoute::kHipBf16Baseline256:
    case QwenGemmRoute::kHipBf16Baseline512:
      return "baseline_block";
    case QwenGemmRoute::kHipBf16Wave32Single:
      return "wave32_single_row";
    case QwenGemmRoute::kHipPrefillF32Blas:
      return "hipblas";
    case QwenGemmRoute::kHipPrefillBf16Fp32:
      return "bf16_fp32";
  }
  return "unknown";
}

struct QwenGemmResolution {
  QwenGemmRoute route{QwenGemmRoute::kRejected};
  QwenGemmRejection rejection{QwenGemmRejection::kNone};

  [[nodiscard]] constexpr bool accepted() const noexcept {
    return route != QwenGemmRoute::kRejected;
  }
};

[[nodiscard]] constexpr QwenGemmResolution RejectQwenGemm(
    QwenGemmRejection rejection) noexcept {
  return {.route = QwenGemmRoute::kRejected, .rejection = rejection};
}

[[nodiscard]] constexpr QwenGemmResolution ResolveQwenGemmRoute(
    const QwenGemmRequest& request) noexcept {
  if (request.batch_size == 0 || request.m == 0 || request.k == 0) {
    return RejectQwenGemm(QwenGemmRejection::kZeroShape);
  }
  const std::size_t max_size = std::numeric_limits<std::size_t>::max();
  if (request.m > max_size / request.k ||
      request.batch_size > max_size / request.m ||
      request.batch_size > max_size / request.k) {
    return RejectQwenGemm(QwenGemmRejection::kShapeOverflow);
  }

  const auto format = DescribeQwenGemmFormat(request.type);
  const bool supported = request.mode == QwenGemmMode::kCpu ? format.cpu_direct
                         : request.mode == QwenGemmMode::kHipPrefill
                             ? format.hip_prefill_direct
                             : format.hip_decode_direct;
  if (!supported) {
    return RejectQwenGemm(QwenGemmRejection::kUnsupportedFormat);
  }
  if (format.quantized &&
      (format.block_elements == 0 || request.k % format.block_elements != 0)) {
    return RejectQwenGemm(QwenGemmRejection::kMisalignedQuantK);
  }

  if (request.mode == QwenGemmMode::kHipPrefill) {
    if (format.quantized) {
      constexpr std::size_t kRowsPerBlock = 4;
      constexpr std::size_t kRoundUp = kRowsPerBlock - 1U;
      const std::size_t max_grid = std::numeric_limits<std::uint32_t>::max();
      if (request.batch_size > max_grid || request.m > max_size - kRoundUp ||
          (request.m + kRoundUp) / kRowsPerBlock > max_grid) {
        return RejectQwenGemm(QwenGemmRejection::kShapeOverflow);
      }
    } else {
      const std::size_t max_blas =
          static_cast<std::size_t>(std::numeric_limits<int>::max());
      if (request.batch_size > max_blas || request.m > max_blas ||
          request.k > max_blas) {
        return RejectQwenGemm(QwenGemmRejection::kShapeOverflow);
      }
    }
  } else if (request.mode == QwenGemmMode::kHipDecode ||
             request.mode == QwenGemmMode::kHipMtp) {
    if (request.m > std::numeric_limits<std::uint32_t>::max()) {
      return RejectQwenGemm(QwenGemmRejection::kShapeOverflow);
    }
  }

  if (request.mode == QwenGemmMode::kCpu) {
    switch (request.type) {
      case core::GgmlType::kF32:
        return {.route = QwenGemmRoute::kCpuF32Rows};
      case core::GgmlType::kBF16:
        return {.route = QwenGemmRoute::kCpuBf16Rows};
      case core::GgmlType::kF16:
        return {.route = QwenGemmRoute::kCpuF16Rows};
      default:
        return {.route = QwenGemmRoute::kCpuQuantDot};
    }
  }

  if (request.mode == QwenGemmMode::kHipPrefill) {
    if (format.quantized) {
      return {.route = QwenGemmRoute::kHipPrefillQuantDirect};
    }
    if (request.type == core::GgmlType::kF32) {
      return {.route = QwenGemmRoute::kHipPrefillF32Blas};
    }
    return {.route = QwenGemmRoute::kHipPrefillBf16Fp32};
  }

  if (format.quantized) {
    return {.route = QwenGemmRoute::kHipQuantDirect};
  }
  if (request.type == core::GgmlType::kF32) {
    return {.route = request.k >= 16384 ? QwenGemmRoute::kHipF32Baseline512
                                        : QwenGemmRoute::kHipF32Baseline256};
  }
  if (request.k % 8 == 0 && request.k < 8192) {
    return {.route = QwenGemmRoute::kHipBf16Wave32Single};
  }
  return {.route = request.k >= 16384 ? QwenGemmRoute::kHipBf16Baseline512
                                      : QwenGemmRoute::kHipBf16Baseline256};
}

}  // namespace gufo::models::qwen

#endif  // GUFO_MODELS_QWEN_GEMM_ROUTE_HPP_
