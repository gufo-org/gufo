#ifndef GUFO_MODELS_GEMMA4_PREFILL_WEIGHTS_HPP_
#define GUFO_MODELS_GEMMA4_PREFILL_WEIGHTS_HPP_

#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace gufo::models::gemma4 {

/// Prefill copies of the dense projections, kept beside the GGUF weights
/// that decode reads. Prefill results are bit-identical in every mode.
enum class PrefillWeights : std::uint32_t {
  /// GGUF weights only.
  kOff,
  /// K-quant weights as PrefillLayout::kCompact.
  kCompact,
  /// The fastest layout per projection, binary16 where it pays.
  kDecoded,
};

/// --prefill-weights: off, compact or decoded; empty is off. Throws
/// std::invalid_argument on any other name.
[[nodiscard]] inline PrefillWeights ParsePrefillWeightsOption(
    std::string_view name) {
  if (name.empty() || name == "off") {
    return PrefillWeights::kOff;
  }
  if (name == "compact") {
    return PrefillWeights::kCompact;
  }
  if (name == "decoded") {
    return PrefillWeights::kDecoded;
  }
  throw std::invalid_argument(
      "--prefill-weights must be off, compact or decoded");
}

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_PREFILL_WEIGHTS_HPP_
