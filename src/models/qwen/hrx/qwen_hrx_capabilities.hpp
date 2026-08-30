#ifndef GUFO_MODELS_QWEN_HRX_QWEN_HRX_CAPABILITIES_HPP_
#define GUFO_MODELS_QWEN_HRX_QWEN_HRX_CAPABILITIES_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include "src/core/model_registry.hpp"
#include "src/models/qwen/hrx/qwen_hrx_contract.hpp"
#include "src/models/qwen/hrx/qwen_hrx_manifest.hpp"

namespace gufo::hrx {

/// High-level primitive classifications for native HRX model execution.
enum class HrxRequiredPrimitive : std::uint8_t {
  kEmbedding,
  kAttention,
  kSsm,
  kFfn,
  kArgmax,
};

enum class HrxOptionalPrimitive : std::uint8_t {
  kSpeculativeVerifier,
  kKvCacheQuantization,
  kMultiTokenDecode,
};

struct QwenHrxCapabilityReport {
  bool is_capable{false};
  std::vector<std::string> available_required_primitives;
  std::vector<std::string> missing_required_primitives;
  std::vector<std::string> enabled_optional_primitives;
  std::vector<std::string> missing_optional_primitives;

  [[nodiscard]] std::string ToString() const;
};

/// Probes kernel artifacts in `kernels_dir` against model contract.
[[nodiscard]] QwenHrxCapabilityReport ProbeHrxCapabilities(
    const std::string& kernels_dir, const core::ModelConfig& config,
    std::string* error_msg = nullptr);

/// Probes an existing manifest against model contract.
[[nodiscard]] QwenHrxCapabilityReport ProbeHrxCapabilities(
    const HrxArtifactManifest& manifest, const core::ModelConfig& config,
    std::string* error_msg = nullptr);

}  // namespace gufo::hrx

#endif  // GUFO_MODELS_QWEN_HRX_QWEN_HRX_CAPABILITIES_HPP_
