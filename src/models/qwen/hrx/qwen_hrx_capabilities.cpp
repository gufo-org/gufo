#include "src/models/qwen/hrx/qwen_hrx_capabilities.hpp"

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace gufo::hrx {

std::string QwenHrxCapabilityReport::ToString() const {
  std::ostringstream ss;
  ss << "HRX Native Capabilities: capable=" << (is_capable ? "yes" : "no")
     << "\n";
  ss << "  Available Required Primitives:\n";
  for (const auto& prim : available_required_primitives) {
    ss << "    - " << prim << "\n";
  }
  if (!missing_required_primitives.empty()) {
    ss << "  Missing Required Primitives:\n";
    for (const auto& prim : missing_required_primitives) {
      ss << "    - " << prim << "\n";
    }
  }
  if (!enabled_optional_primitives.empty()) {
    ss << "  Enabled Optional Primitives:\n";
    for (const auto& prim : enabled_optional_primitives) {
      ss << "    - " << prim << "\n";
    }
  }
  if (!missing_optional_primitives.empty()) {
    ss << "  Missing Optional Primitives:\n";
    for (const auto& prim : missing_optional_primitives) {
      ss << "    - " << prim << "\n";
    }
  }
  return ss.str();
}

QwenHrxCapabilityReport ProbeHrxCapabilities(const std::string& kernels_dir,
                                             const core::ModelConfig& config,
                                             std::string* error_msg) {
  auto manifest =
      HrxArtifactManifest::LoadFromDirectory(kernels_dir, error_msg);
  if (manifest == nullptr) {
    QwenHrxCapabilityReport report;
    report.is_capable = false;
    report.missing_required_primitives = {"Embedding", "Attention", "SSM",
                                          "FFN", "Argmax"};
    return report;
  }
  return ProbeHrxCapabilities(*manifest, config, error_msg);
}

QwenHrxCapabilityReport ProbeHrxCapabilities(
    const HrxArtifactManifest& manifest, const core::ModelConfig& config,
    std::string* error_msg) {
  QwenHrxCapabilityReport report;

  const auto contract = QwenHrxArtifactContract::FromConfig(config, error_msg);
  if (!contract.has_value()) {
    report.is_capable = false;
    report.missing_required_primitives = {"Embedding", "Attention", "SSM",
                                          "FFN", "Argmax"};
    return report;
  }

  std::unordered_set<std::string> present_artifacts;
  for (const auto& entry : manifest.Entries()) {
    if (!entry.sha256.empty()) {
      present_artifacts.insert(entry.name);
    }
  }

  const auto has_all = [&](const std::vector<std::string>& names) {
    return std::all_of(
        names.begin(), names.end(), [&](const std::string& name) {
          return present_artifacts.find(name) != present_artifacts.end();
        });
  };

  // 1. Embedding
  if (has_all({"qwen_q8_embedding"})) {
    report.available_required_primitives.push_back("Embedding");
  } else {
    report.missing_required_primitives.push_back("Embedding");
  }

  // 2. Attention
  if (has_all({"qwen_rmsnorm_qkv", "qwen_split_q_gate", "qwen_per_head_rmsnorm",
               "qwen_rope_kv", "qwen_attention_decode", "qwen_residual_add",
               "qwen_copy"})) {
    report.available_required_primitives.push_back("Attention");
  } else {
    report.missing_required_primitives.push_back("Attention");
  }

  // 3. SSM
  if (has_all({"qwen_rmsnorm_qkv", "qwen_ssm_conv", "qwen_deltanet_prepare",
               "qwen_deltanet_recurrence", "qwen_residual_add", "qwen_copy"})) {
    report.available_required_primitives.push_back("SSM");
  } else {
    report.missing_required_primitives.push_back("SSM");
  }

  // 4. FFN
  if (has_all({"qwen_rmsnorm", "qwen_swiglu", "qwen_q8_gemv_k5120",
               "qwen_q8_gemv_k6144", "qwen_q8_gemv_k17408",
               "qwen_down_residual", "qwen_copy"}) ||
      has_all({"qwen_rmsnorm", "qwen_swiglu_pointwise", "qwen_q8_gemv_k5120",
               "qwen_q8_gemv_k6144", "qwen_q8_gemv_k17408", "qwen_residual_add",
               "qwen_copy"})) {
    report.available_required_primitives.push_back("FFN");
  } else {
    report.missing_required_primitives.push_back("FFN");
  }

  // 5. Argmax
  if (has_all({"qwen_rmsnorm", "qwen_q8_vocab_gemv_k5120", "qwen_argmax"})) {
    report.available_required_primitives.push_back("Argmax");
  } else {
    report.missing_required_primitives.push_back("Argmax");
  }

  // Optional: Multi-token decode
  if (has_all({"qwen_q8_gemv_k17408_wg256"})) {
    report.enabled_optional_primitives.push_back("MultiTokenDecode");
  } else {
    report.missing_optional_primitives.push_back("MultiTokenDecode");
  }

  // Optional: Speculative verifier oracle
  if (has_all({"qwen_q8_decode_oracle"})) {
    report.enabled_optional_primitives.push_back("SpeculativeVerifier");
  } else {
    report.missing_optional_primitives.push_back("SpeculativeVerifier");
  }

  report.is_capable = report.missing_required_primitives.empty();
  return report;
}

}  // namespace gufo::hrx
