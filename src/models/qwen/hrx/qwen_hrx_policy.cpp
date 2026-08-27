#include "src/models/qwen/hrx/qwen_hrx_policy.hpp"

#include <sstream>
#include <vector>

namespace gufo::hrx {

namespace {

std::vector<std::string_view> SplitComma(std::string_view str) {
  std::vector<std::string_view> tokens;
  std::size_t start = 0;
  while (start < str.size()) {
    const std::size_t end = str.find(',', start);
    if (end == std::string_view::npos) {
      const auto token = str.substr(start);
      if (!token.empty()) {
        tokens.push_back(token);
      }
      break;
    }
    const auto token = str.substr(start, end - start);
    if (!token.empty()) {
      tokens.push_back(token);
    }
    start = end + 1;
  }
  return tokens;
}

}  // namespace

QwenHrxExecutionPolicy QwenHrxExecutionPolicy::Parse(std::string_view spec,
                                                     std::string* error_msg) {
  QwenHrxExecutionPolicy policy;
  if (spec.empty() || spec == "none") {
    return policy;
  }
  if (spec == "all") {
    policy.fused_swiglu_bf16 = true;
    policy.fused_down_residual_bf16 = true;
    policy.fused_rmsnorm_qkv_bf16 = true;
    policy.fused_rope_kv_bf16 = true;
    return policy;
  }

  const auto tokens = SplitComma(spec);
  for (const auto& token : tokens) {
    if (token == "none") {
      policy = QwenHrxExecutionPolicy{};
    } else if (token == "all") {
      policy.fused_swiglu_bf16 = true;
      policy.fused_down_residual_bf16 = true;
      policy.fused_rmsnorm_qkv_bf16 = true;
      policy.fused_rope_kv_bf16 = true;
    } else if (token == "swiglu") {
      policy.fused_swiglu_bf16 = true;
    } else if (token == "down-residual") {
      policy.fused_down_residual_bf16 = true;
    } else if (token == "rmsnorm-qkv") {
      policy.fused_rmsnorm_qkv_bf16 = true;
    } else if (token == "rope-kv") {
      policy.fused_rope_kv_bf16 = true;
    } else {
      if (error_msg != nullptr) {
        *error_msg = "Unknown HRX fusion flag: " + std::string(token) +
                     " (supported: none, all, swiglu, down-residual, "
                     "rmsnorm-qkv, rope-kv)";
      }
      return policy;
    }
  }
  return policy;
}

std::string QwenHrxExecutionPolicy::ToString() const {
  std::vector<std::string> enabled;
  if (fused_swiglu_bf16) {
    enabled.push_back("swiglu");
  }
  if (fused_down_residual_bf16) {
    enabled.push_back("down-residual");
  }
  if (fused_rmsnorm_qkv_bf16) {
    enabled.push_back("rmsnorm-qkv");
  }
  if (fused_rope_kv_bf16) {
    enabled.push_back("rope-kv");
  }
  if (enabled.empty()) {
    return "none";
  }
  std::ostringstream ss;
  for (std::size_t i = 0; i < enabled.size(); ++i) {
    if (i > 0)
      ss << ",";
    ss << enabled[i];
  }
  return ss.str();
}

}  // namespace gufo::hrx
