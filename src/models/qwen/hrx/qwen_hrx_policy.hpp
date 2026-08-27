#ifndef GUFO_MODELS_QWEN_HRX_QWEN_HRX_POLICY_HPP_
#define GUFO_MODELS_QWEN_HRX_QWEN_HRX_POLICY_HPP_

#include <cstdint>
#include <string>
#include <string_view>

namespace gufo::hrx {

struct QwenHrxExecutionPolicy {
  bool fused_swiglu_bf16{false};
  bool fused_down_residual_bf16{false};
  bool fused_rmsnorm_qkv_bf16{false};
  bool fused_rope_kv_bf16{false};

  [[nodiscard]] static QwenHrxExecutionPolicy Parse(
      std::string_view spec, std::string* error_msg = nullptr);
  [[nodiscard]] std::string ToString() const;
};

}  // namespace gufo::hrx

#endif  // GUFO_MODELS_QWEN_HRX_QWEN_HRX_POLICY_HPP_
