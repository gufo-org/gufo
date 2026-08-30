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
  /// Q8_0 routes that need no additional artifacts.
  bool fused_ffn_gate_up{false};
  bool fused_ssm_alpha_beta{false};
  bool residual_ping_pong{false};
  bool chunked_prefill{false};
  bool int8_prefill{false};
  bool wmma_prefill{false};
  bool blocked_prefill{false};
  bool fused_swiglu_quantize{false};
  bool fused_norm_quantize{false};
  bool fused_readout_quantize{false};
  /// Gate/up projection writes the post-SwiGLU blocked Q8 operand directly.
  bool fused_gate_up_swiglu_quantize{false};
  /// Blocked projection stages two adjacent K blocks per global load pair.
  bool paired_k_stage{false};
  /// Split K across workgroups for projections that fit one row group.
  bool split_k_narrow_rows{false};

  [[nodiscard]] static QwenHrxExecutionPolicy Parse(
      std::string_view spec, std::string* error_msg = nullptr);
  [[nodiscard]] std::string ToString() const;
};

}  // namespace gufo::hrx

#endif  // GUFO_MODELS_QWEN_HRX_QWEN_HRX_POLICY_HPP_
