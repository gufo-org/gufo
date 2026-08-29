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
    policy.fused_ffn_gate_up = true;
    policy.fused_ssm_alpha_beta = true;
    policy.residual_ping_pong = true;
    policy.chunked_prefill = true;
    policy.int8_prefill = true;
    policy.wmma_prefill = true;
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
      policy.fused_ffn_gate_up = true;
      policy.fused_ssm_alpha_beta = true;
      policy.residual_ping_pong = true;
      policy.chunked_prefill = true;
      policy.int8_prefill = true;
      policy.wmma_prefill = true;
    } else if (token == "swiglu") {
      policy.fused_swiglu_bf16 = true;
    } else if (token == "down-residual") {
      policy.fused_down_residual_bf16 = true;
    } else if (token == "rmsnorm-qkv") {
      policy.fused_rmsnorm_qkv_bf16 = true;
    } else if (token == "rope-kv") {
      policy.fused_rope_kv_bf16 = true;
    } else if (token == "q8") {
      policy.fused_ffn_gate_up = true;
      policy.fused_ssm_alpha_beta = true;
      policy.residual_ping_pong = true;
      policy.chunked_prefill = true;
      policy.int8_prefill = true;
    } else if (token == "chunked-prefill") {
      policy.chunked_prefill = true;
    } else if (token == "blocked-prefill") {
      // The blocked route runs inside the chunked prefill stages and consumes
      // int8 operands; it just uses a 128-token tile instead of eight.
      policy.chunked_prefill = true;
      policy.int8_prefill = true;
      policy.blocked_prefill = true;
    } else if (token == "swiglu-quant") {
      // SwiGLU folded into the blocked activation quantizer; only the blocked
      // route consumes the quantized payload, so it implies that route.
      policy.chunked_prefill = true;
      policy.int8_prefill = true;
      policy.blocked_prefill = true;
      policy.fused_swiglu_quantize = true;
    } else if (token == "norm-quant") {
      // RMSNorm folded into the blocked activation quantizer; only the blocked
      // route consumes the quantized payload, so it implies that route.
      policy.chunked_prefill = true;
      policy.int8_prefill = true;
      policy.blocked_prefill = true;
      policy.fused_norm_quantize = true;
    } else if (token == "readout-quant") {
      // The DeltaNet readout writes the activation the blocked projection
      // consumes, so folding the quantizer into it implies that route.
      policy.chunked_prefill = true;
      policy.int8_prefill = true;
      policy.blocked_prefill = true;
      policy.fused_readout_quantize = true;
    } else if (token == "int8-prefill") {
      // The int8 route runs inside the chunked prefill stages.
      policy.chunked_prefill = true;
      policy.int8_prefill = true;
    } else if (token == "wmma-prefill") {
      // WMMA consumes the padded int8 activation representation and therefore
      // runs only inside the int8 chunked-prefill route.
      policy.chunked_prefill = true;
      policy.int8_prefill = true;
      policy.wmma_prefill = true;
    } else if (token == "ffn-gate-up") {
      policy.fused_ffn_gate_up = true;
    } else if (token == "ssm-alpha-beta") {
      policy.fused_ssm_alpha_beta = true;
    } else if (token == "ping-pong") {
      policy.residual_ping_pong = true;
    } else {
      if (error_msg != nullptr) {
        *error_msg = "Unknown HRX fusion flag: " + std::string(token) +
                     " (supported: none, all, swiglu, down-residual, "
                     "rmsnorm-qkv, rope-kv, q8, ffn-gate-up, "
                     "ssm-alpha-beta, ping-pong, "
                     "chunked-prefill, int8-prefill, blocked-prefill, "
                     "swiglu-quant, norm-quant, readout-quant, wmma-prefill)";
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
  if (fused_ffn_gate_up) {
    enabled.push_back("ffn-gate-up");
  }
  if (fused_ssm_alpha_beta) {
    enabled.push_back("ssm-alpha-beta");
  }
  if (residual_ping_pong) {
    enabled.push_back("ping-pong");
  }
  if (chunked_prefill) {
    enabled.push_back("chunked-prefill");
  }
  if (int8_prefill) {
    enabled.push_back("int8-prefill");
  }
  if (wmma_prefill) {
    enabled.push_back("wmma-prefill");
  }
  if (blocked_prefill) {
    enabled.push_back("blocked-prefill");
  }
  if (fused_swiglu_quantize) {
    enabled.push_back("swiglu-quant");
  }
  if (fused_norm_quantize) {
    enabled.push_back("norm-quant");
  }
  if (fused_readout_quantize) {
    enabled.push_back("readout-quant");
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
