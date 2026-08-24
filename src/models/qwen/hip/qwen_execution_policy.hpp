#ifndef STRIX_MODELS_QWEN_HIP_QWEN_EXECUTION_POLICY_HPP_
#define STRIX_MODELS_QWEN_HIP_QWEN_EXECUTION_POLICY_HPP_

#include <cstdint>

namespace strix::hip {

/// Immutable route policy for one Qwen GPU executor. Resolve this before HIP
/// graph capture and create a separate executor for each A/B candidate.
enum class QwenExecutionMode : std::uint8_t {
  kDecode,
  kPrefill,
};

struct QwenExecutionPolicy {
  bool fuse_qk_norm_rope_kv{true};
  bool fuse_residual_rmsnorm{false};
  bool fuse_prefill_ffn_swiglu{false};
  bool fuse_decode_rmsnorm_swiglu{false};
  bool fuse_decode_ssm_output_residual{false};
  bool fuse_prefill_ssm_post_norm_gate{false};
  bool fuse_decode_rmsnorm_projection{false};
  bool prefetch_next_layer{false};

  [[nodiscard]] static constexpr QwenExecutionPolicy Production() noexcept {
    return {};
  }

  /// Stable bit fingerprint suitable for telemetry and graph-cache identity.
  [[nodiscard]] constexpr std::uint64_t Fingerprint() const noexcept {
    return (static_cast<std::uint64_t>(fuse_qk_norm_rope_kv) << 0U) |
           (static_cast<std::uint64_t>(fuse_residual_rmsnorm) << 1U) |
           (static_cast<std::uint64_t>(fuse_prefill_ffn_swiglu) << 2U) |
           (static_cast<std::uint64_t>(fuse_decode_rmsnorm_swiglu) << 3U) |
           (static_cast<std::uint64_t>(fuse_decode_ssm_output_residual) << 4U) |
           (static_cast<std::uint64_t>(fuse_prefill_ssm_post_norm_gate) << 5U) |
           (static_cast<std::uint64_t>(fuse_decode_rmsnorm_projection) << 6U) |
           (static_cast<std::uint64_t>(prefetch_next_layer) << 7U);
  }
};

/// Resolved route decisions for one layer invocation. This record is pure data
/// so policy selection can be tested independently from HIP launches.
struct QwenLayerRoutePlan {
  QwenExecutionMode mode{QwenExecutionMode::kDecode};
  bool full_attention{false};
  bool fuse_qk_norm_rope_kv{false};
  bool fuse_residual_rmsnorm{false};
  bool fuse_ffn_swiglu{false};
  bool fuse_ssm_epilogue{false};
  bool fuse_rmsnorm_projection{false};
  bool prefetch_next_layer{false};

  [[nodiscard]] constexpr std::uint64_t Fingerprint() const noexcept {
    return (static_cast<std::uint64_t>(mode == QwenExecutionMode::kPrefill)
            << 0U) |
           (static_cast<std::uint64_t>(full_attention) << 1U) |
           (static_cast<std::uint64_t>(fuse_qk_norm_rope_kv) << 2U) |
           (static_cast<std::uint64_t>(fuse_residual_rmsnorm) << 3U) |
           (static_cast<std::uint64_t>(fuse_ffn_swiglu) << 4U) |
           (static_cast<std::uint64_t>(fuse_ssm_epilogue) << 5U) |
           (static_cast<std::uint64_t>(fuse_rmsnorm_projection) << 6U) |
           (static_cast<std::uint64_t>(prefetch_next_layer) << 7U);
  }
};

[[nodiscard]] constexpr QwenLayerRoutePlan ResolveQwenLayerRoute(
    const QwenExecutionPolicy& policy, QwenExecutionMode mode,
    bool full_attention) noexcept {
  const bool decode = mode == QwenExecutionMode::kDecode;
  return {
      .mode = mode,
      .full_attention = full_attention,
      .fuse_qk_norm_rope_kv =
          full_attention && policy.fuse_qk_norm_rope_kv,
      .fuse_residual_rmsnorm = policy.fuse_residual_rmsnorm,
      .fuse_ffn_swiglu =
          decode ? policy.fuse_decode_rmsnorm_swiglu
                 : policy.fuse_prefill_ffn_swiglu,
      .fuse_ssm_epilogue =
          !full_attention &&
          (decode ? policy.fuse_decode_ssm_output_residual
                  : policy.fuse_prefill_ssm_post_norm_gate),
      .fuse_rmsnorm_projection =
          decode && policy.fuse_decode_rmsnorm_projection,
      .prefetch_next_layer = decode && policy.prefetch_next_layer,
  };
}

}  // namespace strix::hip

#endif  // STRIX_MODELS_QWEN_HIP_QWEN_EXECUTION_POLICY_HPP_
