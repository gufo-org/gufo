#ifndef GUFO_MODELS_QWEN_HIP_EXECUTION_POLICY_HPP_
#define GUFO_MODELS_QWEN_HIP_EXECUTION_POLICY_HPP_

#include <cstddef>
#include <cstdint>

namespace gufo::hip {

enum class QwenKvCacheStorage : std::uint8_t {
  kFp32,
  kFp16,
};

enum class QwenRecurrentStateStorage : std::uint8_t {
  kFp32,
  kBf16,
};

[[nodiscard]] constexpr std::size_t QwenRecurrentStateElementBytes(
    QwenRecurrentStateStorage storage) noexcept {
  return storage == QwenRecurrentStateStorage::kBf16 ? sizeof(std::uint16_t)
                                                     : sizeof(float);
}

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
  QwenKvCacheStorage kv_cache_storage{QwenKvCacheStorage::kFp16};
  QwenRecurrentStateStorage recurrent_state_storage{
      QwenRecurrentStateStorage::kFp32};

  [[nodiscard]] static constexpr QwenExecutionPolicy Production() noexcept {
    return {};
  }

  [[nodiscard]] constexpr bool UsesFp16AttentionKv() const noexcept {
    return kv_cache_storage == QwenKvCacheStorage::kFp16;
  }

  [[nodiscard]] constexpr bool UsesBf16RecurrentState() const noexcept {
    return recurrent_state_storage == QwenRecurrentStateStorage::kBf16;
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
           (static_cast<std::uint64_t>(prefetch_next_layer) << 7U) |
           (static_cast<std::uint64_t>(UsesFp16AttentionKv()) << 8U) |
           (static_cast<std::uint64_t>(UsesBf16RecurrentState()) << 9U);
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

enum class QwenRouteRejection : std::uint32_t {
  kNone = 0,
  kQkNormRopeKvRequiresAttention = 1U << 0U,
  kSsmEpilogueRequiresSsm = 1U << 1U,
  kPrefillFfnSwiGluRequiresPrefill = 1U << 2U,
  kDecodeFfnSwiGluRequiresDecode = 1U << 3U,
  kDecodeSsmEpilogueRequiresDecode = 1U << 4U,
  kPrefillSsmEpilogueRequiresPrefill = 1U << 5U,
  kRmsNormProjectionRequiresDecode = 1U << 6U,
  kPrefetchRequiresDecode = 1U << 7U,
};

[[nodiscard]] constexpr QwenRouteRejection operator|(
    QwenRouteRejection lhs, QwenRouteRejection rhs) noexcept {
  return static_cast<QwenRouteRejection>(static_cast<std::uint32_t>(lhs) |
                                         static_cast<std::uint32_t>(rhs));
}

constexpr QwenRouteRejection& operator|=(QwenRouteRejection& lhs,
                                         QwenRouteRejection rhs) noexcept {
  lhs = lhs | rhs;
  return lhs;
}

[[nodiscard]] constexpr bool HasQwenRouteRejection(
    QwenRouteRejection reasons, QwenRouteRejection reason) noexcept {
  return (static_cast<std::uint32_t>(reasons) &
          static_cast<std::uint32_t>(reason)) != 0U;
}

struct QwenLayerRouteResolution {
  QwenLayerRoutePlan plan;
  QwenRouteRejection rejected{QwenRouteRejection::kNone};
};

[[nodiscard]] constexpr QwenLayerRouteResolution
ResolveQwenLayerRouteWithReasons(const QwenExecutionPolicy& policy,
                                 QwenExecutionMode mode,
                                 bool full_attention) noexcept {
  const bool decode = mode == QwenExecutionMode::kDecode;
  QwenRouteRejection rejected = QwenRouteRejection::kNone;

  if (policy.fuse_qk_norm_rope_kv && !full_attention) {
    rejected |= QwenRouteRejection::kQkNormRopeKvRequiresAttention;
  }
  if (policy.fuse_prefill_ffn_swiglu && decode) {
    rejected |= QwenRouteRejection::kPrefillFfnSwiGluRequiresPrefill;
  }
  if (policy.fuse_decode_rmsnorm_swiglu && !decode) {
    rejected |= QwenRouteRejection::kDecodeFfnSwiGluRequiresDecode;
  }
  if (policy.fuse_decode_ssm_output_residual && !decode) {
    rejected |= QwenRouteRejection::kDecodeSsmEpilogueRequiresDecode;
  }
  if (policy.fuse_prefill_ssm_post_norm_gate && decode) {
    rejected |= QwenRouteRejection::kPrefillSsmEpilogueRequiresPrefill;
  }
  const bool requested_mode_ssm_epilogue =
      decode ? policy.fuse_decode_ssm_output_residual
             : policy.fuse_prefill_ssm_post_norm_gate;
  if (requested_mode_ssm_epilogue && full_attention) {
    rejected |= QwenRouteRejection::kSsmEpilogueRequiresSsm;
  }
  if (policy.fuse_decode_rmsnorm_projection && !decode) {
    rejected |= QwenRouteRejection::kRmsNormProjectionRequiresDecode;
  }
  if (policy.prefetch_next_layer && !decode) {
    rejected |= QwenRouteRejection::kPrefetchRequiresDecode;
  }

  return {
      .plan =
          {
              .mode = mode,
              .full_attention = full_attention,
              .fuse_qk_norm_rope_kv =
                  full_attention && policy.fuse_qk_norm_rope_kv,
              .fuse_residual_rmsnorm = policy.fuse_residual_rmsnorm,
              .fuse_ffn_swiglu = decode ? policy.fuse_decode_rmsnorm_swiglu
                                        : policy.fuse_prefill_ffn_swiglu,
              .fuse_ssm_epilogue =
                  !full_attention && requested_mode_ssm_epilogue,
              .fuse_rmsnorm_projection =
                  decode && policy.fuse_decode_rmsnorm_projection,
              .prefetch_next_layer = decode && policy.prefetch_next_layer,
          },
      .rejected = rejected,
  };
}

[[nodiscard]] constexpr QwenLayerRoutePlan ResolveQwenLayerRoute(
    const QwenExecutionPolicy& policy, QwenExecutionMode mode,
    bool full_attention) noexcept {
  return ResolveQwenLayerRouteWithReasons(policy, mode, full_attention).plan;
}

/// FNV-1a-style composition helpers for a deterministic executor-local graph
/// workload identity. Callers add configuration fields and resolved route
/// fingerprints; raw addresses must never participate.
[[nodiscard]] constexpr std::uint64_t
BeginQwenGraphWorkloadIdentity() noexcept {
  return 14695981039346656037ULL;
}

[[nodiscard]] constexpr std::uint64_t ExtendQwenGraphWorkloadIdentity(
    std::uint64_t identity, std::uint64_t value) noexcept {
  return (identity ^ value) * 1099511628211ULL;
}

enum class QwenGraphRejection : std::uint32_t {
  kNone = 0,
  kLogitsNotRequested = 1U << 0U,
  kSplitKAttentionRequired = 1U << 1U,
  kLayerPrefetchEnabled = 1U << 2U,
  kGraphDisabled = 1U << 3U,
};

[[nodiscard]] constexpr QwenGraphRejection operator|(
    QwenGraphRejection lhs, QwenGraphRejection rhs) noexcept {
  return static_cast<QwenGraphRejection>(static_cast<std::uint32_t>(lhs) |
                                         static_cast<std::uint32_t>(rhs));
}

constexpr QwenGraphRejection& operator|=(QwenGraphRejection& lhs,
                                         QwenGraphRejection rhs) noexcept {
  lhs = lhs | rhs;
  return lhs;
}

[[nodiscard]] constexpr bool HasQwenGraphRejection(
    QwenGraphRejection reasons, QwenGraphRejection reason) noexcept {
  return (static_cast<std::uint32_t>(reasons) &
          static_cast<std::uint32_t>(reason)) != 0U;
}

[[nodiscard]] constexpr QwenGraphRejection ResolveQwenGraphRejections(
    bool compute_logits, bool split_k_attention, bool layer_prefetch,
    bool graph_enabled) noexcept {
  QwenGraphRejection rejected = QwenGraphRejection::kNone;
  if (!compute_logits) {
    rejected |= QwenGraphRejection::kLogitsNotRequested;
  }
  if (split_k_attention) {
    rejected |= QwenGraphRejection::kSplitKAttentionRequired;
  }
  if (layer_prefetch) {
    rejected |= QwenGraphRejection::kLayerPrefetchEnabled;
  }
  if (!graph_enabled) {
    rejected |= QwenGraphRejection::kGraphDisabled;
  }
  return rejected;
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_EXECUTION_POLICY_HPP_
