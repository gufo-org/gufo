#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/weights.hpp"

namespace gufo::models::gemma4::rocm {

/// One weight resident in device memory, still in its GGUF encoding.
struct DeviceTensor {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{0};
  std::uint32_t experts{1};  ///< Stacked [rows][cols] expert matrices.

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] const float* f32() const noexcept {
    return static_cast<const float*>(data);
  }
};

struct DeviceLayer {
  DeviceTensor attn_norm, attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm,
      attn_output, post_attn_norm, ffn_norm, ffn_gate, ffn_up, ffn_down,
      post_ffn_norm;
  float output_scale{1.0F};
  /// attn_q, attn_k (and attn_v) stored back to back when they share a
  /// format: one projection whose rows are [q | k | v], with the separate
  /// tensors above as views into it. Empty otherwise.
  DeviceTensor attn_qkv;
  /// ffn_gate and ffn_up back to back ([gate | up] rows), likewise.
  DeviceTensor ffn_gate_up;
  // Routed experts (see LayerWeights); empty on dense models.
  DeviceTensor router, router_scale, pre_ffn_norm_2, post_ffn_norm_1,
      post_ffn_norm_2, gate_up_exps, down_exps, down_exps_scale;
};

struct DeviceDraft {
  Config config;
  DeviceTensor pre_projection, post_projection, token_embd, output_norm;
  const float* rope_factors{nullptr};
  std::vector<DeviceLayer> layers;
};

/// The target (and optionally its MTP drafter) resident on the GPU.
/// Projection formats with a binary16-activation prefill GEMM.
[[nodiscard]] constexpr bool HalfPrefillFormat(core::GgmlType type) noexcept {
  return type == core::GgmlType::kQ4_0 || type == core::GgmlType::kQ4_K ||
         type == core::GgmlType::kQ5_K || type == core::GgmlType::kQ6_K ||
         type == core::GgmlType::kQ8_0 || type == core::GgmlType::kF16;
}

class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const ModelWeights& weights, const core::GgufReader& reader,
      const DraftWeights* draft, const core::GgufReader* draft_reader,
      std::string* error_msg = nullptr);

  [[nodiscard]] const Config& config() const noexcept { return config_; }
  [[nodiscard]] std::uint32_t vocab_size() const noexcept { return vocab_; }
  [[nodiscard]] const DeviceTensor& token_embd() const noexcept {
    return token_embd_;
  }
  [[nodiscard]] const DeviceTensor& output() const noexcept { return output_; }
  [[nodiscard]] const DeviceTensor& output_norm() const noexcept {
    return output_norm_;
  }
  /// Global-layer rope divisors, [head_dim_global / 2] on the device.
  [[nodiscard]] const float* rope_factors() const noexcept {
    return rope_factors_;
  }
  /// Leading global-layer rope pairs that turn at some position; the
  /// divisors of the others (1e30 in this checkpoint) leave them unrotated.
  [[nodiscard]] std::uint32_t global_rope_pairs() const noexcept {
    return global_rope_pairs_;
  }
  [[nodiscard]] const std::vector<DeviceLayer>& layers() const noexcept {
    return layers_;
  }
  [[nodiscard]] bool has_draft() const noexcept { return has_draft_; }
  [[nodiscard]] const DeviceDraft& draft() const noexcept { return draft_; }
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }
  /// Widest reduction dimension among the projections (activation staging).
  [[nodiscard]] std::size_t max_cols() const noexcept { return max_cols_; }
  /// Widest reduction among the binary16 dense projections (0 without any):
  /// their prefill GEMM reads binary16 activation rows.
  [[nodiscard]] std::size_t max_half_cols() const noexcept {
    return max_half_cols_;
  }
  /// Whether prefill runs on binary16 activations: expert models, and dense
  /// models whose every projection has a binary16 prefill GEMM.
  [[nodiscard]] bool half_prefill() const noexcept { return half_prefill_; }

private:
  DeviceModel() = default;

  Config config_;
  std::uint32_t vocab_{0};
  DeviceTensor token_embd_, output_, output_norm_;
  float* rope_factors_{nullptr};
  std::uint32_t global_rope_pairs_{0};
  std::vector<DeviceLayer> layers_;
  bool has_draft_{false};
  DeviceDraft draft_;
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
  std::size_t max_cols_{0};
  std::size_t max_half_cols_{0};
  bool half_prefill_{false};
};

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_DEVICE_MODEL_HPP_
