#ifndef GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "staging/models/glm5_next/weights.hpp"

namespace gufo::models::glm5_next::rocm {

/// One weight resident in device memory, still in its GGUF encoding. The GEMV
/// tier decodes the artifact's mixed map (Q4_K/Q5_K/Q8_0 trunk, IQ2_XXS/IQ1_M
/// expert gate/up, IQ3_XXS/IQ4_XS expert down, Q2_K/Q3_K nextn) on the fly;
/// the fused operators read the F32 norms, convolutions and per-head scalars
/// directly. Nothing is dequantized at load, so the resident footprint is the
/// artifact's own.
struct DeviceTensor {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{0};
  std::uint32_t experts{1};
  std::size_t row_bytes{0};

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] const float* f32() const noexcept {
    return static_cast<const float*>(data);
  }
  /// Base address of one stacked matrix (expert or attention head).
  [[nodiscard]] const void* Expert(std::uint32_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) +
           row_bytes * static_cast<std::size_t>(rows) * e;
  }
};

/// The device-side mirror of one layer. DSA layers fill the MLA and indexer
/// fields; KDA layers fill the ssm_* fields. The dense FFN fields are present
/// on the leading blocks only, the expert trios on all other trunk blocks and
/// the nextn block. Hyper-connections wrap every block. The nextn block (the
/// last entry) additionally fills the nextn_* fields.
struct DeviceLayer {
  bool dsa{false};
  DeviceTensor hc_attn_fn, hc_attn_base, hc_attn_scale;
  DeviceTensor hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
  DeviceTensor attn_norm, ffn_norm;
  // Absorbed MLA + DSA (dsa layers). The indexer fields are set together
  // with the MLA ones.
  DeviceTensor attn_q_a, attn_q_a_norm, attn_q_b, attn_kv_a, attn_kv_a_norm,
      attn_k_b, attn_v_b, attn_out;
  DeviceTensor idx_q_b, idx_k, idx_k_norm, idx_k_bias, idx_proj, idx_comp_gate,
      idx_comp_ape;
  // KDA linear layers (non-dsa blocks).
  DeviceTensor ssm_q, ssm_k, ssm_v, ssm_conv_q, ssm_conv_k, ssm_conv_v, ssm_f_a,
      ssm_f_b, ssm_g_a, ssm_g_b, ssm_beta, ssm_a, ssm_dt, ssm_norm, ssm_out;
  // Dense FFN (leading blocks) or MoE FFN (all other blocks).
  DeviceTensor ffn_gate, ffn_up, ffn_down, router, exp_probs_b, ffn_gate_exps,
      ffn_up_exps, ffn_down_exps, shexp_gate, shexp_up, shexp_down;
  // nextn block only: fuses the previous hidden and the embedded token.
  DeviceTensor nextn_enorm, nextn_hnorm, nextn_eh_proj, nextn_shared_head_norm;
};

/// The whole artifact (45 trunk layers plus the nextn draft block at index
/// `config().num_layers`) uploaded to the GPU. Weights stay in their GGUF
/// encoding; the executor decodes them on the fly. Split artifacts arrive as
/// one reader whose mapped regions span the shards, so one upload pass
/// suffices.
class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  /// Streams every tensor from the same open file set used to bind its
  /// metadata. `layers` already contains the nextn block as its last entry.
  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const ModelWeights& weights, const core::GgufReader& reader,
      std::string* error_msg = nullptr);

  [[nodiscard]] const Config& config() const noexcept { return config_; }
  [[nodiscard]] const DeviceTensor& token_embd() const noexcept {
    return token_embd_;
  }
  [[nodiscard]] const DeviceTensor& output() const noexcept { return output_; }
  [[nodiscard]] const DeviceTensor& output_norm() const noexcept {
    return output_norm_;
  }
  [[nodiscard]] const std::vector<DeviceLayer>& layers() const noexcept {
    return layers_;
  }
  /// The self-speculative draft block (`layers_[config_.num_layers]`).
  [[nodiscard]] const DeviceLayer& nextn() const noexcept {
    return layers_[config_.num_layers];
  }
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }

private:
  DeviceModel() = default;

  Config config_;
  DeviceTensor token_embd_;
  DeviceTensor output_;
  DeviceTensor output_norm_;
  std::vector<DeviceLayer> layers_;
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
};

}  // namespace gufo::models::glm5_next::rocm

#endif  // GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_
