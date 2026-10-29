#ifndef GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/nemotron35_bolt/weights.hpp"

namespace gufo::models::nemotron35_bolt::rocm {

/// One tensor resident on the device, in its file format.
struct DeviceTensor {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  int rows{0};
  int cols{0};
  float scale{1.0F};  ///< NVFP4 per-tensor output scale.

  [[nodiscard]] const float* f32() const noexcept {
    return static_cast<const float*>(data);
  }
};

struct DeviceLayer {
  BlockType type{BlockType::kMamba};
  DeviceTensor norm;
  // Mamba-2.
  DeviceTensor ssm_in, ssm_conv1d, ssm_conv1d_b, ssm_dt_b, ssm_a, ssm_d,
      ssm_norm, ssm_out;
  // Attention.
  DeviceTensor attn_q, attn_k, attn_v, attn_out;
  // LatentMoE.
  int num_experts{0};
  int experts_used{0};
  int expert_ff{0};
  DeviceTensor router, router_bias, latent_down, latent_up, up_exps, down_exps,
      up_shexp, down_shexp;
};

/// Device copies of every trunk tensor. Small F32 tensors keep F32; matrices
/// keep their GGUF block format for the quantized matrix kernels.
class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const ModelWeights& weights, const core::GgufReader& reader,
      std::string* error_msg = nullptr);

  [[nodiscard]] const DeviceTensor& token_embd() const noexcept {
    return token_embd_;
  }
  [[nodiscard]] const DeviceTensor& output_norm() const noexcept {
    return output_norm_;
  }
  [[nodiscard]] const DeviceTensor& output() const noexcept { return output_; }
  [[nodiscard]] const std::vector<DeviceLayer>& layers() const noexcept {
    return layers_;
  }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept { return bytes_; }
  /// The IQ2_XXS grid as BF16 pairs for the vector products.
  [[nodiscard]] const std::uint32_t* iq2xxs_grid() const noexcept {
    return iq2xxs_grid_;
  }

private:
  DeviceModel() = default;

  std::uint32_t* iq2xxs_grid_{nullptr};
  DeviceTensor token_embd_;
  DeviceTensor output_norm_;
  DeviceTensor output_;
  std::vector<DeviceLayer> layers_;
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
};

struct DeviceDrafterLayer {
  DeviceTensor attn_norm, attn_conv_base, attn_conv_proj, attn_q, attn_k,
      attn_v, attn_q_norm, attn_k_norm, attn_out, attn_sinks, ffn_norm,
      ffn_conv_base, ffn_conv_proj, ffn_gate, ffn_up, ffn_down;
};

struct DeviceLiliCorrLayer {
  DeviceTensor attn_norm, mlp_norm, in_proj, in_proj_b, out_proj, out_proj_b,
      mlp_up, mlp_up_b, mlp_down, mlp_down_b;
};

/// Device copies of the LiLiCorr drafter, in their file formats.
struct DeviceDrafter {
  ~DeviceDrafter();
  DeviceDrafter() = default;
  DeviceDrafter(const DeviceDrafter&) = delete;
  DeviceDrafter& operator=(const DeviceDrafter&) = delete;

  [[nodiscard]] static std::unique_ptr<DeviceDrafter> Upload(
      const DrafterWeights& weights, const core::GgufReader& reader,
      std::string* error_msg = nullptr);

  DeviceTensor fc, enc_norm, output_norm;
  std::vector<DeviceDrafterLayer> layers;
  DeviceTensor token_proj, token_proj_b, pass_proj, pass_proj_b, context_proj,
      context_proj_b, feature_norm, feature_norm_b, feature_up, feature_up_b,
      feature_down, feature_down_b, slot_embd, rank_embd, attention_bias;
  std::vector<DeviceLiliCorrLayer> lc_layers;
  DeviceTensor lc_output_norm, anchor_norm, factor_in, factor_in_b, out_head,
      out_head_b, in_head, in_head_b, anchor_head, anchor_head_b;
  std::vector<void*> allocations;
  std::size_t bytes{0};
};

}  // namespace gufo::models::nemotron35_bolt::rocm

#endif  // GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_DEVICE_MODEL_HPP_
