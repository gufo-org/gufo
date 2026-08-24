#ifndef STRIX_MODELS_QWEN_SSM_HPP_
#define STRIX_MODELS_QWEN_SSM_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "src/core/model_config.hpp"
#include "src/models/qwen/state.hpp"

namespace strix::models {

/// Gated DeltaNet recurrent state cache across sequence positions for all SSM
/// layers.
class QwenSsmCache {
public:
  QwenSsmCache(std::uint32_t num_layers, std::size_t conv_channels,
               std::uint32_t conv_kernel, std::uint32_t num_heads,
               std::uint32_t key_dim, std::uint32_t val_dim);

  void Reset() noexcept;

  /// Returns rolling conv state buffer for the given layer.
  [[nodiscard]] std::span<float> GetConvState(std::uint32_t layer) noexcept;

  /// Returns recurrent DeltaNet state matrix for given layer and head
  /// [key_dim * val_dim]
  [[nodiscard]] std::span<float> GetDeltaNetState(std::uint32_t layer,
                                                  std::uint32_t head) noexcept;

  [[nodiscard]] std::uint32_t NumHeads() const noexcept { return num_heads_; }
  [[nodiscard]] std::uint32_t KeyDim() const noexcept { return key_dim_; }
  [[nodiscard]] std::uint32_t ValDim() const noexcept { return val_dim_; }

  [[nodiscard]] std::uint32_t NumLayers() const noexcept { return num_layers_; }
  [[nodiscard]] std::size_t ConvChannels() const noexcept {
    return conv_channels_;
  }
  [[nodiscard]] std::uint32_t ConvKernel() const noexcept {
    return conv_kernel_;
  }

private:
  std::uint32_t num_layers_;
  std::size_t conv_channels_;
  std::uint32_t conv_kernel_;
  std::uint32_t num_heads_;
  std::uint32_t key_dim_;
  std::uint32_t val_dim_;
  std::vector<float> conv_states_;
  std::vector<float> deltanet_states_;
};

/// Complete non-owning tensor and shape contract for one Qwen SSM layer.
/// The nine tensor references intentionally match the SSM subset of
/// QwenLayerWeights; dimensions make the slice independently executable.
struct QwenSsmParameters {
  QwenTensorRef qkv;
  QwenTensorRef gate;
  QwenTensorRef a;
  QwenTensorRef dt;
  QwenTensorRef alpha;
  QwenTensorRef beta;
  QwenTensorRef norm;
  QwenTensorRef output;
  QwenTensorRef conv1d;
  std::uint32_t key_head_count = 0;
  std::uint32_t value_head_count = 0;
  std::uint32_t key_dim = 0;
  std::uint32_t val_dim = 0;
  std::uint32_t conv_kernel = 0;
};

[[nodiscard]] inline QwenSsmParameters MakeQwenSsmParameters(
    const QwenLayerWeights& layer, const core::ModelConfig& config) noexcept {
  return {
      .qkv = layer.attn_qkv,
      .gate = layer.attn_gate,
      .a = layer.ssm_a,
      .dt = layer.ssm_dt,
      .alpha = layer.ssm_alpha,
      .beta = layer.ssm_beta,
      .norm = layer.ssm_norm,
      .output = layer.ssm_out,
      .conv1d = layer.ssm_conv1d,
      .key_head_count = config.ssm_group_count,
      .value_head_count = config.ssm_time_step_rank,
      .key_dim = config.ssm_state_size,
      .val_dim = config.SsmValueSize(),
      .conv_kernel = config.ssm_conv_kernel,
  };
}

/// Computes Qwen 3.5 Gated DeltaNet linear attention from a self-contained
/// tensor slice. Invalid shapes safely zero-fill `out` without touching state.
void ForwardSSM(std::span<const float> x_normed,
                const QwenSsmParameters& parameters, QwenSsmCache& ssm_cache,
                std::uint32_t layer_idx, std::span<float> ssm_qkv_scratch,
                std::span<float> ssm_gate_scratch,
                std::span<float> ssm_out_scratch,
                std::span<float> out) noexcept;

/// Compatibility entry point for whole-layer production callers. It constructs
/// QwenSsmParameters and delegates to the slice overload above.
void ForwardSSM(std::span<const float> x_normed, const QwenLayerWeights& layer,
                const core::ModelConfig& config, QwenSsmCache& ssm_cache,
                std::uint32_t layer_idx, std::span<float> ssm_qkv_scratch,
                std::span<float> ssm_gate_scratch,
                std::span<float> ssm_out_scratch,
                std::span<float> out) noexcept;

}  // namespace strix::models

#endif  // STRIX_MODELS_QWEN_SSM_HPP_
