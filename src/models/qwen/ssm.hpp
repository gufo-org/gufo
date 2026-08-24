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

/// Computes Qwen 3.5 Gated DeltaNet linear attention operator:
/// 1. Linear QKV, Gate, Alpha, Beta projections
/// 2. 1D Causal Convolution with rolling conv state
/// 3. Associative retrieval: u = S^T * k
/// 4. Delta update: S = alpha * S + beta * (k * (v - u)^T)
/// 5. Readout: o = S^T * q
/// 6. Gated activation: y = RMSNorm(o) * SiLU(gate)
/// 7. Linear output projection to out
void ForwardSSM(std::span<const float> x_normed, const QwenLayerWeights& layer,
                const core::ModelConfig& config, QwenSsmCache& ssm_cache,
                std::uint32_t layer_idx, std::span<float> ssm_qkv_scratch,
                std::span<float> ssm_gate_scratch,
                std::span<float> ssm_out_scratch,
                std::span<float> out) noexcept;

}  // namespace strix::models

#endif  // STRIX_MODELS_QWEN_SSM_HPP_
