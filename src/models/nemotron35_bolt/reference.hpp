#ifndef GUFO_MODELS_NEMOTRON35_BOLT_REFERENCE_HPP_
#define GUFO_MODELS_NEMOTRON35_BOLT_REFERENCE_HPP_

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/models/nemotron35_bolt/weights.hpp"

namespace gufo::models::nemotron35_bolt {

/// Single-token, float32, scalar reference of the Nemotron 3.5 Bolt trunk.
/// It pins the semantics of every operator the GPU runtime must reproduce:
/// Mamba-2 (causal conv, selective scan, gated grouped RMSNorm), NoPE GQA,
/// and LatentMoE with sigmoid routing, a selection-only bias and ReLU^2
/// experts. The operator order follows llama.cpp's nemotron_h graph.
/// Speed is irrelevant here.
class ReferenceModel {
public:
  ReferenceModel(const ModelWeights& weights, std::uint32_t max_context);

  /// Runs one token at the next position; `logits` (vocab floats) receives
  /// the output distribution when non-empty.
  [[nodiscard]] bool Step(std::int32_t token, std::span<float> logits,
                          std::string* error_msg = nullptr);

  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  void Reset();

private:
  struct MambaState {
    std::vector<float> conv;  ///< [kernel-1][channels] raw x|B|C, oldest first
    std::vector<float> ssm;   ///< [heads][head_dim][state_size]
  };
  struct AttentionState {
    std::vector<float> k;  ///< [pos][kv_heads][head_dim]
    std::vector<float> v;  ///< [pos][kv_heads][head_dim]
  };

  void Mamba(const LayerWeights& l, MambaState& s, std::span<const float> x,
             std::span<float> out);
  void Attention(const LayerWeights& l, AttentionState& s,
                 std::span<const float> x, std::span<float> out);
  void Moe(const LayerWeights& l, std::uint32_t il, std::span<const float> x,
           std::span<float> out);

  const ModelWeights& w_;
  const Config& c_;
  std::uint32_t max_context_;
  std::uint32_t position_{0};
  std::vector<MambaState> mamba_;
  std::vector<AttentionState> attention_;
};

}  // namespace gufo::models::nemotron35_bolt

#endif  // GUFO_MODELS_NEMOTRON35_BOLT_REFERENCE_HPP_
