#ifndef GUFO_MODELS_LFM3_WEIGHTS_HPP_
#define GUFO_MODELS_LFM3_WEIGHTS_HPP_
#include <span>

#include "src/models/lfm3/config.hpp"
namespace gufo::models::lfm3 {
struct Tensor {
  const void* data{};
  core::GgmlType type{};
  std::uint32_t cols{}, rows{};
  std::size_t bytes{};
};
struct LayerWeights {
  Tensor norm, ffn_norm, post_norm, post_ffn_norm;
  Tensor gate, up, down;
  Tensor conv, in_proj, out_proj;
  Tensor q, k, v, q_norm, k_norm, attn_gate, attn_out;
};
struct Weights {
  Config config;
  Tensor embedding, output, output_norm, rope_freqs;
  std::vector<LayerWeights> layers;
  // Views borrow the GGUF reader's mapping.
  static std::optional<Weights> Bind(const core::GgufReader& reader,
                                     std::string* error = nullptr);
};
void DecodeRow(const Tensor& tensor, std::uint32_t row,
               std::span<float> output);
}  // namespace gufo::models::lfm3
#endif
