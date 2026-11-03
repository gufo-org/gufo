#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_REFERENCE_CPU_ENCODER_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_REFERENCE_CPU_ENCODER_HPP_

#include <cstdint>
#include <span>
#include <vector>

#include "config.hpp"
#include "tensor.hpp"

namespace gufo::models::embedding_gemma2::reference {

/// Float32 CPU implementation of the text embedding path. It is the
/// independent oracle for the HIP runtime: slow, single-threaded, and written
/// to read like the model definition.
class CpuEncoder {
public:
  /// `store` and `config` must outlive the encoder.
  CpuEncoder(const TextConfig& config, const TensorStore& store);

  /// Runs the backbone over one sequence and returns the projected token
  /// states, row-major [tokens, embedding_dim].
  [[nodiscard]] std::vector<float> EncodeTokens(
      std::span<const std::uint32_t> token_ids) const;

  /// Mean-pools the projected token states and L2-normalizes the result.
  [[nodiscard]] std::vector<float> Embed(
      std::span<const std::uint32_t> token_ids) const;

private:
  const TextConfig& config_;
  const TensorStore& store_;
};

}  // namespace gufo::models::embedding_gemma2::reference

#endif  // GUFO_MODELS_EMBEDDING_GEMMA2_REFERENCE_CPU_ENCODER_HPP_
