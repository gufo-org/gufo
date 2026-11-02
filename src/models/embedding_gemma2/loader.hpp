#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_LOADER_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_LOADER_HPP_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "config.hpp"
#include "tensor.hpp"

namespace gufo::models::embedding_gemma2 {

/// Tensor name prefix of the text backbone in model.safetensors. The vision
/// and audio towers share the file; unused towers are not uploaded.
inline constexpr const char* kTextTensorPrefix = "language_model.";

struct LoadResult {
  bool ok{false};
  std::string error;
  ModelConfig config;
  std::unique_ptr<TensorStore> store;
  /// Keeps the memory-mapped checkpoint alive while tensors point into it.
  std::vector<std::shared_ptr<void>> mappings;
};

/// Maps and validates the text backbone and, when requested, the vision
/// and audio towers and their projections. Unrequested towers are skipped.
[[nodiscard]] LoadResult LoadModelDirectory(const std::string& model_dir,
                                            bool include_vision = false,
                                            bool include_audio = false);
[[nodiscard]] bool LooksLikeEmbeddingGemma2(const std::string& model_dir);

struct ExpectedTensor {
  std::string name;
  std::vector<std::uint64_t> shape;
};

/// Names every tensor the text backbone needs, with its expected shape.
[[nodiscard]] std::vector<ExpectedTensor> ExpectedTextTensors(
    const TextConfig& config);

[[nodiscard]] std::vector<ExpectedTensor> ExpectedVisionTensors();
[[nodiscard]] std::vector<ExpectedTensor> ExpectedAudioTensors();

}  // namespace gufo::models::embedding_gemma2

#endif  // GUFO_MODELS_EMBEDDING_GEMMA2_LOADER_HPP_
